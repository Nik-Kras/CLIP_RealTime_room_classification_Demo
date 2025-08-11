#!/usr/bin/env python3
"""
Gradio UI: Real-time Room Classification from Webcam using OpenAI CLIP (zero-shot)

macOS + Python 3.9 fixes included:
1) **Gradio schema workaround**: Patch gradio_client schema helpers to tolerate boolean schemas
   (avoids `APIInfoParseError: Cannot parse schema True`).
2) **Localhost**: Launch with `share=True` so you always get a public URL.
3) **Model selector**: Choose a stronger/weaker CLIP backbone at runtime (e.g., ViT-B/32 vs ViT-L/14@336px).

Install (in your venv):
  pip install -U "gradio==4.44.1" "gradio_client==1.3.0" fastapi uvicorn starlette anyio
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install git+https://github.com/openai/CLIP.git
  pip install pillow

Run:
  python app.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# WORKAROUND: Some gradio/gradio_client versions (esp. on Py 3.9) choke when
# parsing JSON Schema fields like additionalProperties=True/False.
# We defensively patch the relevant helpers to accept boolean schemas.
# ---------------------------------------------------------------------------
try:
    import gradio_client.utils as _gcu  # type: ignore
    _orig_get_type = getattr(_gcu, "get_type", None)
    _orig_json_to_py = getattr(_gcu, "_json_schema_to_python_type", None)

    def _safe_get_type(schema):  # type: ignore
        if not isinstance(schema, dict):
            return "any"
        return _orig_get_type(schema) if _orig_get_type else "any"

    def _safe_json_schema_to_python_type(schema, defs=None):  # type: ignore
        # Handle booleans directly
        if isinstance(schema, bool):
            return "dict[str, any]" if schema else "dict[str, never]"
        # Normalize additionalProperties when it's a boolean
        if isinstance(schema, dict):
            ap = schema.get("additionalProperties", None)
            if isinstance(ap, bool):
                schema = dict(schema)  # shallow copy
                # Replace True with an empty subschema, False with a null-like schema
                schema["additionalProperties"] = {} if ap else {"type": "null"}
        try:
            if _orig_json_to_py:
                return _orig_json_to_py(schema, defs)
        except Exception:
            pass
        return "any"

    if _orig_get_type:
        _gcu.get_type = _safe_get_type  # type: ignore
    if _orig_json_to_py:
        _gcu._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore
except Exception:
    # If patch fails, continue; newer clients won't need it.
    pass

try:
    import clip  # OpenAI CLIP
except ImportError:
    raise SystemExit("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

# ----------------------------
# Zero-shot prompt generation
# ----------------------------
ROOMS = [
    "kitchen",
    "bathroom",
    "living room",
    "bedroom",
    "outside",
]


def generate_prompts() -> Dict[str, List[str]]:
    """Generate ~50 natural-language prompts per room class."""
    base_templates = [
        "a photo of a {}",
        "an indoor scene of a {}",
        "this looks like a {}",
        "a wide-angle shot of a {}",
        "a smartphone picture of a {}",
        "a well-lit {}",
        "a dimly lit {}",
        "a messy {}",
        "a clean {}",
        "a modern {}",
        "a cozy {}",
        "a minimalist {}",
        "an everyday {}",
        "a typical {}",
        "a realistic {}",
        "a high-resolution {}",
        "a documentary photo of a {}",
        "a candid shot of a {}",
        "a room that is a {}",
        "the interior of a {}",
        "the inside of a {}",
        "a space that appears to be a {}",
        "a snapshot of a {}",
        "a still image of a {}",
        "a view of a {}",
        "a {} with furniture",
        "a {} with people",
        "an empty {}",
        "a daylight {}",
        "a nighttime {}",
        "a cluttered {}",
        "a spacious {}",
        "a small {}",
        "a bright {}",
        "a dark {}",
        "a warm-toned {}",
        "a cool-toned {}",
        "an HDR photo of a {}",
        "an aesthetic {}",
        "a snapshot on social media of a {}",
        "a vlog frame inside a {}",
        "a first-person view of a {}",
        "a security camera frame of a {}",
        "a GoPro frame of a {}",
        "a webcam frame of a {}",
        "a 35mm film photo of a {}",
        "a Polaroid of a {}",
        "a candid home photo of a {}",
        "an architectural photo of a {}",
        "an example of a {}",
    ]

    variants = {
        "kitchen": ["kitchen", "home kitchen", "domestic kitchen", "cooking area", "kitchen space"],
        "bathroom": ["bathroom", "washroom", "restroom", "toilet room", "bathroom interior"],
        "living room": ["living room", "lounge", "sitting room", "family room", "living area"],
        "bedroom": ["bedroom", "sleeping room", "master bedroom", "guest bedroom", "bedroom interior"],
        "outside": ["garden", "backyard", "outdoors", "patio", "outside area"],
    }

    prompts: Dict[str, List[str]] = {}
    for room in ROOMS:
        room_prompts = set()
        for t in base_templates:
            for v in variants[room]:
                room_prompts.add(t.format(v))
                if len(room_prompts) >= 50:
                    break
            if len(room_prompts) >= 50:
                break
        i = 1
        while len(room_prompts) < 50:
            room_prompts.add(f"a photo that looks like a {room} ({i})")
            i += 1
        prompts[room] = list(room_prompts)[:50]
    return prompts


# ----------------------------
# CLIP model setup
# ----------------------------
AVAILABLE_MODELS = [
    "ViT-B/32",         # fast, light
    "ViT-B/16",
    "ViT-L/14",         # stronger
    "ViT-L/14@336px",   # strongest (slower, higher-res)
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
]

@dataclass
class ClipContext:
    device: torch.device
    model: torch.nn.Module
    preprocess: any
    text_features: torch.Tensor  # shape (N_prompts, D)
    prompt_index_to_room: List[str]  # len N_prompts
    rooms: List[str]
    model_name: str


def build_clip(model_name: str) -> ClipContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    prompts_by_room = generate_prompts()

    all_prompts: List[str] = []
    prompt_index_to_room: List[str] = []
    for room, plist in prompts_by_room.items():
        for p in plist:
            all_prompts.append(p)
            prompt_index_to_room.append(room)

    with torch.no_grad():
        text_tokens = clip.tokenize(all_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)

    return ClipContext(
        device=device,
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        prompt_index_to_room=prompt_index_to_room,
        rooms=ROOMS,
        model_name=model_name,
    )


CTX: ClipContext | None = None


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def classify_frame(image: Image.Image) -> Tuple[str, float]:
    global CTX
    assert CTX is not None, "Model not initialized"

    img_input = CTX.preprocess(image).unsqueeze(0).to(CTX.device)
    image_features = CTX.model.encode_image(img_input)
    image_features = F.normalize(image_features.float(), dim=-1)

    sims = image_features @ CTX.text_features.T
    sims = sims.squeeze(0)

    best_per_room: Dict[str, float] = {room: -1e9 for room in CTX.rooms}
    for idx, room in enumerate(CTX.prompt_index_to_room):
        val = float(sims[idx].item())
        if val > best_per_room[room]:
            best_per_room[room] = val

    room, score = max(best_per_room.items(), key=lambda kv: kv[1])

    scores = torch.tensor(list(best_per_room.values()))
    conf = torch.softmax(scores, dim=0)[CTX.rooms.index(room)].item()
    return room, float(conf)


# ----------------------------
# Overlay helpers
# ----------------------------
FONT = None

def get_font(size: int = 32):
    global FONT
    if FONT is None:
        try:
            FONT = ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            FONT = ImageFont.load_default()
    return FONT


def draw_label(image: Image.Image, label: str, confidence: float) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    text = f"{label.upper()}  {confidence*100:4.1f}%"
    font = get_font(max(24, img.width // 30))

    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 12
    box = (10, 10, 10 + (bbox[2]-bbox[0]) + pad*2, 10 + (bbox[3]-bbox[1]) + pad*2)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle(box, radius=12, fill=(0, 0, 0, 160))

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((box[0] + pad, box[1] + pad), text, font=font, fill=(255, 255, 255))
    return img


# ----------------------------
# Gradio App
# ----------------------------

def startup(selected_model: str) -> str:
    """Build (or rebuild) the CLIP context with the selected model."""
    global CTX
    t0 = time.time()
    CTX = build_clip(selected_model)
    t1 = time.time()

    import gradio as _gr
    try:
        import gradio_client as _grc
        _client_ver = _grc.__version__
    except Exception:
        _client_ver = "(no gradio_client)"

    return (
        f"Model: {CTX.model_name} on {CTX.device}\n"
        f"Text embeddings: {CTX.text_features.shape[0]} prompts\n"
        f"Init: {t1 - t0:.1f}s\n"
        f"Gradio {_gr.__version__} / gradio_client {_client_ver}"
    )


def reload_model(selected_model: str) -> str:
    return startup(selected_model)


def process_frame(frame: np.ndarray) -> Tuple[np.ndarray, str]:
    if frame is None:
        return None, ""
    pil = Image.fromarray(frame.astype(np.uint8))
    room, conf = classify_frame(pil)
    annotated = draw_label(pil, room, conf)
    return np.array(annotated), f"{room} ({conf*100:.1f}%)"


with gr.Blocks(css=".label-box {font-size: 1.2rem; font-weight: 600}") as demo:
    gr.Markdown("# 🏠 Real-time Room Detector (CLIP Zero-Shot)\nChoose a model, allow webcam, and watch the label update live.")

    with gr.Row():
        model_select = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value='ViT-L/14', # "ViT-B/32",
            label="CLIP model (stronger ⇒ slower)",
        )
        status = gr.Markdown("Initializing…")

    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, mirror_webcam=True, label="Webcam", height=420)
        out_img = gr.Image(label="Annotated Stream", height=420)
    label_text = gr.Textbox(label="Predicted Room", interactive=False, elem_classes=["label-box"])

    # Wiring
    cam.stream(process_frame, inputs=cam, outputs=[out_img, label_text], concurrency_limit=2)
    demo.load(fn=startup, inputs=model_select, outputs=status)
    model_select.change(fn=reload_model, inputs=model_select, outputs=status)

if __name__ == "__main__":
    # Force a shareable link to avoid localhost/proxy issues on some macOS setups
    demo.queue().launch(share=True, max_threads=4)  # queue enables smooth streaming
