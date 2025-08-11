from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

# --- Optional compatibility patch for older gradio_client on Python 3.9/3.10 ---
try:
    import gradio_client.utils as _gcu  # type: ignore
    _orig_get_type = getattr(_gcu, "get_type", None)
    _orig_json_to_py = getattr(_gcu, "_json_schema_to_python_type", None)

    def _safe_get_type(schema):  # type: ignore
        if not isinstance(schema, dict):
            return "any"
        return _orig_get_type(schema) if _orig_get_type else "any"

    def _safe_json_schema_to_python_type(schema, defs=None):  # type: ignore
        if isinstance(schema, bool):
            return "dict[str, any]" if schema else "dict[str, never]"
        if isinstance(schema, dict):
            ap = schema.get("additionalProperties", None)
            if isinstance(ap, bool):
                schema = dict(schema)
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
    pass
# -------------------------------------------------------------------------------

try:
    import clip  # OpenAI CLIP (https://github.com/openai/CLIP)
except ImportError:
    raise SystemExit("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

ROOMS = ["kitchen", "bathroom", "living room", "bedroom", "outside"]

def generate_prompts() -> Dict[str, List[str]]:
    base_templates = [
        "a photo of a {}", "an indoor scene of a {}", "this looks like a {}",
        "a wide-angle shot of a {}", "a smartphone picture of a {}",
        "a well-lit {}", "a dimly lit {}", "a messy {}", "a clean {}",
        "a modern {}", "a cozy {}", "a minimalist {}", "an everyday {}",
        "a typical {}", "a realistic {}", "a high-resolution {}",
        "a documentary photo of a {}", "a candid shot of a {}",
        "a room that is a {}", "the interior of a {}", "the inside of a {}",
        "a space that appears to be a {}", "a snapshot of a {}",
        "a still image of a {}", "a view of a {}", "a {} with furniture",
        "a {} with people", "an empty {}", "a daylight {}", "a nighttime {}",
        "a cluttered {}", "a spacious {}", "a small {}", "a bright {}",
        "a dark {}", "a warm-toned {}", "a cool-toned {}", "an HDR photo of a {}",
        "an aesthetic {}", "a snapshot on social media of a {}", "a vlog frame inside a {}",
        "a first-person view of a {}", "a security camera frame of a {}",
        "a GoPro frame of a {}", "a webcam frame of a {}",
        "a 35mm film photo of a {}", "a Polaroid of a {}", "a candid home photo of a {}",
        "an architectural photo of a {}", "an example of a {}",
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

AVAILABLE_MODELS = [
    "ViT-B/32",        # fast
    "ViT-B/16",
    "ViT-L/14",        # strong
    "ViT-L/14@336px",  # strongest (slower, higher-res)
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
    text_features: torch.Tensor
    prompt_index_to_room: List[str]
    rooms: List[str]
    model_name: str
    use_fp16: bool

def _encode_text(model, tokens, use_fp16: bool):
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_fp16):
            feats = model.encode_text(tokens)
    else:
        feats = model.encode_text(tokens)
    return feats

def _encode_image(model, img_tensor, use_fp16: bool):
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_fp16):
            feats = model.encode_image(img_tensor)
    else:
        feats = model.encode_image(img_tensor)
    return feats

def build_clip(model_name: str, use_fp16: bool) -> ClipContext:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
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
        text_features = _encode_text(model, text_tokens, use_fp16=use_fp16)
        text_features = F.normalize(text_features.float(), dim=-1)  # keep as float32 for stability

    return ClipContext(
        device=device,
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        prompt_index_to_room=prompt_index_to_room,
        rooms=ROOMS,
        model_name=model_name,
        use_fp16=use_fp16,
    )

CTX: ClipContext | None = None

@torch.no_grad()
def classify_frame(image: Image.Image) -> Tuple[str, float]:
    global CTX
    assert CTX is not None, "Model not initialized"
    img_input = CTX.preprocess(image).unsqueeze(0).to(CTX.device)
    image_features = _encode_image(CTX.model, img_input, use_fp16=CTX.use_fp16)
    image_features = F.normalize(image_features.float(), dim=-1)
    sims = image_features @ CTX.text_features.T
    sims = sims.squeeze(0)
    best_per_room: Dict[str, float] = {room: -1e9 for room in CTX.rooms}
    for idx, room in enumerate(CTX.prompt_index_to_room):
        val = float(sims[idx].item())
        if val > best_per_room[room]:
            best_per_room[room] = val
    room, _ = max(best_per_room.items(), key=lambda kv: kv[1])
    scores = torch.tensor(list(best_per_room.values()))
    conf = torch.softmax(scores, dim=0)[CTX.rooms.index(room)].item()
    return room, float(conf)

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

def startup(selected_model: str, use_fp16: bool) -> str:
    global CTX
    t0 = time.time()
    CTX = build_clip(selected_model, use_fp16=use_fp16)
    t1 = time.time()
    try:
        import gradio_client as _grc
        _client_ver = _grc.__version__
    except Exception:
        _client_ver = "(no gradio_client)"
    return (
        f"Model: {CTX.model_name} on {CTX.device}\n"
        f"FP16: {CTX.use_fp16}\n"
        f"Text embeddings: {CTX.text_features.shape[0]} prompts\n"
        f"Init: {t1 - t0:.1f}s\n"
        f"gradio {gr.__version__} / gradio_client {_client_ver}"
    )

def reload_model(selected_model: str, use_fp16: bool) -> str:
    return startup(selected_model, use_fp16)

def process_frame(frame: np.ndarray, threshold_percent: float) -> Tuple[np.ndarray, str]:
    if frame is None:
        return None, ""
    pil = Image.fromarray(frame.astype(np.uint8))
    room, conf = classify_frame(pil)
    threshold = float(threshold_percent) / 100.0
    if conf >= threshold:
        annotated = draw_label(pil, room, conf)
        label_out = f"{room} ({conf*100:.1f}%)"
    else:
        annotated = pil
        label_out = f"below {threshold_percent:.0f}% threshold ({conf*100:.1f}%)"
    return np.array(annotated), label_out

with gr.Blocks(css=".label-box {font-size: 1.2rem; font-weight: 600}") as demo:
    gr.Markdown("# 🏠 Real-time Room Detector (CLIP Zero-Shot) — GPU Ready")

    with gr.Row():
        model_select = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="ViT-L/14@336px",  # default to strongest
            label="CLIP model (stronger ⇒ slower)"
        )
        fp16_toggle = gr.Checkbox(
            value=torch.cuda.is_available(),  # default to True if CUDA
            label="Use FP16 (GPU only)"
        )
        threshold_slider = gr.Slider(
            minimum=0, maximum=100, step=1, value=40,
            label="Confidence threshold (%) — show label only above this (random = 20%)"
        )
    status = gr.Markdown("Initializing…")

    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, mirror_webcam=True, label="Webcam", height=420)
        out_img = gr.Image(label="Annotated Stream", height=420)
    label_text = gr.Textbox(label="Predicted Room", interactive=False, elem_classes=["label-box"])

    cam.stream(process_frame, inputs=[cam, threshold_slider], outputs=[out_img, label_text], concurrency_limit=2)
    demo.load(fn=startup, inputs=[model_select, fp16_toggle], outputs=status)
    model_select.change(fn=reload_model, inputs=[model_select, fp16_toggle], outputs=status)
    fp16_toggle.change(fn=reload_model, inputs=[model_select, fp16_toggle], outputs=status)

if __name__ == "__main__":
    # Server deploy defaults (no public tunneling):
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, max_threads=8)
