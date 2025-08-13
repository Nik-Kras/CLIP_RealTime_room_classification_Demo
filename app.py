from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import pandas as pd

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

# ========================= Sophisticated prompt engineering =========================
TARGET_PROMPTS_PER_ROOM = 90  # set between 50–100 if you want to tune

ROOM_ALIASES = {
    "kitchen": ["kitchen", "home kitchen", "domestic kitchen", "cooking area", "galley kitchen"],
    "bathroom": ["bathroom", "washroom", "restroom", "toilet room", "lavatory"],
    "living room": ["living room", "lounge", "sitting room", "family room", "living area"],
    "bedroom": ["bedroom", "sleeping room", "master bedroom", "guest bedroom"],
    "outside": ["garden", "backyard", "outdoors", "patio", "yard"],
}

ROOM_OBJECTS = {
    "kitchen": [
        "stove", "gas stove", "induction cooktop", "oven", "microwave",
        "range hood", "kettle", "toaster", "blender", "fridge", "refrigerator",
        "kitchen island", "countertop", "cutting board", "knife block",
        "pots and pans", "spice rack", "dish rack", "kitchen cabinets",
        "kitchen sink", "faucet", "dish soap", "paper towel holder",
        "bar stools", "backsplash", "pantry shelves",
    ],
    "bathroom": [
        "toilet", "toilet seat", "toilet paper holder",
        "bathroom sink", "vanity", "mirror", "medicine cabinet",
        "shower", "glass shower door", "bathtub", "shower curtain",
        "towel rack", "toothbrush", "toothpaste", "soap dispenser",
        "bath mat", "tiles", "hair dryer", "razor",
    ],
    "living room": [
        "sofa", "couch", "sectional", "armchair",
        "coffee table", "TV", "television", "TV stand", "remote control",
        "bookshelf", "floor lamp", "table lamp", "throw pillow",
        "rug", "side table", "wall art", "soundbar",
        "game console", "indoor plant", "fireplace", "mantel", "curtains",
    ],
    "bedroom": [
        "bed", "pillow", "blanket", "duvet", "bedsheet",
        "nightstand", "bedside lamp", "alarm clock",
        "wardrobe", "closet", "dresser", "mirror",
        "headboard", "vanity table", "laundry basket",
        "crib", "bunk bed",
    ],
    "outside": [
        "grass", "lawn", "trees", "bushes", "fence",
        "patio furniture", "outdoor table", "barbecue", "grill",
        "deck", "balcony", "flower bed", "garden hose",
        "shed", "pathway", "stones", "planter box", "swing", "trampoline",
    ],
}

ROOM_ACTIVITIES = {
    "kitchen": [
        "cooking dinner", "chopping vegetables", "boiling water",
        "washing dishes", "prepping food", "brewing coffee",
    ],
    "bathroom": [
        "brushing teeth", "washing hands", "taking a shower",
        "applying makeup", "shaving", "using the toilet",
    ],
    "living room": [
        "watching TV", "playing video games", "reading on the sofa",
        "relaxing on the couch", "hosting guests",
    ],
    "bedroom": [
        "sleeping", "making the bed", "reading in bed",
        "getting dressed", "folding clothes",
    ],
    "outside": [
        "gardening", "barbecuing", "sitting on patio furniture",
        "watering plants", "mowing the lawn",
    ],
}

FRAMES = [
    "close-up", "cropped", "partial view", "corner of the room",
    "top-down", "low angle", "high angle", "over-the-shoulder",
    "through a doorway", "in a mirror reflection", "wide-angle",
    "smartphone photo", "security camera frame", "GoPro frame",
    "dim lighting", "bright daylight", "warm-toned light", "cool-toned light",
]

BASE_TEMPLATES = [
    "a photo of a {alias}",
    "the interior of a {alias}",
    "an everyday {alias}",
    "a realistic {alias}",
    "a {alias} with typical furniture",
]

OBJECT_TEMPLATES = [
    "a {alias} with a {obj}",
    "a {frame} of a {obj} in a {alias}",
    "POV in the {alias} looking at the {obj}",
    "a {alias} countertop showing a {obj}",
    "a {obj} next to a wall in a {alias}",
    "the corner of a {alias} showing a {obj}",
]

ACTIVITY_TEMPLATES = [
    "a {alias} where someone is {act}",
    "a {frame} showing {act} in the {alias}",
    "signs of {act} in a {alias}",
]

DISAMBIG_SPECIALS = [
    "{alias} with a bathroom sink",
    "{alias} with a kitchen sink",
    "{alias} showing a toilet seat",
    "{alias} showing a bed",
    "{alias} showing a sofa",
]

def _mk_room_aliases(room: str) -> List[str]:
    return ROOM_ALIASES.get(room, [room])

def _prioritized_prompts_for_room(room: str) -> List[str]:
    aliases = _mk_room_aliases(room)
    objs = ROOM_OBJECTS.get(room, [])
    acts = ROOM_ACTIVITIES.get(room, [])

    out: List[str] = []
    seen = set()

    def add(fmt: str, **kw):
        s = fmt.format(**kw)
        if s not in seen:
            seen.add(s)
            out.append(s)

    # 1) Strong class-name cues
    for alias in aliases:
        for t in BASE_TEMPLATES:
            add(t, alias=alias)

    # 2) Disambiguation specials
    for alias in aliases:
        for t in DISAMBIG_SPECIALS:
            add(t, alias=alias)

    # 3) Object-centric + limited frame combos
    for alias in aliases:
        for obj in objs:
            add("a {alias} containing a {obj}", alias=alias, obj=obj)
            for t in OBJECT_TEMPLATES:
                for frame in FRAMES[:6]:
                    add(t, alias=alias, obj=obj, frame=frame)

    # 4) Activity-centric + limited frame combos
    for alias in aliases:
        for act in acts:
            for t in ACTIVITY_TEMPLATES:
                for frame in FRAMES[:6]:
                    add(t, alias=alias, act=act, frame=frame)

    # 5) Frame-only variants (extreme crops)
    for alias in aliases:
        for frame in FRAMES:
            add("a {frame} of a {alias}", alias=alias, frame=frame)

    return out

def generate_prompts() -> Dict[str, List[str]]:
    prompts_by_room: Dict[str, List[str]] = {}
    for room in ROOMS:
        p = _prioritized_prompts_for_room(room)
        head = p[:40]
        tail = p[40:]
        random.seed(42)
        random.shuffle(tail)
        capped = (head + tail)[:TARGET_PROMPTS_PER_ROOM]
        if len(capped) < 50:
            capped = (capped + p[len(capped):])[:50]
        prompts_by_room[room] = capped
    return prompts_by_room
# ======================= /Sophisticated prompt engineering =======================

AVAILABLE_MODELS = [
    "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px",
    "RN50", "RN101", "RN50x4", "RN50x16",
]

# -------- object cues (short, high-signal) ----------
OBJECT_CUES = {
    "kitchen": ["stove", "oven", "range hood", "kitchen sink", "fridge", "microwave",
                "cutting board", "knife block", "pots and pans", "spice rack"],
    "bathroom": ["toilet", "toilet seat", "bathroom sink", "shower", "bathtub",
                 "toothbrush", "soap dispenser", "towel rack"],
    "living room": ["sofa", "couch", "coffee table", "TV", "bookshelf", "floor lamp",
                    "throw pillow", "rug", "soundbar"],
    "bedroom": ["bed", "headboard", "pillow", "blanket", "nightstand", "wardrobe",
                "dresser", "bedside lamp"],
    "outside": ["grass", "lawn", "trees", "patio furniture", "barbecue", "garden hose",
                "fence", "flower bed"],
}
# ----------------------------------------------------

@dataclass
class ClipContext:
    device: torch.device
    model: torch.nn.Module
    preprocess: any
    text_features: torch.Tensor               # engineered prompts
    prompt_index_to_room: List[str]
    prompts_all: List[str]
    rooms: List[str]
    model_name: str
    use_fp16: bool
    cue_features_by_room: Dict[str, torch.Tensor]  # per-room object-cue embeddings

def _encode_text(model, tokens, use_fp16: bool):
    if torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            feats = model.encode_text(tokens)
    else:
        feats = model.encode_text(tokens)
    return feats

def _encode_image(model, img_tensor, use_fp16: bool):
    if torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
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
    all_prompts, prompt_index_to_room = [], []
    for room, plist in prompts_by_room.items():
        for p in plist:
            all_prompts.append(p)
            prompt_index_to_room.append(room)

    with torch.no_grad():
        text_tokens = clip.tokenize(all_prompts).to(device)
        text_features = _encode_text(model, text_tokens, use_fp16=use_fp16)
        text_features = F.normalize(text_features.float(), dim=-1)

        cue_features_by_room: Dict[str, torch.Tensor] = {}
        for room, cue_list in OBJECT_CUES.items():
            toks = clip.tokenize(cue_list).to(device)
            feats = _encode_text(model, toks, use_fp16=use_fp16)
            cue_features_by_room[room] = F.normalize(feats.float(), dim=-1)

    return ClipContext(
        device=device,
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        prompt_index_to_room=prompt_index_to_room,
        prompts_all=all_prompts,
        rooms=ROOMS,
        model_name=model_name,
        use_fp16=use_fp16,
        cue_features_by_room=cue_features_by_room,
    )

# -------- lazy init guard ----------
CTX: ClipContext | None = None
DEFAULT_MODEL = "ViT-L/14@336px"
DEFAULT_FP16 = torch.cuda.is_available()
CURRENT_MODEL = DEFAULT_MODEL
CURRENT_FP16 = DEFAULT_FP16
FRAME_IDX = 0
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def ensure_ctx():
    global CTX
    if CTX is None:
        CTX = build_clip(CURRENT_MODEL, use_fp16=CURRENT_FP16)
# -----------------------------------

@torch.no_grad()
def classify_frame_with_debug(image: Image.Image) -> Tuple[str, float, np.ndarray, List[str], List[str], np.ndarray]:
    """
    Returns:
      room, conf,
      avg_sims_full (N_prompts,),
      prompts_all (list of N strings),
      prompt_rooms (list of N room labels),
      class_probs (len(ROOMS),) final per-class probabilities
    """
    global CTX
    ensure_ctx()

    # 5-crop to handle partial views
    w, h = image.size
    crop_w, crop_h = int(0.75 * w), int(0.75 * h)
    crops = [
        image.crop((0, 0, crop_w, crop_h)),
        image.crop((w - crop_w, 0, w, crop_h)),
        image.crop((0, h - crop_h, crop_w, h)),
        image.crop((w - crop_w, h - crop_h, w, h)),
    ]
    cx0, cy0 = (w - crop_w) // 2, (h - crop_h) // 2
    crops.append(image.crop((cx0, cy0, cx0 + crop_w, cy0 + crop_h)))

    per_crop_prompt_sims = []
    per_crop_best_full_by_room = []

    # per-crop compute similarities
    for crop in crops:
        img_input = CTX.preprocess(crop).unsqueeze(0).to(CTX.device)
        img_feats = _encode_image(CTX.model, img_input, use_fp16=CTX.use_fp16)
        img_feats = F.normalize(img_feats.float(), dim=-1)

        sims_full = (img_feats @ CTX.text_features.T).squeeze(0)  # (N_prompts,)
        per_crop_prompt_sims.append(sims_full.cpu())

        best_full = {room: -1e9 for room in CTX.rooms}
        for idx, room in enumerate(CTX.prompt_index_to_room):
            s = float(sims_full[idx].item())
            if s > best_full[room]:
                best_full[room] = s
        per_crop_best_full_by_room.append(best_full)

    # average per-prompt similarities across crops
    avg_sims_full = torch.stack(per_crop_prompt_sims, dim=0).mean(dim=0)  # (N_prompts,)

    # average "best full" across crops
    best_full_avg = {room: np.mean([d[room] for d in per_crop_best_full_by_room]) for room in CTX.rooms}

    # object cue scores (best cue over room) averaged across crops
    cue_scores = {}
    for room in CTX.rooms:
        crop_vals = []
        for crop in crops:
            img_input = CTX.preprocess(crop).unsqueeze(0).to(CTX.device)
            img_feats = _encode_image(CTX.model, img_input, use_fp16=CTX.use_fp16)
            img_feats = F.normalize(img_feats.float(), dim=-1)
            sims_cues = (img_feats @ CTX.cue_features_by_room[room].T).squeeze(0)
            crop_vals.append(float(sims_cues.max().item()))
        cue_scores[room] = float(np.mean(crop_vals))

    # weighted mix + temperature softmax
    name_weight, cue_weight = 0.4, 0.6
    mix_scores = torch.tensor([name_weight * best_full_avg[r] + cue_weight * cue_scores[r] for r in CTX.rooms])
    T = 0.5
    class_probs = torch.softmax(mix_scores / T, dim=0)
    best_idx = int(torch.argmax(class_probs).item())
    room = CTX.rooms[best_idx]
    conf = float(class_probs[best_idx].item())

    # convert for logging
    prompts_all = CTX.prompts_all
    prompt_rooms = CTX.prompt_index_to_room
    return room, conf, avg_sims_full.numpy(), prompts_all, prompt_rooms, class_probs.cpu().numpy()

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

def _save_debug(image: Image.Image,
                avg_sims_full: np.ndarray,
                prompts_all: List[str],
                prompt_rooms: List[str],
                class_probs: np.ndarray):
    """Write JPEG + per-prompt CSV into data/."""
    global FRAME_IDX
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_IDX += 1
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"frame_{ts}_{FRAME_IDX:06d}"
    img_path = DATA_DIR / f"{stem}.jpg"
    csv_path = DATA_DIR / f"{stem}.csv"

    image.convert("RGB").save(img_path, quality=90)

    # map room -> prob for quick lookup
    room_to_prob = {room: float(class_probs[i]) for i, room in enumerate(ROOMS)}

    df = pd.DataFrame({
        "prompt": prompts_all,
        "score": avg_sims_full.astype(float),
        "room": prompt_rooms,
        "class_prob": [room_to_prob[r] for r in prompt_rooms],
    })
    df.to_csv(csv_path, index=False)

def startup(selected_model: str, use_fp16: bool) -> str:
    global CTX, CURRENT_MODEL, CURRENT_FP16
    CURRENT_MODEL = selected_model
    CURRENT_FP16 = use_fp16
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

def process_frame(frame: np.ndarray, threshold_percent: float, save_debug: bool) -> Tuple[np.ndarray, str]:
    if frame is None:
        return None, ""
    ensure_ctx()
    pil = Image.fromarray(frame.astype(np.uint8))

    room, conf, avg_sims_full, prompts_all, prompt_rooms, class_probs = classify_frame_with_debug(pil)

    if save_debug:
        try:
            _save_debug(pil, avg_sims_full, prompts_all, prompt_rooms, class_probs)
        except Exception as e:
            # keep streaming even if disk write fails
            print(f"[debug-save] {e}")

    threshold = float(threshold_percent) / 100.0
    if conf >= threshold:
        annotated = draw_label(pil, room, conf)
        label_out = f"{room} ({conf*100:.1f}%)"
    else:
        annotated = pil
        label_out = f"below {threshold_percent:.0f}% threshold ({conf*100:.1f}%)"
    return np.array(annotated), label_out

# -------------------------- UI --------------------------
with gr.Blocks(css=".label-box {font-size: 1.2rem; font-weight: 600}") as demo:
    gr.Markdown("# 🏠 Real-time Room Detector (CLIP Zero-Shot) — GPU Ready")

    with gr.Row():
        model_select = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="ViT-B/32", # "ViT-L/14@336px",
            label="CLIP model (stronger ⇒ slower)"
        )
        fp16_toggle = gr.Checkbox(
            value=torch.cuda.is_available(),
            label="Use FP16 (GPU only)"
        )
        threshold_slider = gr.Slider(
            minimum=0, maximum=100, step=1, value=40,
            label="Confidence threshold (%) — show label only above this (random = 20%)"
        )
        save_debug_checkbox = gr.Checkbox(
            value=True,
            label="Save per-frame debug logs (images + CSV to ./data)"
        )
    status = gr.Markdown("Initializing…")
    demo.load(fn=startup, inputs=[model_select, fp16_toggle], outputs=status)

    with gr.Row():
        try:
            cam = gr.Image(
                sources=["webcam"],
                streaming=True,
                webcam_options=gr.WebcamOptions(mirror=True),
                label="Webcam",
                height=420
            )
        except AttributeError:
            # Older Gradio fallback
            cam = gr.Image(
                sources=["webcam"],
                streaming=True,
                mirror_webcam=True,
                label="Webcam",
                height=420
            )
        out_img = gr.Image(label="Annotated Stream", height=420)
    label_text = gr.Textbox(label="Predicted Room", interactive=False, elem_classes=["label-box"])

    cam.stream(
        process_frame,
        inputs=[cam, threshold_slider, save_debug_checkbox],
        outputs=[out_img, label_text],
        concurrency_limit=2
    )
    model_select.change(fn=reload_model, inputs=[model_select, fp16_toggle], outputs=status)
    fp16_toggle.change(fn=reload_model, inputs=[model_select, fp16_toggle], outputs=status)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, max_threads=8)
