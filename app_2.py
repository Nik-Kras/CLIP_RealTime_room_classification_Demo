import os
import time
import math
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

# ---- CLIP: use the 'clip-anytorch' package (works on MPS/CPU/CUDA) ----
#   pip install clip-anytorch
import clip

# ----------------------------
# Device selection (favor MPS)
# ----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# ----------------------------
# Prompt generation
# ----------------------------
ROOMS = ["kitchen", "bathroom", "living room", "bedroom", "outside / garden"]

def generate_room_prompts():
    """
    Generate 50 diverse zero-shot prompts per room class.
    We mix photography styles, contexts, synonyms, and descriptors.
    """
    styles = [
        "a photo of", "an indoor photo of", "a candid snapshot of", "a wide-angle shot of",
        "a realistic picture of", "a detailed image of", "a natural light photo of",
        "a smartphone photo of", "a DSLR photograph of", "a softly lit photo of",
        "a high-contrast photo of", "a documentary-style photo of", "a lifestyle photo of",
        "a minimalist photo of", "a cozy photo of", "a bright daylight photo of",
        "a dimly lit photo of", "a professional real-estate photo of", "an interior design photo of",
        "a clean modern photo of", "a slightly messy photo of", "a wide shot of", "a close-up scene of",
        "a panorama of", "a vignette of", "a still image of", "a natural scene of",
        "a real-world photo of", "a 35mm film photo of", "a point-and-shoot photo of",
        "a neutral color photo of", "a warm color photo of", "a cool color photo of",
        "a well-lit photo of", "a grainy photo of", "a crisp photo of", "a handheld photo of",
        "an editorial photo of", "a catalog photo of", "a documentary photo of",
        "a candid interior of", "a lived-in interior of", "a tidy interior of",
        "a staged interior of", "a daytime scene of", "an evening scene of",
        "a morning scene of", "a night scene of", "a naturalistic photo of", "a realistic scene of",
    ]

    # Room-specific synonyms/phrases
    variants = {
        "kitchen": [
            "a kitchen", "a home kitchen", "a modern kitchen", "a rustic kitchen",
            "a small apartment kitchen", "a spacious kitchen", "a kitchen with cabinets and countertop",
            "a kitchen with a sink and stove", "a kitchen with an oven and fridge",
            "a kitchen island", "a kitchen with cookware", "a kitchen with dishes",
        ],
        "bathroom": [
            "a bathroom", "a home bathroom", "a modern bathroom", "a tiled bathroom",
            "a small apartment bathroom", "a spacious bathroom",
            "a bathroom with a sink and mirror", "a bathroom with a toilet and shower",
            "a bathroom vanity", "a bathtub and shower", "a bathroom with towels",
            "a bathroom with toiletries",
        ],
        "living room": [
            "a living room", "a lounge", "a sitting room", "a family room",
            "a modern living room", "a cozy living room", "a minimalist living room",
            "a living room with a sofa and coffee table", "a living room with a TV",
            "a living room with a rug and lamp", "a living room with bookshelves",
            "a living room with wall art",
        ],
        "bedroom": [
            "a bedroom", "a master bedroom", "a guest bedroom", "a small bedroom",
            "a cozy bedroom", "a minimalist bedroom", "a bedroom with a bed and nightstand",
            "a bedroom with pillows and duvet", "a bedroom with a wardrobe",
            "a bedroom with a desk", "a bedroom with a lamp", "a bedroom with curtains",
        ],
        "outside / garden": [
            "a garden", "a backyard", "an outdoor patio", "a lawn with plants",
            "a flower garden", "a vegetable garden", "a terrace", "a balcony garden",
            "a park-like yard", "an outdoor seating area", "a garden with trees and shrubs",
            "a garden path",
        ],
    }

    # Short context phrases that bias towards typical objects/fixtures
    anchors = {
        "kitchen": ["stove", "oven", "fridge", "countertop", "kitchen sink", "cabinetry", "cookware", "cutting board"],
        "bathroom": ["toilet", "shower", "bathtub", "mirror", "sink", "towels", "tile walls", "toiletries"],
        "living room": ["sofa", "couch", "coffee table", "TV", "bookshelf", "lamps", "rug", "wall art"],
        "bedroom": ["bed", "pillow", "duvet", "nightstand", "wardrobe", "dresser", "lamp", "curtains"],
        "outside / garden": ["grass", "plants", "flowers", "trees", "hedges", "outdoor furniture", "path", "fence"],
    }

    per_class = {}
    for room in ROOMS:
        prompts = set()
        # Build combinations until we have 50 unique prompts
        for s in styles:
            for v in variants[room]:
                # Basic
                prompts.add(f"{s} {v}.")
                # With anchors
                for a in anchors[room]:
                    prompts.add(f"{s} {v} featuring {a}.")
                    if len(prompts) >= 50:
                        break
                if len(prompts) >= 50:
                    break
            if len(prompts) >= 50:
                break

        # If somehow under 50, pad with generic phrasing variations
        base = list(prompts)
        i = 0
        while len(prompts) < 50:
            prompts.add(f"a realistic photo of {variants[room][i % len(variants[room])]}.")
            i += 1

        per_class[room] = list(prompts)[:50]
    return per_class

ROOM_PROMPTS = generate_room_prompts()

# Build a flattened list and index ranges per class for fast aggregation
ALL_TEXTS = []
CLASS_RANGES = {}  # room -> (start, end)
cursor = 0
for room in ROOMS:
    start = cursor
    ALL_TEXTS.extend(ROOM_PROMPTS[room])
    cursor += len(ROOM_PROMPTS[room])
    CLASS_RANGES[room] = (start, cursor)

# ----------------------------
# Load CLIP
# ----------------------------
MODEL_NAME = "ViT-B/32"  # good speed/accuracy tradeoff
model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
model.eval()
torch.set_grad_enabled(False)

# Tokenize & encode all text prompts once
with torch.no_grad():
    text_tokens = clip.tokenize(ALL_TEXTS)  # shape [250, 77]
text_tokens = text_tokens.to(DEVICE)
with torch.no_grad():
    text_embeds = model.encode_text(text_tokens).float()
text_embeds = F.normalize(text_embeds, dim=-1)  # cosine-ready

# ----------------------------
# Inference / Scoring helpers
# ----------------------------
TOPK_PER_CLASS = 5     # aggregate the top-5 prompt matches per class
EMA_ALPHA = 0.2        # temporal smoothing factor for class scores

class TemporalState:
    def __init__(self):
        self.ema_scores = None  # torch tensor [num_classes]
        self.last_time = None
        self.fps = 0.0

    def update_fps(self):
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                # smooth fps
                instant = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * instant if self.fps > 0 else instant
        self.last_time = now

STATE = TemporalState()

# Preallocate class order list for stable output
CLASS_ORDER = ROOMS[:]

def score_frame(np_frame):
    """
    np_frame: uint8 HxWxC (RGB) from Gradio webcam
    returns: (overlayed_frame, top_label, confidence)
    """
    STATE.update_fps()

    pil = Image.fromarray(np_frame)
    img = preprocess(pil).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    with torch.no_grad():
        image_embed = model.encode_image(img).float()
    image_embed = F.normalize(image_embed, dim=-1)  # [1,512]

    # Cosine similarities with all prompts: [1, 250]
    sims = (image_embed @ text_embeds.T).squeeze(0)  # [250]

    # Aggregate per class
    class_scores = []
    for room in CLASS_ORDER:
        s, e = CLASS_RANGES[room]
        room_sims = sims[s:e]  # [50]
        topk = torch.topk(room_sims, k=min(TOPK_PER_CLASS, room_sims.shape[0])).values
        class_scores.append(topk.mean())
    class_scores = torch.stack(class_scores)  # [5]

    # EMA smoothing across frames
    if STATE.ema_scores is None:
        STATE.ema_scores = class_scores
    else:
        STATE.ema_scores = (1 - EMA_ALPHA) * STATE.ema_scores + EMA_ALPHA * class_scores

    # Softmax to get a confidence-like value (temperature for calibration if desired)
    probs = torch.softmax(STATE.ema_scores * 10.0, dim=-1)  # sharpen a bit
    top_idx = int(torch.argmax(probs).item())
    top_label = CLASS_ORDER[top_idx]
    confidence = float(probs[top_idx].item())

    # Draw overlay
    overlay = np_frame.copy()
    h, w, _ = overlay.shape
    label_text = f"{top_label.upper()}  |  conf: {confidence:.2f}  |  FPS: {STATE.fps:.1f}"

    # background rectangle for readability
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Measure text box
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    box_w, box_h = tw + 2*pad, th + 2*pad
    x, y = 20, 30  # top-left anchor
    cv2.rectangle(overlay, (x-5, y-25), (x-5 + box_w + 10, y-25 + box_h + 10), (0, 0, 0), -1)
    cv2.putText(overlay, label_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay, top_label, confidence

# ----------------------------
# Gradio app
# ----------------------------
def process_frame(frame):
    """
    Gradio streams webcam frames (RGB numpy array).
    We return the annotated frame and textual outputs.
    """
    if frame is None:
        return None, "", 0.0
    overlay, label, conf = score_frame(frame)
    return overlay, label, conf

with gr.Blocks(title="Room Detector (CLIP Zero-Shot)") as demo:
    gr.Markdown(
        """
        # 🔎 Real-Time Room Detector (CLIP Zero-Shot)
        Streams from your webcam and predicts whether you're in a **kitchen, bathroom, living room, bedroom, or outside/garden**.
        """
    )

    with gr.Row():
        cam = gr.Image(
            label="Webcam",
            sources=["webcam"],
            streaming=True,
            mirror_webcam=True,
            height=420,
        )
        out = gr.Image(label="Annotated Stream", height=420)
    with gr.Row():
        room_lbl = gr.Label(label="Predicted Room", num_top_classes=1)
        conf_num = gr.Number(label="Confidence", precision=3)

    cam.stream(process_frame, [cam], [out, room_lbl, conf_num])

    gr.Markdown(
        "Model: **CLIP ViT-B/32** • Aggregation: **top-5 prompt mean per class** • Temporal smoothing: **EMA α=0.2**"
    )

if __name__ == "__main__":
    # On Mac with M3, MPS should be auto-picked; if you want to force CPU:
    #   export PYTORCH_ENABLE_MPS_FALLBACK=1
    #   set DEVICE to 'cpu'
    demo.launch(server_name="0.0.0.0", server_port=7860)
