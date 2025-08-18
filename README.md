# Real-time Room Detector (CLIP Zero-Shot) — GPU/Server Deploy

Detect the room type (kitchen, bathroom, living room, bedroom, outside) from a live webcam stream using OpenAI CLIP zero-shot classification. Optimized to take advantage of **NVIDIA RTX 4090** with **FP16**.

## 📽 Demo Video
[![Watch the demo](https://img.youtube.com/vi/--gecpzyppg/maxresdefault.jpg)](https://youtu.be/--gecpzyppg?si=yVkUI7idJDEwjv1V)

## 1) Server prerequisites

- Ubuntu 22.04/24.04 (recommended) or similar Linux with an NVIDIA GPU (e.g., RTX 4090).
- Recent **NVIDIA driver** installed (e.g., 535+).
- **Python 3.10+** recommended (3.9 works with the included compatibility patch).
- Optional but recommended: `ffmpeg` for webcam/video support.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev ffmpeg
```

## 2) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Install CUDA-enabled PyTorch

Pick the latest CUDA wheel for your server from the official PyTorch site. On recent systems with CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> If cu124 isn’t available on your system, try `cu121` instead by swapping the index URL.

## 4) Install the rest

```bash
pip install -U gradio gradio_client fastapi uvicorn starlette anyio pillow
pip install git+https://github.com/openai/CLIP.git
```

> If you are constrained to Python 3.9 and run into `gradio_client` schema errors, pin to:
> `pip install "gradio==4.44.1" "gradio_client==1.3.0"`
> The app includes a small compatibility patch that avoids the crash.

## 5) Run

```bash
python app.py
```

You’ll see logs like:

```
Running on local URL:  http://0.0.0.0:7860
```

Open `http://<server-ip>:7860` in your browser.

### Webcam notes
- Browser access to a *server-side* webcam is usually not available.
- For remote demos, open the app in a local browser with a **local webcam** (e.g., via reverse proxy or SSH tunnel), or switch the UI to accept a video file / RTSP stream as input. (We can add that input mode on request.)

## 6) Using GPU + the strongest model

- In the UI:
  - **Model:** defaults to **`ViT-L/14@336px`** (most accurate, slower).
  - **Use FP16:** enabled by default on CUDA (uses `torch.cuda.amp.autocast` for big speedups).
  - **Confidence threshold:** default **40%** (hide label below threshold).
- You can switch models on the fly; the app re-embeds prompts automatically.

## 7) Production tips

- Set a fixed port and bind to all interfaces (already done):  
  `launch(server_name="0.0.0.0", server_port=7860, share=False)`
- Put the app behind a reverse proxy (NGINX/Caddy) with HTTPS for client demos.
- Monitor GPU use: `watch -n 1 nvidia-smi`.
- If you want higher throughput, reduce resolution in the webcam source or switch to `ViT-L/14` (non-336) or `ViT-B/16`.

## 8) Optional: RTSP / file input

If you need to analyze a security camera or an uploaded file instead of a webcam:
- Replace the `gr.Image(... sources=["webcam"], streaming=True ...)` with a `gr.Video` component.
- Decode frames with `cv2.VideoCapture` for RTSP and feed them into the same `process_frame()` function.
- I can provide this variant on request.

---

### Troubleshooting

- **Schema error from `gradio_client`** on Python 3.9:
  - The app already patches around it.
  - Alternatively, upgrade to Python 3.10+ and latest `gradio` + `gradio_client`.

- **Slow init** with `ViT-L/14@336px`:
  - It’s the heaviest model; first-time load downloads weights. Subsequent runs are faster.

- **CUDA not found**:
  - Verify `torch.cuda.is_available()` in Python.
  - Ensure you installed the CUDA-enabled wheels (step 3) and that the NVIDIA driver is present.
