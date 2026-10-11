# 🎥 Video Production Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DubeyDivyansh/DubeyDivyansh/blob/main/quick_demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/target_GPU-NVIDIA_Tesla_T4_(16GB)-green.svg)](https://cloud.google.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

A modular, production-ready AI video generation engine designed specifically to run with **zero cloud costs** on **Google Colab's Free NVIDIA Tesla T4 GPU (16GB VRAM)** with full **Google Drive persistence**.

---

## 🌟 Architecture & Features

This engine cleanly separates business logic and heavy neural network pipelines into testable Python modules:

```
video_production_engine/
├── GEMINI.md                          # Machine-readable agent directive
├── README.md                          # Engine documentation & usage guide
├── requirements.txt                   # Production dependencies
└── src/
    ├── audio/                         # Stage 1: Zero-shot voice cloning
    │   └── f5_engine.py               # F5-TTS flow matching wrapper
    ├── video/                         # Stage 2: Animation & lip-sync
    │   ├── sadtalker_engine.py        # 3DMM portrait animation + GFPGAN
    │   └── latentsync_engine.py       # Latent diffusion lip-sync (whole body)
    ├── utils/                         # Core infrastructure & guards
    │   ├── memory.py                  # VRAM memory fence & allocator
    │   ├── storage.py                 # Google Drive persistence & caches
    │   └── media.py                   # FFmpeg slicers & audio extractors
    └── pipeline.py                    # Unified 1-line production orchestrator
```

---

## 🚀 Key Innovations for Colab T4

1. **Tesla T4 (sm_75) FP16 Patch:**
   * LatentSync default code checks `torch.cuda.get_device_capability()[0] > 7`. On T4 (`7.5`), this evaluates to `False`, erroneously forcing FP32 (>20GB OOM).
   * Our engine automatically patches this to `>= 7`, enforcing `torch.float16` and keeping inference under **13.5 GB VRAM**.
2. **Deterministic VRAM Fencing:**
   * Runs `flush_vram()` between audio and video stages, dropping residual memory to `< 500 MB`.
3. **Google Drive Checkpoint Persistence:**
   * All heavy model checkpoints live in `/content/drive/MyDrive/video_gen_platform/checkpoints/` so you **never re-download 6GB+ weights** across sessions.
4. **Thin Notebook Interface:**
   * Colab is simply the runner. Logic lives in clean, version-controlled `.py` modules.

---

## 💻 Quickstart in Google Colab

Inside your Colab notebook, execute the full pipeline in just 4 clean lines:

```python
# 1. Clone repository & install dependencies
!git clone https://github.com/DubeyDivyansh/DubeyDivyansh.git /content/app
%cd /content/app
!pip install -q -r video_production_engine/requirements.txt

# 2. Import the production pipeline
import sys
sys.path.append("/content/app/video_production_engine")
from src.pipeline import VideoProductionPipeline

# 3. Initialize with Google Drive persistence
pipeline = VideoProductionPipeline(use_drive=True)

# 4. Generate your personalized video!
video_path = pipeline.produce_video(
    face_image_path="/content/custom_face.jpeg",
    ref_audio_path="/content/custom_audio.wav",
    script_text="Hello! Welcome to our personalized video generation platform.",
    output_name="my_first_avatar.mp4",
    engine="sadtalker"  # or 'latentsync' for full-body video
)
```

---

## 📊 VRAM Budget on Tesla T4

| Stage | Module | Peak VRAM | Status |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Voice)** | F5-TTS | **~2.8 GB** | ✅ Safe |
| **Fence** | `flush_vram()` | **< 0.5 GB** | 🧹 Reset |
| **Stage 2 (Video)** | SadTalker + GFPGAN | **~4.5 GB** | ✅ Safe |
| **Stage 2 (Alt)** | LatentSync (FP16) | **~12.2 GB** | ✅ Fits in 15.3GB |
