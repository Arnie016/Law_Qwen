# Video Generation Models - Text to Video

How it works: You give a text prompt → Model generates video frames → Saves as MP4/GIF

## How Video Generation Works

**Two approaches:**

1. **Text → Image → Video** (2-step):
   - First: Generate image with Stable Diffusion
   - Then: Animate image with Stable Video Diffusion

2. **Text → Video** (direct):
   - Model generates video directly from text
   - Examples: AnimateDiff, ModelScope, Open-Sora

---

## Models You Can Download

### 1. Stable Video Diffusion (Image-to-Video) ⭐ **RECOMMENDED**

**Best quality, works reliably**

```bash
# Install
pip install diffusers transformers accelerate

# Download and use
python3 << EOF
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import torch

# Step 1: Generate image first (or use existing)
from diffusers import StableDiffusionPipeline
pipe_img = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe_img("a cat walking on the street").images[0]
image.save("base_image.png")

# Step 2: Animate the image
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.bfloat16
).to("cuda")

video_frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
# Save as GIF or video
EOF
```

**Model:** `stabilityai/stable-video-diffusion-img2vid-xt`
- **Downloads:** 97.9K | **Likes:** 3,180
- **Size:** ~3-4GB
- **Link:** https://hf.co/stabilityai/stable-video-diffusion-img2vid-xt

---

### 2. AnimateDiff (Text-to-Video)

**Direct text-to-video generation**

```bash
pip install diffusers transformers accelerate

python3 << EOF
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.utils import export_to_video
import torch

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "frankjoshua/toonyou_beta6",
    motion_adapter=adapter,
    torch_dtype=torch.bfloat16
).to("cuda")

video = pipe("a cat walking on the street", num_frames=16).frames[0]
export_to_video(video, "output.mp4")
EOF
```

**Models needed:**
- Motion adapter: `guoyww/animatediff-motion-adapter-v1-5-2`
- Base model: `frankjoshua/toonyou_beta6` (or other compatible models)

---

### 3. ModelScope (Text-to-Video)

**Chinese model, good quality**

```bash
pip install modelscope diffusers

python3 << EOF
from modelscope import snapshot_download
from diffusers import DiffusionPipeline
import torch

model_dir = snapshot_download("AI-ModelScope/stable-video-diffusion-img2vid-xt")
pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# Use similar to Stable Video Diffusion
EOF
```

---

### 4. Open-Sora (Text-to-Video, Experimental)

**Open source Sora alternative**

```bash
# Requires custom installation
# See: https://github.com/hpcaitech/Open-Sora
git clone https://github.com/hpcaitech/Open-Sora.git
cd Open-Sora
pip install -e .
```

**Note:** May need more setup, but produces good results

---

## Quick Start: Two-Step Process (Easiest)

**Step 1: Generate image**
```bash
python3 << EOF
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe("a cat walking on the street").images[0]
image.save("base_image.png")
EOF
```

**Step 2: Animate image**
```bash
python3 << EOF
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.bfloat16
).to("cuda")

image = Image.open("base_image.png")
video_frames = pipe(image, num_frames=25).frames[0]

# Save as GIF
from PIL import Image
video_frames[0].save("output.gif", save_all=True, append_images=video_frames[1:], duration=100, loop=0)
EOF
```

---

## Model Comparison

| Model | Type | Size | Quality | Speed |
|-------|------|------|---------|-------|
| **Stable Video Diffusion XT** | Image→Video | ~4GB | ⭐⭐⭐⭐⭐ | Fast |
| **AnimateDiff** | Text→Video | ~2-3GB | ⭐⭐⭐⭐ | Medium |
| **Open-Sora** | Text→Video | ~5-10GB | ⭐⭐⭐⭐⭐ | Slow |

---

## Recommended: Start with Stable Video Diffusion

**Why:**
- Most reliable
- Best quality
- Works with ROCm
- Easy to use

**Quick command:**
```bash
pip install diffusers transformers accelerate pillow

# Then use the code above
```

---

## Save Video Output

```python
# Save as GIF
from PIL import Image
frames[0].save("output.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

# Save as MP4 (requires imageio-ffmpeg)
pip install imageio imageio-ffmpeg
import imageio
imageio.mimwrite("output.mp4", frames, fps=8)
```

---

## Notes

- Video generation uses more GPU memory than images
- With 8x MI300X (1.5TB GPU memory), you can generate long videos
- Each frame takes ~1-2 seconds on GPU
- 25 frames = ~25-50 seconds of generation time


