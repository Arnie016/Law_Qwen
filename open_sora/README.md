# Open-Sora Quickstart

Text-to-video generation using Open-Sora (or compatible models) on AMD MI300X.

## Installation

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

cd /path/to/open_sora
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 quickstart.py --prompt "A cat walking on the street" --frames 16 --output output.gif
```

## Open-Sora Setup

Open-Sora requires custom installation. See: https://github.com/hpcaitech/Open-Sora

### Alternative: Use Available Video Models

If Open-Sora isn't available, use compatible models:

1. **AnimateDiff**: Text-to-video via motion adapter
2. **ModelScope**: Video generation models
3. **Stable Video Diffusion**: Image-to-video

### Example with AnimateDiff

```python
from diffusers import AnimateDiffPipeline, MotionAdapter

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "frankjoshua/toonyou_beta6",
    motion_adapter=adapter,
    torch_dtype=torch.bfloat16
).to("cuda")

video = pipe(prompt, num_inference_steps=25).frames
```

## Notes

- Current implementation uses placeholder (Stable Diffusion for frames)
- Replace with actual Open-Sora inference when model is available
- ROCm-compatible operations
- Adjust model paths based on availability


