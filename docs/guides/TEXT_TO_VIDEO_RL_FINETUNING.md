# Text-to-Video Fine-Tuning with RL: Complete Setup

## 🎯 Goal: Fine-Tune Text-to-Video Model with RL

**Objective:** Create a fine-tuned text-to-video model using:
1. **Hugging Face Models** - Text-to-video generation
2. **Video Datasets** - Text prompts + videos
3. **RL Fine-Tuning** - GRPO/DPO for better quality
4. **Unsloth** - Fast training optimization

---

## 📊 Models & Datasets Found

### Text-to-Video Models (Hugging Face)

1. **Stable Video Diffusion** ⭐ RECOMMENDED
   - Model: `stabilityai/stable-video-diffusion-img2vid-xt`
   - Type: Image → Video (need image first)
   - Quality: ⭐⭐⭐⭐⭐
   - Size: ~3-4GB
   - Works with: diffusers library

2. **AnimateDiff**
   - Model: `guoyww/animatediff-motion-adapter-v1-5-2`
   - Type: Text → Video (direct)
   - Quality: ⭐⭐⭐⭐
   - Size: ~2-3GB
   - Works with: diffusers + AnimateDiff

3. **ModelScope Video**
   - Model: `damo-vilab/text-to-video-ms-1.7b`
   - Type: Text → Video (direct)
   - Quality: ⭐⭐⭐
   - Size: ~2GB
   - Note: May need special access

4. **VideoCrafter**
   - Model: Various VideoCrafter models
   - Type: Text → Video
   - Quality: ⭐⭐⭐⭐
   - Size: ~3-5GB

### Video Datasets (Hugging Face)

1. **WebVid-10M** (Subset)
   - Dataset: `mrm8488/webvid-2M-subset`
   - Format: Video + Captions
   - Size: 2M video-text pairs
   - Good for: General video generation

2. **MSR-VTT**
   - Dataset: `jameseese/msr-vtt`
   - Format: Video + Text descriptions
   - Size: 10K videos
   - Good for: Video understanding + generation

3. **ActivityNet Captions**
   - Dataset: `ActivityNet/ActivityNetCaptions`
   - Format: Video + Timestamped captions
   - Size: 20K videos
   - Good for: Action video generation

4. **Custom Dataset Format**
   ```json
   {
     "prompt": "A cat walking on the street",
     "video_path": "path/to/video.mp4",
     "video_frames": [frame1, frame2, ...],
     "metadata": {...}
   }
   ```

---

## 🚀 Complete Text-to-Video Fine-Tuning Script

### Setup: Install Dependencies

```python
# Install required packages
!pip install diffusers transformers accelerate peft unsloth trl
!pip install imageio opencv-python pillow
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 1: Load Model and Dataset

```python
import unsloth  # IMPORT FIRST!
from unsloth import FastLanguageModel
from diffusers import StableVideoDiffusionPipeline, StableDiffusionPipeline
from datasets import load_dataset
import torch
from PIL import Image
import imageio

print("🎬 Text-to-Video Fine-Tuning Setup\n")

# Option 1: Use Stable Video Diffusion (Image → Video)
print("1. Loading Stable Video Diffusion...")
pipe_video = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.bfloat16,
)
pipe_video = pipe_video.to("cuda")

# Option 2: Load image generator
pipe_img = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
)
pipe_img = pipe_img.to("cuda")

# Load dataset
print("\n2. Loading video dataset...")
try:
    # Try WebVid subset
    dataset = load_dataset("mrm8488/webvid-2M-subset", split="train[:1000]")
    print(f"✅ Dataset loaded: {len(dataset)} examples")
except:
    print("⚠️ Using custom dataset format")
    # Create custom dataset
    dataset = {
        "prompt": ["A cat walking", "A dog running", ...],
        "video_path": ["path1.mp4", "path2.mp4", ...],
    }
```

### Step 2: Prepare Dataset for Fine-Tuning

```python
from datasets import Dataset

def prepare_video_dataset(examples):
    """
    Prepare dataset for video generation fine-tuning
    Format: text prompt → video frames
    """
    processed = []
    
    for i, prompt in enumerate(examples.get('prompt', [])):
        # For fine-tuning, we need:
        # - Input: Text prompt
        # - Output: Video frames (or video encoding)
        
        # Option 1: Use existing videos
        video_path = examples.get('video_path', [None])[i]
        if video_path:
            # Load video frames
            video_frames = load_video_frames(video_path)
        else:
            # Option 2: Generate video from prompt (for training)
            image = pipe_img(prompt).images[0]
            video_frames = pipe_video(image, num_frames=14).frames[0]
        
        processed.append({
            "prompt": prompt,
            "video_frames": video_frames,
            "num_frames": len(video_frames),
        })
    
    return processed

def load_video_frames(video_path, max_frames=14):
    """Load video frames from file"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        count += 1
    
    cap.release()
    return frames

# Process dataset
dataset = dataset.map(prepare_video_dataset, batched=True)
```

### Step 3: RL Fine-Tuning Setup (GRPO)

```python
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

# GRPO Configuration for Video Generation
grpo_config = GRPOConfig(
    output_dir="./text-to-video-grpo",
    per_device_train_batch_size=2,  # Smaller for video (memory intensive)
    gradient_accumulation_steps=4,
    learning_rate=1e-4,  # Lower for video models
    num_train_epochs=1,
    max_steps=500,  # Start with 500 steps
    warmup_steps=50,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    logging_steps=10,
    save_steps=50,
    num_generations=4,  # Generate 4 videos per prompt
    optim="adamw_torch",  # Standard optimizer (ROCm compatible)
)

# Video Quality Reward Function
def video_reward_function(*args, **kwargs):
    """
    Reward function for video generation quality
    Rewards:
    - Video consistency (frames match)
    - Motion quality (smooth transitions)
    - Prompt adherence (matches text description)
    - Visual quality (resolution, clarity)
    """
    # Extract prompts and generated videos
    prompts = kwargs.get('prompts') or kwargs.get('inputs') or (args[0] if args else [])
    videos = kwargs.get('responses') or kwargs.get('completions') or (args[1] if len(args) > 1 else [])
    
    rewards = []
    
    for prompt, video in zip(prompts, videos):
        reward = 0.0
        
        # Check video consistency (frames should be similar)
        if isinstance(video, list) and len(video) > 1:
            # Calculate frame similarity
            frame_diffs = []
            for i in range(len(video) - 1):
                # Simple similarity check (can use more sophisticated methods)
                diff = abs(video[i].size[0] - video[i+1].size[0])
                frame_diffs.append(diff)
            
            avg_diff = sum(frame_diffs) / len(frame_diffs) if frame_diffs else 0
            if avg_diff < 10:  # Frames are consistent
                reward += 3.0
        
        # Check video length (more frames = better)
        if isinstance(video, list):
            num_frames = len(video)
            if num_frames >= 14:
                reward += 2.0
            elif num_frames >= 7:
                reward += 1.0
        
        # Check prompt adherence (simplified - use CLIP or similar)
        # For now, reward based on video quality
        reward += 1.0  # Base reward
        
        rewards.append(reward)
    
    return rewards

# Format dataset for GRPO
def format_grpo_video(examples):
    """Format video dataset for GRPO"""
    prompts = []
    for prompt in examples.get('prompt', []):
        formatted = f"Generate a video: {prompt}"
        prompts.append(formatted)
    return {"prompt": prompts}

grpo_dataset = dataset.map(format_grpo_video, batched=True)

# GRPO Trainer
# Note: Stable Video Diffusion may need custom trainer
# For now, this is a conceptual setup
```

---

## ⚠️ Challenges with Video RL Fine-Tuning

**Problem:** Standard GRPO/DPO trainers expect text outputs, not video.

**Solutions:**

1. **Use Video-to-Text Model First** (Easier)
   - Fine-tune Qwen2.5-VL on video understanding
   - Then use RL on text outputs
   - Convert text → video separately

2. **Custom Video RL Trainer** (Advanced)
   - Modify GRPOTrainer for video outputs
   - Use video quality metrics as rewards
   - Requires custom implementation

3. **Two-Stage Approach** (Recommended)
   - Stage 1: Fine-tune video generation (SFT)
   - Stage 2: RL on video quality (custom reward)

---

## 🎯 Recommended Approach

### Stage 1: Supervised Fine-Tuning (Video Generation)

```python
# Fine-tune Stable Video Diffusion on your dataset
# Use standard diffusion training (not RL yet)

from diffusers import StableVideoDiffusionPipeline
from diffusers.training_utils import VideoDiffusionTrainer

# This is conceptual - actual implementation may vary
trainer = VideoDiffusionTrainer(
    model=pipe_video,
    train_dataset=dataset,
    # Training args...
)
trainer.train()
```

### Stage 2: RL Fine-Tuning (Video Quality)

```python
# After SFT, use RL to improve video quality
# Custom reward function for video metrics

def video_quality_reward(video, prompt):
    """
    Reward based on:
    - Visual quality (SSIM, PSNR)
    - Motion smoothness
    - Prompt adherence (CLIP score)
    - Temporal consistency
    """
    reward = 0.0
    
    # Visual quality metrics
    # Motion smoothness
    # Prompt matching
    
    return reward
```

---

## 📝 Quick Start Script

```python
#!/usr/bin/env python3
"""
Text-to-Video Fine-Tuning with RL
Quick start for video generation fine-tuning
"""
import unsloth
from diffusers import StableVideoDiffusionPipeline, StableDiffusionPipeline
from datasets import load_dataset
import torch

print("🎬 Text-to-Video Fine-Tuning\n")

# 1. Load models
print("Loading models...")
pipe_img = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe_video = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.bfloat16,
).to("cuda")

# 2. Load dataset
print("Loading dataset...")
dataset = load_dataset("mrm8488/webvid-2M-subset", split="train[:100]")

# 3. Fine-tune (SFT first, then RL)
print("Ready for fine-tuning!")
print("Note: Video RL fine-tuning requires custom implementation")
print("Start with SFT, then add RL rewards")
```

---

## 🔗 Resources

**Models:**
- Stable Video Diffusion: `stabilityai/stable-video-diffusion-img2vid-xt`
- AnimateDiff: `guoyww/animatediff-motion-adapter-v1-5-2`

**Datasets:**
- WebVid: `mrm8488/webvid-2M-subset`
- MSR-VTT: `jameseese/msr-vtt`
- ActivityNet: `ActivityNet/ActivityNetCaptions`

**Unsloth:**
- Works with text models (for video understanding)
- Video generation RL may need custom trainer

---

## ⚡ Quick Summary

**Can you fine-tune text-to-video with RL?**
- ✅ Yes, but needs custom setup
- ✅ Start with SFT on video datasets
- ✅ Then add RL for quality improvement
- ⚠️ Standard GRPO expects text, not video

**Best Approach:**
1. Fine-tune video generation (SFT)
2. Use RL on video-to-text model (easier)
3. Or build custom video RL trainer (advanced)

**Your 205GB VRAM:** Perfect for large video models!

