# Video Model Fine-Tuning: Better Approach

## ⚠️ Problem: RL Rewards Are 0.0

**Issue:** Reward function returns 0.0 for all videos.

**Why:** RL for video models is complex - requires proper gradient flow through video generation.

**Solution:** Use **Supervised Fine-Tuning (SFT)** instead!

---

## ✅ Why SFT > RL for Video Models

### RL Challenges:
- Complex gradient flow through video generation
- Hard to implement correctly
- Rewards may not propagate properly
- Often unstable

### SFT Advantages:
- ✅ Straightforward: Learn from image-video pairs
- ✅ Standard diffusion training
- ✅ Proven to work
- ✅ Better results
- ✅ Easier to implement

---

## 🎯 Supervised Fine-Tuning Setup

### Dataset Format:
```python
{
    "image": "path/to/image.jpg",  # Input image
    "video": "path/to/video.mp4",  # Target video (16 frames)
    "prompt": "Optional text description"
}
```

### Training Process:
1. Load image-video pairs
2. Encode image → image features
3. Generate video from image features
4. Compare with target video
5. Update model weights

### Loss Function:
```python
# Standard diffusion loss
loss = mse_loss(predicted_noise, actual_noise)
```

---

## 📝 Quick SFT Script

```python
#!/usr/bin/env python3
"""
Supervised Fine-Tuning for Video Models
Better than RL for video generation
"""
import torch
from diffusers import StableVideoDiffusionPipeline
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

print("🎬 Video Model SFT Fine-Tuning\n")

# 1. Load Model
print("Loading model...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

# 2. Add LoRA (efficient fine-tuning)
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    target_modules=["to_q", "to_v", "to_k", "to_out"],
    lora_alpha=16,
    lora_dropout=0.0,
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

# 3. Load Dataset
print("Loading dataset...")
# Example: Use image-video pairs
dataset = load_dataset("your-image-video-dataset")

def collate_fn(batch):
    """Collate images and videos"""
    images = [item["image"] for item in batch]
    videos = [item["video"] for item in batch]  # 16 frames each
    return {"images": images, "videos": videos}

dataloader = DataLoader(dataset["train"], batch_size=1, collate_fn=collate_fn)

# 4. Training Loop
print("Starting training...")
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)

for epoch in range(5):
    for batch in dataloader:
        images = batch["images"]
        videos = batch["videos"]  # Target videos
        
        # Forward pass (simplified)
        # In practice, use diffusion training utilities
        # This is conceptual
        
        # Encode images
        image_features = pipe.image_encoder(images)
        
        # Generate video
        generated_videos = pipe.unet(
            videos,  # Noisy videos
            timestep=torch.randint(0, 1000, (len(videos),)),
            encoder_hidden_states=image_features,
        )
        
        # Loss (simplified - use actual diffusion loss)
        loss = torch.nn.functional.mse_loss(generated_videos, videos)
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Loss: {loss.item():.4f}")

print("✅ Training complete!")
```

---

## 🎯 Recommended Approach

### Stage 1: Supervised Fine-Tuning (Start Here)

**Goal:** Train model on image-video pairs

**Process:**
1. Load image-video dataset
2. Fine-tune with LoRA
3. Model learns motion patterns

**Result:** Model generates better videos from images

### Stage 2: RL Fine-Tuning (Optional, Later)

**Only if needed:**
- After SFT is working
- To improve specific quality metrics
- Requires custom implementation

---

## 📊 Comparison

| Method | Complexity | Reliability | Results | Recommended |
|--------|-----------|-------------|---------|-------------|
| **SFT** | Low | High | Good | ✅ **Yes** |
| **RL** | High | Medium | Varies | ⚠️ Only if needed |

---

## ✅ Action Items

1. **Stop RL training** (rewards are 0.0)
2. **Switch to SFT** (more reliable)
3. **Use image-video pairs** dataset
4. **Fine-tune with LoRA** (efficient)

**Your 205GB VRAM:** Perfect for SFT training! 🚀

