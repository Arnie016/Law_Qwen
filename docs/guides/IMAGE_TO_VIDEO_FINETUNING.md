# Image-to-Video Fine-Tuning: How It Works

## 🎬 Overview

**Image-to-Video Models:** Take an image as input, generate a video showing motion from that image.

**Example:**
- Input: Photo of a cat
- Output: Video of cat walking (16 frames)

---

## 🔧 Fine-Tuning Process

### Standard Fine-Tuning (SFT - Supervised Fine-Tuning)

**1. Dataset Format:**
```
{
  "image": "path/to/image.jpg",
  "video": "path/to/video.mp4",  # Target video (16 frames)
  "prompt": "Optional text description"
}
```

**2. Training Process:**

```python
# Pseudocode
for each (image, video) pair:
    # Encode image
    image_embeddings = vision_encoder(image)
    
    # Generate video from image
    generated_video = model(image_embeddings)
    
    # Compare with target video
    loss = mse_loss(generated_video, target_video)
    
    # Update model
    loss.backward()
    optimizer.step()
```

**3. What the Model Learns:**
- How to generate motion from static images
- What kind of motion matches each image type
- Temporal consistency (smooth transitions between frames)

---

## 🎯 Two-Stage Fine-Tuning Approach

### Stage 1: Image-to-Video Base Training

**Goal:** Learn general image-to-video generation

**Process:**
1. Load pre-trained model (e.g., Stable Video Diffusion)
2. Train on image-video pairs
3. Model learns: "Given this image, generate video showing motion"

**Example:**
- Input: Photo of ocean waves
- Target: Video of waves moving
- Model learns: Static water → Moving waves

### Stage 2: Domain-Specific Fine-Tuning

**Goal:** Specialize for specific types of videos

**Process:**
1. Start from Stage 1 model
2. Fine-tune on domain-specific data (e.g., sports videos, nature, etc.)
3. Model learns: "Generate videos in this specific style"

**Example:**
- Domain: Sports videos
- Input: Photo of basketball player
- Target: Video of player shooting
- Model learns: Sports-specific motion patterns

---

## 🧠 How Diffusion Models Work

### Image-to-Video Diffusion Process:

**1. Forward Process (Training):**
```
Original Video (16 frames)
    ↓
Add noise progressively
    ↓
Noisy Video (pure noise)
```

**2. Reverse Process (Generation):**
```
Noisy Video
    ↓
Denoise step by step
    ↓
Clean Video (16 frames)
```

**3. Training:**
- Model learns to predict noise at each step
- Given noisy video + image, predict noise
- Remove predicted noise = cleaner video

---

## 📊 Fine-Tuning Techniques

### 1. **LoRA (Low-Rank Adaptation)** ⭐ Recommended

**Why:** Efficient, fast, memory-friendly

**How:**
```python
# Add LoRA adapters to model
model = get_peft_model(model, LoRAConfig(
    r=16,  # Rank
    target_modules=["to_q", "to_v", "to_k", "to_out"],  # Which layers
))

# Train only LoRA weights (not full model)
# Freezes base model, only updates LoRA adapters
```

**Benefits:**
- Train only 1-5% of parameters
- Much faster training
- Lower memory usage
- Can switch between fine-tuned versions easily

### 2. **Full Fine-Tuning**

**How:**
```python
# Train entire model
for param in model.parameters():
    param.requires_grad = True

# Update all weights
loss.backward()
optimizer.step()
```

**Benefits:**
- Maximum flexibility
- Can learn complex patterns

**Drawbacks:**
- Requires more memory
- Slower training
- Risk of overfitting

### 3. **Adapter Layers**

**How:**
```python
# Add adapter modules
# Keep base model frozen
# Only train adapter layers
```

**Benefits:**
- Memory efficient
- Easy to add/remove

---

## 🔄 Training Loop Explained

### Step-by-Step:

**1. Data Loading:**
```python
for batch in dataloader:
    images = batch["image"]  # Input images
    videos = batch["video"]   # Target videos (16 frames)
```

**2. Forward Pass:**
```python
# Encode image
image_features = vision_encoder(images)

# Generate video
noise = torch.randn_like(videos)  # Random noise
timestep = random.randint(0, 1000)  # Random timestep
noisy_video = add_noise(videos, noise, timestep)

# Predict noise
predicted_noise = model(noisy_video, image_features, timestep)
```

**3. Loss Calculation:**
```python
# Compare predicted noise with actual noise
loss = mse_loss(predicted_noise, noise)
```

**4. Backward Pass:**
```python
loss.backward()  # Compute gradients
optimizer.step()  # Update weights
```

---

## 🎨 Specialized Fine-Tuning Scenarios

### 1. **Style Transfer**

**Goal:** Generate videos in specific style

**Example:**
- Fine-tune on animated videos
- Model learns: "Generate animated-style motion"
- Input: Real photo → Output: Animated video

### 2. **Domain Adaptation**

**Goal:** Adapt to specific domain

**Example:**
- Fine-tune on medical videos
- Model learns: "Generate medical procedure videos"
- Input: Medical image → Output: Procedure video

### 3. **Quality Improvement**

**Goal:** Improve video quality

**Example:**
- Fine-tune on high-quality videos
- Model learns: "Generate sharper, smoother videos"
- Input: Any image → Output: High-quality video

---

## ⚙️ Key Hyperparameters

### Important Settings:

**1. Learning Rate:**
- LoRA: 1e-4 to 5e-4
- Full fine-tuning: 1e-5 to 1e-4
- Too high = unstable training
- Too low = slow learning

**2. Batch Size:**
- Video generation is memory-intensive
- Start small: 1-4 per GPU
- Use gradient accumulation if needed

**3. Training Steps:**
- Small dataset: 500-1000 steps
- Large dataset: 1000-5000 steps
- Monitor validation loss

**4. Frame Count:**
- Default: 14-16 frames
- More frames = longer videos
- More frames = more memory

---

## 🎯 Practical Example: Fine-Tuning Stable Video Diffusion

```python
from diffusers import StableVideoDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch

# 1. Load pre-trained model
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt"
)

# 2. Add LoRA
lora_config = LoraConfig(
    r=16,
    target_modules=["to_q", "to_v", "to_k", "to_out"],
    lora_alpha=16,
)
model = get_peft_model(pipe.unet, lora_config)

# 3. Prepare dataset
dataset = load_dataset("your-image-video-pairs")

# 4. Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch["image"]
        videos = batch["video"]  # Target videos
        
        # Forward pass
        noise = torch.randn_like(videos)
        timestep = torch.randint(0, 1000, (len(videos),))
        noisy_videos = add_noise(videos, noise, timestep)
        
        # Predict noise
        predicted_noise = model(noisy_videos, images, timestep)
        
        # Loss
        loss = mse_loss(predicted_noise, noise)
        
        # Backward
        loss.backward()
        optimizer.step()
```

---

## ✅ Summary

**Image-to-Video Fine-Tuning:**

1. **Dataset:** Image-video pairs
2. **Model:** Pre-trained diffusion model (Stable Video Diffusion)
3. **Method:** LoRA (efficient) or full fine-tuning
4. **Training:** Learn to predict noise, then denoise to generate video
5. **Goal:** Model learns to generate motion from static images

**Key Points:**
- Model learns motion patterns from training data
- LoRA is most efficient for fine-tuning
- Fine-tune on domain-specific data for better results
- Video generation is memory-intensive (your 205GB VRAM helps!)

**Your Setup:**
- ✅ 205GB VRAM - Perfect for video fine-tuning
- ✅ ROCm GPU - AMD optimized
- ✅ Ready to fine-tune!

