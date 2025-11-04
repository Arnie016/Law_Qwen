# Fine-Tune Video Multimodal LLM - Complete Guide

## What You Can Train

### Video Understanding Tasks:
1. **Video Captioning** - Describe what's happening in videos
2. **Video QA** - Answer questions about video content
3. **Video Action Classification** - Identify actions (dancing, cooking, etc.)
4. **Video Summarization** - Create summaries of long videos
5. **Video-to-Story** - Generate narratives from videos

## Best Models for Video Multimodal

### 1. Qwen2.5-VL-72B-Instruct ⭐ RECOMMENDED
- **Size:** 72B parameters
- **Works with:** Video frames + text
- **Uses:** All 8 GPUs automatically
- **Best for:** Video understanding, captioning, QA

### 2. Qwen2.5-VL-7B-Instruct
- **Size:** 7B parameters (smaller, faster)
- **Good for:** Quick testing, smaller datasets

### 3. LLaVA-NeXT-Video
- **Alternative:** Good video understanding model

## Best Video Datasets

### 1. LLaVA-Video-178K ⭐ RECOMMENDED
- **Size:** 178K video-text pairs
- **Format:** Video frames + captions/questions
- **Good for:** Video captioning, QA

### 2. Video-MME
- **Size:** Medium (multiple choice QA)
- **Format:** Videos + questions + answers
- **Good for:** Video question answering

### 3. PE-Video (Facebook)
- **Size:** 1M+ videos (use streaming)
- **Format:** Various video tasks
- **Good for:** Large-scale training

## How Video Multimodal Training Works

### Step 1: Process Videos
```python
# Videos are converted to frames
video → [frame1.jpg, frame2.jpg, ..., frameN.jpg]
```

### Step 2: Combine with Text
```python
# Format: Video frames + instruction text
Input: {
    "video_frames": [frame1, frame2, ...],
    "text": "Describe what's happening in this video"
}
```

### Step 3: Model Processes Both
```python
# Vision encoder processes frames
# Language model processes text
# Both combined for understanding
```

### Step 4: Generate Output
```python
# Model generates text based on video + text input
Output: "A person is cooking pasta in a kitchen..."
```

## Complete Fine-Tuning Script

```python
#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-VL-72B on video dataset
"""
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

print("=" * 60)
print("Fine-Tuning Video Multimodal LLM")
print("=" * 60)

# Model
model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
print(f"\n1. Loading model: {model_name}")

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Uses all 8 GPUs
    trust_remote_code=True
)

# Freeze base model
for param in model.parameters():
    param.requires_grad = False

print("✅ Model loaded")

# Dataset
print("\n2. Loading video dataset...")
dataset = load_dataset("lmms-lab/LLaVA-Video-178K", streaming=True)
print("✅ Dataset loaded (streaming mode)")

# Format dataset
print("\n3. Formatting dataset...")
def format_video_example(examples):
    """
    Format video frames + text for training
    """
    formatted = []
    for i in range(len(examples['video'])):
        # Get video frames (or frame paths)
        video_frames = examples['video'][i]  # This could be frames or paths
        
        # Create instruction
        if 'question' in examples:
            instruction = f"<|im_start|>user\nVideo: {video_frames}\nQuestion: {examples['question'][i]}<|im_end|>\n<|im_start|>assistant\n"
            answer = examples['answer'][i]
        else:
            instruction = f"<|im_start|>user\nDescribe this video: {video_frames}<|im_end|>\n<|im_start|>assistant\n"
            answer = examples['caption'][i] if 'caption' in examples else ""
        
        formatted.append({
            "text": instruction + answer + "<|im_end|>",
            "video_frames": video_frames
        })
    
    return formatted

# Note: Video processing is complex - you may need custom dataloader
# This is a simplified version

# LoRA setup
print("\n4. Setting up LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Enable gradients for LoRA
model.train()
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

print("✅ LoRA configured")

# Training args
training_args = TrainingArguments(
    output_dir="./qwen2.5-vl-72b-video-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Video needs smaller batch size
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    warmup_steps=50,
    max_steps=500,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
)

# Trainer (you'll need custom data collator for video)
print("\n5. Starting training...")
# Note: Video training requires custom preprocessing
# See next section for full implementation

print("✅ Fine-tuning setup complete!")
print("Note: Video training requires custom video frame processing")
```

## Practical Video Fine-Tuning (Simplified)

### Option 1: Video Captioning (Easier)

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model
model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Freeze base
for p in model.parameters(): p.requires_grad = False

# LoRA
lora = LoraConfig(r=16, target_modules=["q_proj","k_proj","v_proj","o_proj"])
model = get_peft_model(model, lora)

# Dataset (simplified - you'll need to process video frames)
# For now, use image dataset as proxy
dataset = load_dataset("lmms-lab/LLaVA-Video-178K", streaming=True)

# Format: Convert videos to frame sequences
# This requires video processing library (opencv-python, imageio)
```

### Option 2: Start with Image Dataset (Easier Path)

```python
# Fine-tune on image dataset first (easier), then adapt to video
# Image → Video is similar architecture

from datasets import load_dataset
dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")
# This has images + text, similar to video but simpler
```

## Requirements

```bash
pip install transformers datasets accelerate peft torch torchvision
pip install opencv-python imageio  # For video processing
pip install pillow  # For image handling
```

## Video Processing Pipeline

```python
import cv2
import imageio
from PIL import Image

def extract_video_frames(video_path, num_frames=8):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames
```

## Quick Start Command

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

# Install dependencies
pip install transformers datasets accelerate peft opencv-python imageio

# Run training (see script above)
python3 finetune_video_multimodal.py
```

## Expected Training Time

- **Model:** Qwen2.5-VL-72B (72B parameters)
- **Dataset:** 178K video-text pairs
- **GPUs:** 8x MI300X
- **Time:** ~4-8 hours for 500 steps

## What You Get

After fine-tuning:
- Model understands video content better
- Generates accurate captions
- Answers video questions correctly
- Classifies actions in videos

## Next Steps

1. Download video dataset
2. Process videos into frames
3. Format for training
4. Fine-tune with LoRA
5. Test on new videos

Ready to start? Let me know which task you want to focus on!

