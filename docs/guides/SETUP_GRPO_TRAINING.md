# Setup GRPO Training with Unsloth

## You're in Docker Container - GPU Working! ✅

Now setup GRPO training instead of long SFT training.

---

## Step 1: Install Unsloth (Inside Container)

```bash
# You're already in container (root@111877ad5094:/workspace#)
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate
```

---

## Step 2: Install Other Packages

```bash
# Install additional packages
pip install transformers datasets
```

---

## Step 3: Download GRPO Script

```bash
# Create directories
mkdir -p /root/scripts/grpo
cd /root/scripts/grpo

# Download GRPO legal training script
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py

# Verify
ls -lh grpo_legal_training.py
```

---

## Step 4: Verify Setup

```bash
# Check Unsloth and GRPO work
python3 << 'EOF'
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import torch
print("✅ Unsloth installed")
print("✅ GRPO available")
print(f"✅ GPU: {torch.cuda.is_available()}")
EOF
```

---

## Step 5: Run GRPO Training

```bash
# Run GRPO training
cd /root/scripts/grpo
python3 grpo_legal_training.py
```

---

## Quick Setup (Copy-Paste)

```bash
# Install Unsloth and dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && \
pip install --no-deps trl peft accelerate && \
pip install transformers datasets && \
mkdir -p /root/scripts/grpo && \
cd /root/scripts/grpo && \
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py && \
python3 << 'EOF'
from unsloth import FastLanguageModel
from trl import GRPOConfig
import torch
print("✅ Ready for GRPO training!")
print(f"GPU: {torch.cuda.is_available()}")
EOF
```

---

## Run GRPO Training

```bash
cd /root/scripts/grpo
python3 grpo_legal_training.py
```

---

## What GRPO Training Does

- **Starts from checkpoint-500** (if available) or base model
- **Trains 500-1000 steps** (much faster than 10k SFT!)
- **Uses legal reasoning rewards** to improve responses
- **Expected time:** ~4-8 hours
- **Expected improvement:** +2-4 points

---

## Expected Output

```
GRPO Legal Reasoning Training
============================================================

GPU Info:
  Available: True
  Device: [GPU Name]
  Memory: 192.0 GB

1. Loading model from checkpoint: ./qwen2.5-32b-law-finetuned/checkpoint-500
...
```

---

## If Checkpoint-500 Doesn't Exist

The script will start from base model instead. That's fine - GRPO will still work!

---

## Training Time Comparison

| Method | Steps | Time | Cost |
|--------|-------|------|------|
| **GRPO** | 500-1000 | **~4-8 hours** | **~$10-20** |
| SFT 10k | 10,000 | ~80 hours | ~$160 |

**GRPO is much faster and cheaper!** 🚀

---

**Install Unsloth and run GRPO training - much faster than 10k steps!**

