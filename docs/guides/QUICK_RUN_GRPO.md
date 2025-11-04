# Quick Start: Run GRPO Training

## Step-by-Step Commands

### Step 1: Download Updated Script

```bash
# You're already in container (root@111877ad5094:/workspace#)
cd /root/scripts/grpo

# Download updated script
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py

# Verify it downloaded
ls -lh grpo_legal_training.py
```

---

### Step 2: Verify Setup

```bash
# Check Unsloth and packages
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

### Step 3: Run GRPO Training

```bash
# Run training
cd /root/scripts/grpo
python3 grpo_legal_training.py
```

---

## One-Liner (Copy-Paste)

```bash
cd /root/scripts/grpo && \
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py && \
python3 grpo_legal_training.py
```

---

## What You'll See

```
GRPO Legal Reasoning Training
============================================================

GPU Info:
  Available: True
  Device: [GPU Name]
  Memory: 192.0 GB

1. Loading model from checkpoint: ./qwen2.5-32b-law-finetuned/checkpoint-500
   ⚠️  Checkpoint not found at ./qwen2.5-32b-law-finetuned/checkpoint-500
   Starting fresh GRPO training
   Adding new LoRA adapters...
✅ Model loaded

2. Loading dataset...
   Using law dataset to create Q&A pairs...
   ✅ Dataset formatted

3. Setting up GRPO config...
4. Starting GRPO training...
   Training for 1000 steps...
```

---

## If Training Runs in Background

```bash
# Run in background
nohup python3 grpo_legal_training.py > grpo_training.log 2>&1 &

# Monitor progress
tail -f grpo_training.log

# Check GPU usage
watch -n 5 'rocm-smi | head -10'
```

---

**Copy-paste the one-liner above - it will download and run the fixed script!** 🚀

