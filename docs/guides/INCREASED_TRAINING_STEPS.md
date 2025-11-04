# Quick Setup Guide - Increased Training Steps

## 🎯 Goal: Train for 10,000 steps (vs 500 previously)

The training script has been updated to use **10,000 steps** instead of 500.

---

## 📥 Step 1: Download Updated Script from GitHub

```bash
# On your droplet (SSH session), run:
cd /root
mkdir -p scripts/training
cd scripts/training

# Download the updated script
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Verify it has max_steps=10000
grep "max_steps" finetune_qwen_law_single_gpu.py
```

---

## 🔍 Step 2: Verify GPU and Dependencies

```bash
# Check GPU
python3 << 'EOF'
import torch
print(f"GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Memory: {props.total_memory/1e9:.1f} GB")
EOF

# Check dependencies
pip3 list | grep -E "(torch|transformers|datasets|peft)" || pip3 install torch transformers datasets peft accelerate
```

---

## 🚀 Step 3: Start Training (10,000 steps)

### Option A: Run in Background (Recommended)

```bash
cd /root/scripts/training

# Start training in background
nohup python3 finetune_qwen_law_single_gpu.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Option B: Run Directly (see output)

```bash
cd /root/scripts/training
python3 finetune_qwen_law_single_gpu.py
```

---

## ⏱️ Expected Training Time

- **Steps:** 10,000 (vs 500 previously)
- **Time:** ~80 hours (~3-4 days) on single GPU
- **Checkpoints:** Saved every 500 steps (20 checkpoints total)
- **Cost:** ~$160 at $1.99/hour

---

## 📊 Monitor Progress

```bash
# Check if training is running
ps aux | grep python

# View latest logs
tail -f /root/scripts/training/training.log

# Check saved checkpoints
ls -lh /root/scripts/training/qwen2.5-32b-law-finetuned/checkpoint-*/

# Check GPU memory usage
watch -n 5 'python3 -c "import torch; print(f\"GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB\")"'
```

---

## 🔄 If Training Interrupts

The script saves checkpoints every 500 steps. To resume:

```bash
# Training will auto-resume from latest checkpoint
# Just run the script again:
cd /root/scripts/training
python3 finetune_qwen_law_single_gpu.py
```

---

## 📈 Checkpoints Will Be Saved

```
checkpoint-500/
checkpoint-1000/
checkpoint-1500/
...
checkpoint-10000/
```

Each checkpoint contains LoRA adapters (~400MB each).

---

## ✅ Quick One-Liner Setup

```bash
cd /root && \
mkdir -p scripts/training && \
cd scripts/training && \
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py && \
python3 finetune_qwen_law_single_gpu.py
```

---

## 🎯 What Changed

- **max_steps:** 500 → **10,000** ✅
- **save_steps:** 50 → **500** (fewer checkpoints, more manageable)
- **Expected improvement:** +2-5 points on evaluation (vs -0.11 currently)

---

**Ready to train! Run the commands above on your droplet.** 🚀

