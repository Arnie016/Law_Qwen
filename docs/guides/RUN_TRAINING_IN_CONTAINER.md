# GPU Working! Setup Training

## ✅ Status: GPU Detected and Working!

- PyTorch: 2.9.0 with ROCm 7.0.0 ✅
- GPU Available: True ✅
- Ready for training!

---

## Step 1: Install Packages (Inside Container)

```bash
# You're already in container (root@111877ad5094:/workspace#)
# Install packages
pip install transformers datasets peft accelerate trl
```

---

## Step 2: Download Training Script

```bash
# Create directories
mkdir -p /root/scripts/training
cd /root/scripts/training

# Download script
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Verify
ls -lh finetune_qwen_law_single_gpu.py
```

---

## Step 3: Verify Packages Installed

```bash
# Check all packages work
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
print("✅ All packages installed!")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.is_available()}")
EOF
```

---

## Step 4: Run Training

```bash
# Run training (will use GPU automatically)
cd /root/scripts/training
python3 finetune_qwen_law_single_gpu.py
```

---

## Quick Setup (Copy-Paste)

```bash
# Run all at once inside container
pip install transformers datasets peft accelerate trl && \
mkdir -p /root/scripts/training && \
cd /root/scripts/training && \
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py && \
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
print("✅ Ready to train!")
print(f"GPU: {torch.cuda.is_available()}")
EOF
```

---

## Run Training

```bash
cd /root/scripts/training
python3 finetune_qwen_law_single_gpu.py
```

---

## Expected Output

You should see:
```
Fine-Tuning Qwen 2.5 32B on Law Dataset (Single GPU)
============================================================

GPU Info:
  Available: True
  Device: [GPU Name]
  Memory: 192.0 GB

1. Loading model: Qwen/Qwen2.5-32B-Instruct
...
```

Training will use GPU automatically! 🚀

---

## Note: Stay in Container

Keep the container session open while training:
- Training runs in container
- Use `Ctrl+D` to detach (training continues)
- Reattach: `docker exec -it rocm /bin/bash`

---

**Install packages and run training - GPU is ready!** ✅

