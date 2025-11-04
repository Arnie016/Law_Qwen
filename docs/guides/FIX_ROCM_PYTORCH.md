# Fix: ROCm PyTorch Not Available

## Problem
ROCm PyTorch index isn't available. Try standard PyTorch first.

---

## Solution 1: Install Standard PyTorch

```bash
source /root/venv/bin/activate

# Install standard PyTorch
pip install torch transformers datasets peft accelerate trl

# Check GPU detection
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not detected - may need ROCm drivers")
EOF
```

---

## Solution 2: Check ROCm Installation

```bash
# Check if ROCm is installed
which rocm-smi || echo "ROCm not found"
rocminfo | head -20 || echo "rocminfo not available"

# Check GPU hardware
lspci | grep -i amd || lspci | grep -i vga
```

---

## Solution 3: Use Standard Training (No Unsloth)

If GPU detection is problematic, use standard transformers/peft:

```bash
cd /root/scripts/training

# Download standard training script
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

source /root/venv/bin/activate

# Install standard packages
pip install torch transformers datasets peft accelerate

# Run training (will work even if GPU detection is tricky)
python finetune_qwen_law_single_gpu.py
```

---

## Quick Fix (Copy-Paste)

```bash
source /root/venv/bin/activate

# Install standard PyTorch
pip install torch transformers datasets peft accelerate trl

# Check GPU
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not detected - will use CPU or check ROCm")
EOF

# If GPU works, try GRPO
# If not, use standard training script
```

---

## Alternative: Use Standard Training Script

Standard transformers/peft doesn't require Unsloth:

```bash
cd /root/scripts/training

# Download standard script
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

source /root/venv/bin/activate
pip install torch transformers datasets peft accelerate

# Run training
python finetune_qwen_law_single_gpu.py
```

This uses standard transformers (no Unsloth), so it's more compatible.

