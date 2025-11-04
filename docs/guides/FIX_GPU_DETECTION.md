# Fix: GPU Not Detected by Unsloth

## Problem
Unsloth can't detect GPU because PyTorch doesn't have ROCm support or GPU isn't visible.

---

## Step 1: Check GPU Detection

```bash
source /root/venv/bin/activate

python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("❌ GPU not detected!")
    print("Need to install PyTorch with ROCm support")
EOF
```

---

## Step 2: Install PyTorch with ROCm

```bash
source /root/venv/bin/activate

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with ROCm (for AMD GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

---

## Step 3: Check ROCm Installation

```bash
# Check if ROCm is installed
rocminfo | grep -i gpu || echo "ROCm not found"

# Check ROCm version
rocm-smi || echo "rocm-smi not available"
```

---

## Step 4: Alternative - Use Standard Training Script

If Unsloth still doesn't work, use the standard transformers/peft approach:

```bash
# Download standard training script (no Unsloth)
cd /root/scripts/training
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Run it
source /root/venv/bin/activate
python finetune_qwen_law_single_gpu.py
```

---

## Quick Fix (Copy-Paste)

```bash
source /root/venv/bin/activate

# Check GPU first
python << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("Installing PyTorch with ROCm...")
EOF

# Install PyTorch with ROCm
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify
python << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
EOF

# Try GRPO again
python grpo_legal_training.py
```

---

## If Still Not Working

Use the standard training script (no Unsloth required):

```bash
cd /root/scripts/training
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

source /root/venv/bin/activate
python finetune_qwen_law_single_gpu.py
```

This uses standard transformers/peft (no Unsloth), so it doesn't require GPU detection at import time.

