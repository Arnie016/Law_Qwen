# Setup New ROCm Droplet - Quick Start

## New Droplet Info
- **IP:** 129.212.184.211
- **Image:** PyTorch (ROCm7) 2.6.0 on Ubuntu 24.04
- **GPU:** MI300X 192GB

---

## Step 1: Connect to Droplet

```bash
ssh root@129.212.184.211
```

---

## Step 2: Verify GPU Works

```bash
# Check ROCm
rocm-smi

# Check PyTorch GPU
python3 << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("❌ GPU not detected")
EOF
```

---

## Step 3: Setup Training Environment

```bash
# Create virtual environment
python3 -m venv /root/venv
source /root/venv/bin/activate

# Install additional packages
pip install transformers datasets peft accelerate trl

# Verify packages
python << 'EOF'
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
print("✅ All packages installed!")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()}")
EOF
```

---

## Step 4: Download Training Script

```bash
# Create directories
mkdir -p /root/scripts/training
cd /root/scripts/training

# Download script
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Make executable
chmod +x finetune_qwen_law_single_gpu.py
```

---

## Step 5: Run Training

```bash
# Make sure venv is activated
source /root/venv/bin/activate

# Run training (should use GPU now!)
python finetune_qwen_law_single_gpu.py
```

---

## Quick Setup Script (All-in-One)

```bash
# Copy-paste this entire block into SSH session
cat > /tmp/setup_training.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "=========================================="
echo "Setting up GPU Training Environment"
echo "=========================================="

# Step 1: Verify GPU
echo ""
echo "1. Checking GPU..."
rocm-smi || echo "⚠️  rocm-smi not available"
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
EOF

# Step 2: Create venv
echo ""
echo "2. Creating virtual environment..."
python3 -m venv /root/venv
source /root/venv/bin/activate

# Step 3: Install packages
echo ""
echo "3. Installing packages..."
pip install --upgrade pip
pip install transformers datasets peft accelerate trl

# Step 4: Download script
echo ""
echo "4. Downloading training script..."
mkdir -p /root/scripts/training
cd /root/scripts/training
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Step 5: Verify everything
echo ""
echo "5. Verifying setup..."
source /root/venv/bin/activate
python << 'EOF'
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
print("✅ All packages installed!")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.is_available()}")
EOF

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  cd /root/scripts/training"
echo "  source /root/venv/bin/activate"
echo "  python finetune_qwen_law_single_gpu.py"
echo ""
SCRIPT

chmod +x /tmp/setup_training.sh
/tmp/setup_training.sh
```

---

## One-Liner Setup

```bash
# Run this entire command in one go
ssh root@129.212.184.211 "bash -s" << 'EOF'
python3 -m venv /root/venv && \
source /root/venv/bin/activate && \
pip install transformers datasets peft accelerate trl && \
mkdir -p /root/scripts/training && \
cd /root/scripts/training && \
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py && \
python3 << 'PYTHON'
import torch
print(f"GPU: {torch.cuda.is_available()}")
PYTHON
EOF
```

---

## Verify GPU First

Before setting up, verify GPU works:

```bash
ssh root@129.212.184.211

# Quick GPU check
rocm-smi
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

If GPU shows `True`, proceed with setup!

---

## Expected Results

After setup, you should see:
- ✅ `rocm-smi` shows GPU info
- ✅ `torch.cuda.is_available()` returns `True`
- ✅ GPU name: MI300X or similar
- ✅ Memory: ~192 GB

---

## Run Training

```bash
cd /root/scripts/training
source /root/venv/bin/activate
python finetune_qwen_law_single_gpu.py
```

Training should now use GPU! 🚀

