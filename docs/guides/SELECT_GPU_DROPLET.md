# How to Select GPU Droplet with ROCm

## ✅ **Solution: Use PyTorch ROCm Image**

DigitalOcean provides **pre-configured ROCm images**! Use the PyTorch image.

---

## Step-by-Step Selection

### 1. Choose GPU Plan
- **MI300X** (1 GPU) - $1.99/hr ← **Select this**
- Or MI300X x8 (8 GPUs) if you need more

### 2. Choose Image → **ROCm Software** section

Scroll down to **"ROCm Software"** section and select:

**✅ PyTorch**  
- PyTorch 2.6.0, ROCm 7.0.0  
- Ready for PyTorch development and training  
- Image ID: `201616009`

This image has:
- ✅ ROCm 7.0.0 pre-installed
- ✅ PyTorch 2.6.0 with ROCm support
- ✅ GPU drivers configured
- ✅ Everything ready to use!

---

## Why This Image?

### Current Problem:
- Ubuntu 25.10 without ROCm
- PyTorch CUDA (NVIDIA) instead of ROCm (AMD)
- ROCm installation failed

### With PyTorch ROCm Image:
- ✅ ROCm 7.0.0 pre-installed
- ✅ PyTorch 2.6.0 with ROCm support
- ✅ GPU drivers configured
- ✅ Works immediately!

---

## After Creating Droplet

### 1. SSH into Droplet
```bash
ssh root@YOUR_NEW_IP
```

### 2. Verify GPU Works
```bash
# Check ROCm
rocm-smi

# Check PyTorch GPU
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

### 3. Setup Training Environment
```bash
# Create venv
python3 -m venv /root/venv
source /root/venv/bin/activate

# Install additional packages
pip install transformers datasets peft accelerate trl

# Download training script
mkdir -p /root/scripts/training
cd /root/scripts/training
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Run training
python finetune_qwen_law_single_gpu.py
```

---

## Alternative Images

If PyTorch image doesn't work, try:

### Option 1: ROCm Software (Base)
- Base ROCm installation
- Image ID: `200217503`
- Install PyTorch yourself

### Option 2: Megatron
- For large-scale training
- ROCm 7.0.0
- Image ID: `201813315`

---

## Quick Setup Script (After Droplet Created)

```bash
# Run this after SSH into new droplet
cat > /tmp/setup_training.sh << 'SCRIPT'
#!/bin/bash
echo "Setting up training environment..."

# Activate venv
python3 -m venv /root/venv
source /root/venv/bin/activate

# Install packages
pip install transformers datasets peft accelerate trl

# Download script
mkdir -p /root/scripts/training
cd /root/scripts/training
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Verify GPU
python3 << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
EOF

echo "✅ Setup complete!"
SCRIPT

chmod +x /tmp/setup_training.sh
/tmp/setup_training.sh
```

---

## Recommendation

**Select:**
1. **GPU Plan:** MI300X (1 GPU) - $1.99/hr
2. **Image:** PyTorch (PyTorch 2.6.0, ROCm 7.0.0) - Image ID: `201616009`

This will give you:
- ✅ ROCm 7.0.0 pre-installed
- ✅ PyTorch 2.6.0 with ROCm support
- ✅ GPU working immediately
- ✅ No manual ROCm installation needed

---

## What About Current Droplet?

Your current droplet (Ubuntu 25.10 without ROCm) has:
- ❌ ROCm not installed
- ❌ GPU not detected
- ❌ Can't use GPU

**Solution:** Create new droplet with PyTorch ROCm image, or contact DigitalOcean to switch image.

---

## After Creating New Droplet

Once you have the new IP, run:

```bash
# SSH into new droplet
ssh root@NEW_IP

# Verify GPU works
rocm-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Setup training (see script above)
```

**Select the PyTorch ROCm image - it has everything pre-configured!** 🚀

