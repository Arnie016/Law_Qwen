# Fix: GPU Not Detected - AMD MI300X Needs ROCm

## Problem
PyTorch installed with CUDA support (`cu128`), but AMD MI300X needs ROCm support.

---

## Step 1: Check ROCm Installation

```bash
# Check if ROCm is installed
rocminfo | head -20 || echo "ROCm not installed"
rocm-smi || echo "rocm-smi not available"

# Check GPU hardware
lspci | grep -i amd
```

---

## Step 2: Install ROCm (if not installed)

```bash
# Check Ubuntu version
lsb_release -a

# Install ROCm (Ubuntu 25.10)
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /usr/share/keyrings/rocm.gpg > /dev/null

echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0/ jammy main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm-dkms -y
```

---

## Step 3: Install PyTorch with ROCm

```bash
source /root/venv/bin/activate

# Uninstall CUDA PyTorch
pip uninstall torch torchvision torchaudio -y

# Try ROCm PyTorch from official source
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# If that fails, try alternative
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

---

## Step 4: Alternative - Check if GPU Works Anyway

Sometimes PyTorch can use GPU even if detection fails:

```bash
source /root/venv/bin/activate

# Test GPU directly
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")

# Try to use GPU device
try:
    if torch.cuda.is_available():
        print("✅ CUDA GPU detected")
    else:
        # Try ROCm device
        if hasattr(torch.backends, 'hip'):
            print("✅ ROCm/HIP available")
        else:
            print("⚠️  No GPU detected")
            
    # Try creating tensor on GPU
    x = torch.randn(100, 100)
    if torch.cuda.is_available():
        x = x.cuda()
        print(f"✅ GPU tensor created: {x.device}")
    else:
        print("⚠️  Using CPU")
except Exception as e:
    print(f"Error: {e}")
EOF
```

---

## Step 5: Use CPU Training (Fallback)

If GPU doesn't work, you can still train on CPU (much slower):

```bash
cd /root/scripts/training

# Download script
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Modify script to use CPU if GPU not available
# Or just run it - transformers will auto-detect CPU
source /root/venv/bin/activate
python finetune_qwen_law_single_gpu.py
```

---

## Quick Check Commands

```bash
# Check GPU hardware
lspci | grep -i amd | grep -i vga

# Check ROCm
which rocm-smi
rocminfo | grep -i gpu

# Check PyTorch
source /root/venv/bin/activate
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"HIP available: {hasattr(torch.backends, 'hip')}")
EOF
```

---

## Most Likely Issue

DigitalOcean droplet might not have ROCm drivers installed. Options:

1. **Install ROCm drivers** (if droplet supports it)
2. **Use CPU training** (slow but works)
3. **Contact DigitalOcean** about ROCm support for MI300X

Check with: `rocm-smi` or `rocminfo` first.

