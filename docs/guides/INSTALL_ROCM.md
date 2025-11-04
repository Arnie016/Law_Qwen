# Install ROCm for MI300X

## GPU Detected! ✅
Your GPU is detected: `Instinct MI300X VF`

Now install ROCm drivers to use it.

---

## Step 1: Install ROCm

```bash
# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /usr/share/keyrings/rocm.gpg > /dev/null

# Add repository (Ubuntu 25.10 uses jammy repo)
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0/ jammy main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install
sudo apt update
sudo apt install rocm-dkms rocm-smi rocminfo -y
```

---

## Step 2: Reboot (if needed)

```bash
# ROCm drivers may need reboot
sudo reboot
```

After reboot, SSH back in and continue.

---

## Step 3: Verify ROCm

```bash
# Check ROCm
rocm-smi
rocminfo | grep -i gpu

# Should show GPU info
```

---

## Step 4: Install PyTorch with ROCm

```bash
source /root/venv/bin/activate

# Uninstall CUDA PyTorch
pip uninstall torch torchvision torchaudio -y

# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify GPU detection
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  Still not detected - may need reboot")
EOF
```

---

## Quick Install (Copy-Paste)

```bash
# Install ROCm
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /usr/share/keyrings/rocm.gpg > /dev/null

echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0/ jammy main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm-dkms rocm-smi rocminfo -y

# Check GPU
rocm-smi

# Install ROCm PyTorch
source /root/venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify
python << 'EOF'
import torch
print(f"GPU: {torch.cuda.is_available()}")
EOF
```

---

## If ROCm Installation Fails

Some DigitalOcean droplets may have ROCm pre-installed but need activation. Check:

```bash
# Check if ROCm modules are loaded
lsmod | grep amdgpu

# Check kernel modules
modprobe amdgpu || echo "Module not available"
```

If ROCm won't install, you may need to:
1. Use DigitalOcean's ROCm-enabled image
2. Contact DigitalOcean support
3. Use CPU training (slower but works)

