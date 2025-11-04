# GPU Diagnostics Commands for DigitalOcean Droplet

## Connect to Droplet

```bash
ssh root@134.199.195.11
```

---

## Comprehensive GPU Diagnostics

### 1. Check GPU Hardware

```bash
# Check PCI devices
lspci | grep -i amd

# Detailed GPU info
lspci -v | grep -A 10 -i amd

# Check GPU in /dev
ls -la /dev/kfd* /dev/dri/* 2>/dev/null || echo "No GPU devices found"
```

### 2. Check ROCm Installation

```bash
# Check if ROCm is installed
which rocm-smi || echo "ROCm not installed"
which rocminfo || echo "ROCminfo not installed"

# Check ROCm packages
dpkg -l | grep -i rocm

# Check ROCm modules
lsmod | grep amdgpu || echo "AMD GPU driver not loaded"
lsmod | grep kfd || echo "KFD (ROCm) not loaded"

# Check kernel modules
modinfo amdgpu 2>/dev/null || echo "amdgpu module not available"
```

### 3. Check System Configuration

```bash
# Check Ubuntu version
lsb_release -a

# Check kernel version
uname -r

# Check if GPU is in device tree
dmesg | grep -i amd
dmesg | grep -i gpu

# Check for GPU-related errors
dmesg | grep -i error | grep -i gpu
```

### 4. Check DigitalOcean Specific

```bash
# Check if DigitalOcean has GPU setup scripts
ls -la /usr/local/bin/*gpu* 2>/dev/null || echo "No GPU scripts"
ls -la /opt/digitalocean/* 2>/dev/null || echo "No DigitalOcean scripts"

# Check systemd services
systemctl list-units | grep -i gpu
systemctl list-units | grep -i rocm

# Check environment variables
env | grep -i gpu
env | grep -i rocm
```

### 5. Check PyTorch GPU Detection

```bash
source /root/venv/bin/activate

python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# Check for ROCm/HIP
if hasattr(torch.version, 'hip'):
    print(f"HIP version: {torch.version.hip}")

# Try to create tensor
try:
    x = torch.randn(10, 10)
    print(f"✅ CPU tensor works")
    if torch.cuda.is_available():
        x = x.cuda()
        print(f"✅ GPU tensor works: {x.device}")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

### 6. Check GPU Memory

```bash
# Check GPU memory via system
cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null || echo "Cannot read GPU memory"

# Check GPU via sysfs
ls -la /sys/class/drm/card*/device/ 2>/dev/null || echo "No GPU devices in sysfs"
```

### 7. Full Diagnostic Script

```bash
cat > /tmp/gpu_diagnostic.sh << 'SCRIPT'
#!/bin/bash
echo "=== GPU DIAGNOSTIC REPORT ==="
echo ""
echo "1. GPU Hardware:"
lspci | grep -i amd
echo ""
echo "2. ROCm Installation:"
which rocm-smi || echo "❌ rocm-smi not found"
which rocminfo || echo "❌ rocminfo not found"
echo ""
echo "3. Kernel Modules:"
lsmod | grep amdgpu || echo "❌ amdgpu module not loaded"
lsmod | grep kfd || echo "❌ kfd module not loaded"
echo ""
echo "4. Device Files:"
ls -la /dev/kfd* /dev/dri/* 2>/dev/null || echo "❌ No GPU device files"
echo ""
echo "5. PyTorch GPU Detection:"
source /root/venv/bin/activate 2>/dev/null
python3 << 'PYTHON'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  GPU Available: {torch.cuda.is_available()}")
PYTHON
echo ""
echo "6. System Info:"
echo "  Ubuntu: $(lsb_release -rs)"
echo "  Kernel: $(uname -r)"
echo ""
echo "=== END REPORT ==="
SCRIPT

chmod +x /tmp/gpu_diagnostic.sh
/tmp/gpu_diagnostic.sh
```

---

## Run Full Diagnostic

```bash
# Connect and run diagnostic
ssh root@134.199.195.11 "bash -s" << 'EOF'
echo "=== GPU DIAGNOSTIC ==="
echo "GPU Hardware:"
lspci | grep -i amd
echo ""
echo "ROCm:"
which rocm-smi || echo "Not installed"
echo ""
echo "Kernel Modules:"
lsmod | grep amdgpu || echo "Not loaded"
echo ""
echo "PyTorch:"
source /root/venv/bin/activate 2>/dev/null
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not in venv"
EOF
```

---

## What to Look For

### ✅ Good Signs:
- GPU appears in `lspci`
- ROCm modules loaded (`amdgpu`, `kfd`)
- Device files exist (`/dev/kfd0`, `/dev/dri/card0`)
- `rocm-smi` works

### ❌ Bad Signs:
- No ROCm modules loaded
- No device files
- ROCm not installed
- PyTorch shows `False` for GPU

---

## Next Steps Based on Results

### If ROCm Not Installed:
1. Contact DigitalOcean support
2. Ask for ROCm-enabled image
3. Check if they have setup scripts

### If ROCm Installed But Not Working:
1. Check kernel modules: `modprobe amdgpu`
2. Check permissions: `ls -la /dev/kfd*`
3. Check logs: `dmesg | grep -i amd`

### If PyTorch Wrong Version:
1. Install ROCm PyTorch
2. Or use CPU training as fallback

