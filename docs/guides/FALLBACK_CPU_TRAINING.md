# Fix: ROCm Not Compatible + PyTorch Uninstalled

## Problem
- ROCm installation failed (Ubuntu 25.10 not compatible)
- ROCm PyTorch wheels don't exist for Python 3.13
- PyTorch was uninstalled

---

## Solution: Reinstall Standard PyTorch + Use CPU Training

```bash
source /root/venv/bin/activate

# Reinstall standard PyTorch
pip install torch transformers datasets peft accelerate trl

# Verify installation
python << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("⚠️  Will use CPU training (slower but works)")
EOF
```

---

## Run Training on CPU

The training script will automatically use CPU if GPU isn't available:

```bash
cd /root/scripts/training

# Make sure script exists
ls -lh finetune_qwen_law_single_gpu.py || \
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

source /root/venv/bin/activate

# Run training (will use CPU)
python finetune_qwen_law_single_gpu.py
```

---

## Why CPU Training?

1. ✅ **Works immediately** - No ROCm setup needed
2. ✅ **Compatible** - Python 3.13 + standard PyTorch
3. ⚠️ **Slower** - But still works for fine-tuning
4. ⚠️ **Smaller models** - Might need to reduce batch size

---

## Modify Script for CPU (Optional)

If you want to explicitly use CPU:

```python
# Change device_map="cuda:0" to device_map="cpu"
# Or remove device_map entirely (auto-detects)
```

---

## Quick Fix (Copy-Paste)

```bash
source /root/venv/bin/activate

# Reinstall PyTorch
pip install torch transformers datasets peft accelerate trl

# Verify
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
EOF

# Run training (CPU mode)
cd /root/scripts/training
python finetune_qwen_law_single_gpu.py
```

---

## Alternative: Contact DigitalOcean

DigitalOcean MI300X droplets might have ROCm pre-installed but need activation. Check:

```bash
# Check if ROCm modules exist
ls /lib/modules/*/kernel/drivers/gpu/drm/amd/ | grep amdgpu || echo "No ROCm modules"

# Check if kernel modules can be loaded
modprobe amdgpu 2>&1 || echo "Module not available"
```

If ROCm modules exist but aren't loaded, DigitalOcean support can help activate them.

---

## For Now: Use CPU Training

CPU training will work, just slower. Fine-tuning a 32B model on CPU will take longer but is feasible for LoRA (only trains ~400MB adapters).

**Reinstall PyTorch and run training - it will use CPU automatically.**

