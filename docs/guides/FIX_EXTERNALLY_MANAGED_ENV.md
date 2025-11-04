# Fix: Externally-Managed-Environment Error

## Solution: Create Virtual Environment

Ubuntu 25.10 blocks system-wide pip installs. Use a virtual environment instead.

---

## Quick Fix (Copy-Paste)

```bash
# Install python3-venv
apt install python3-venv -y

# Create virtual environment
python3 -m venv /root/venv

# Activate virtual environment
source /root/venv/bin/activate

# Now install packages
pip install torch transformers datasets peft accelerate trl
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Verify
python << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.is_available()}")
EOF
```

---

## Alternative: Use --break-system-packages

**⚠️ Warning:** This can break system packages, but works for GPU droplets.

```bash
pip3 install --break-system-packages torch transformers datasets peft accelerate trl
pip3 install --break-system-packages "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Recommended: Virtual Environment Setup

```bash
# Step 1: Install venv
apt install python3-venv -y

# Step 2: Create venv
python3 -m venv /root/venv

# Step 3: Activate (do this every time you SSH)
source /root/venv/bin/activate

# Step 4: Install packages
pip install torch transformers datasets peft accelerate trl
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Step 5: Run scripts (make sure venv is activated)
python grpo_legal_training.py
```

---

## Add to .bashrc (Auto-activate venv)

```bash
# Add this to ~/.bashrc so venv activates automatically
echo 'source /root/venv/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

---

## Check if GPU works

```bash
# After installing in venv
source /root/venv/bin/activate
python << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
EOF
```

