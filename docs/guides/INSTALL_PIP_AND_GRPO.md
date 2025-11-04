# Quick Setup Guide - Installing pip and GRPO

## Step 1: Install pip

```bash
# Install python3-pip
apt update
apt install python3-pip -y

# Verify installation
pip3 --version
```

## Step 2: Install Dependencies

```bash
# Install transformers and basic packages first
pip3 install torch transformers datasets peft accelerate

# Then install Unsloth
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install TRL for GRPO
pip3 install trl
```

## Step 3: Verify Installation

```bash
# Check if everything is installed
python3 << 'EOF'
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed")

try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth installed")
except ImportError:
    print("❌ Unsloth not installed")

try:
    from trl import GRPOConfig, GRPOTrainer
    print("✅ TRL (GRPO) installed")
except ImportError:
    print("❌ TRL not installed")
EOF
```

## Step 4: Download GRPO Script

```bash
mkdir -p /root/scripts/grpo
cd /root/scripts/grpo

curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py
```

## Step 5: Check GPU

```bash
python3 << 'EOF'
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
EOF
```

---

## Full Setup (Copy-Paste)

```bash
# Install pip
apt update && apt install python3-pip -y

# Install basic packages
pip3 install torch transformers datasets peft accelerate trl

# Install Unsloth
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Download GRPO script
mkdir -p /root/scripts/grpo
cd /root/scripts/grpo
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py

# Verify
python3 << 'EOF'
import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig
print("✅ All packages installed!")
print(f"GPU: {torch.cuda.is_available()}")
EOF
```

