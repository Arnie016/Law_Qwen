# Quick Fix: Install PyTorch

## Install PyTorch

```bash
source /root/venv/bin/activate

# Install PyTorch and dependencies
pip install torch transformers datasets peft accelerate trl

# Verify installation
python << 'EOF'
import torch
print(f"✅ PyTorch installed: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
EOF
```

---

## Full Setup (Copy-Paste)

```bash
# Activate venv
source /root/venv/bin/activate

# Install all packages
pip install torch transformers datasets peft accelerate trl

# Verify
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

## Then Run Training

```bash
# Make sure you're in the right directory
cd /root/scripts/training

# Make sure venv is activated
source /root/venv/bin/activate

# Run training script
python finetune_qwen_law_single_gpu.py
```

---

## One-Liner Setup

```bash
source /root/venv/bin/activate && \
pip install torch transformers datasets peft accelerate trl && \
python << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.is_available()}")
EOF
```

