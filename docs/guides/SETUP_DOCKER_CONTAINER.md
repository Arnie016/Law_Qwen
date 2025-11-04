# Setup Training in Docker Container

## ✅ GPU Detected!
ROCm shows GPU is working. PyTorch is in Docker container.

---

## Step 1: Enter Docker Container

```bash
# Enter the ROCm container
docker exec -it rocm /bin/bash
```

You'll see prompt change to: `root@<container-id>:/#`

---

## Step 2: Verify GPU in Container

```bash
# Check GPU
rocm-smi

# Check PyTorch
python3 << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
EOF
```

---

## Step 3: Install Additional Packages

```bash
# Inside container, install packages
pip install transformers datasets peft accelerate trl

# Verify
python3 << 'EOF'
from transformers import AutoModelForCausalLM
from peft import LoraConfig
print("✅ All packages installed!")
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
# Run training (inside container)
cd /root/scripts/training
python3 finetune_qwen_law_single_gpu.py
```

---

## Quick Setup Script (Inside Container)

```bash
# Enter container first
docker exec -it rocm /bin/bash

# Then run this setup script
cat > /tmp/setup_training.sh << 'SCRIPT'
#!/bin/bash
echo "=========================================="
echo "Setting up Training in Docker Container"
echo "=========================================="

# Verify GPU
echo ""
echo "1. Checking GPU..."
rocm-smi | head -5
python3 << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
EOF

# Install packages
echo ""
echo "2. Installing packages..."
pip install transformers datasets peft accelerate trl

# Download script
echo ""
echo "3. Downloading training script..."
mkdir -p /root/scripts/training
cd /root/scripts/training
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

echo ""
echo "✅ Setup complete!"
echo "Run: python3 finetune_qwen_law_single_gpu.py"
SCRIPT

chmod +x /tmp/setup_training.sh
/tmp/setup_training.sh
```

---

## One-Liner Setup (From Host)

```bash
# Run setup inside container
docker exec -it rocm bash << 'EOF'
pip install transformers datasets peft accelerate trl
mkdir -p /root/scripts/training
cd /root/scripts/training
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py
python3 << 'PYTHON'
import torch
print(f"GPU: {torch.cuda.is_available()}")
PYTHON
EOF
```

---

## Run Training (From Host)

```bash
# Run training inside container
docker exec -it rocm bash -c "cd /root/scripts/training && python3 finetune_qwen_law_single_gpu.py"
```

---

## Or Enter Container and Run

```bash
# Enter container
docker exec -it rocm /bin/bash

# Inside container:
cd /root/scripts/training
python3 finetune_qwen_law_single_gpu.py
```

---

## Important Notes

1. **Always work inside Docker container** - PyTorch only there
2. **GPU is accessible** - ROCm works in container
3. **Install packages in container** - Not on host
4. **Scripts saved in container** - Persist in container filesystem

---

## Quick Start

```bash
# 1. Enter container
docker exec -it rocm /bin/bash

# 2. Setup (inside container)
pip install transformers datasets peft accelerate trl
mkdir -p /root/scripts/training
cd /root/scripts/training
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# 3. Run training
python3 finetune_qwen_law_single_gpu.py
```

---

## Verify GPU Works

```bash
# Inside container
docker exec -it rocm bash << 'EOF'
python3 << 'PYTHON'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
PYTHON
EOF
```

**Enter the Docker container first, then everything should work!** 🚀

