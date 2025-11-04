# Single GPU Droplet Setup Guide

## 🚀 Quick Start

### Step 1: Connect to Droplet

```bash
# SSH into your droplet
ssh root@134.199.195.11

# Or if you have SSH key
ssh -i ~/.ssh/id_ed25519 root@134.199.195.11
```

### Step 2: Check GPU Status

```bash
# Check if GPU is available
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} - {props.total_memory/1e9:.1f} GB")
EOF
```

### Step 3: Install Dependencies (if needed)

```bash
# Update system
apt update && apt upgrade -y

# Install Python dependencies
pip3 install torch transformers datasets peft accelerate
```

### Step 4: Download Training Script

```bash
# Create scripts directory
mkdir -p /root/scripts/training
cd /root/scripts/training

# Download single GPU script from GitHub
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

# Make executable
chmod +x finetune_qwen_law_single_gpu.py
```

### Step 5: Run Training

```bash
# Run single GPU training
python3 finetune_qwen_law_single_gpu.py

# Or run in background
nohup python3 finetune_qwen_law_single_gpu.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

---

## 🔧 Full Setup (If Starting Fresh)

### Option A: Using Docker/ROCm Container

```bash
# Check if Docker is installed
docker --version

# If ROCm container exists, enter it
docker exec -it rocm /bin/bash

# Otherwise, install Docker if needed
apt install docker.io -y
systemctl start docker
```

### Option B: Direct Python Setup

```bash
# Install Python 3.10+
apt install python3 python3-pip -y

# Install PyTorch (with ROCm support for AMD)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install transformers and other dependencies
pip3 install transformers datasets peft accelerate bitsandbytes

# Verify installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 📋 Transfer Files from Local Machine

### Upload Training Script

```bash
# From your LOCAL Mac:
scp -i ~/.ssh/id_ed25519 \
    /Users/hema/Desktop/AMD/scripts/training/finetune_qwen_law_single_gpu.py \
    root@134.199.195.11:/root/
```

### Upload Entire Project (Optional)

```bash
# From your LOCAL Mac:
scp -i ~/.ssh/id_ed25519 \
    -r /Users/hema/Desktop/AMD/* \
    root@134.199.195.11:/root/amd_project/
```

---

## 🎯 Run Training Immediately

### Quick One-Liner (Copy-Paste)

```bash
# SSH into droplet
ssh root@134.199.195.11

# Then run:
cd /root && \
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py && \
python3 finetune_qwen_law_single_gpu.py
```

---

## ⚡ Expected Performance

### Single GPU (MI300X, 192GB)

- **Model:** Qwen 2.5 32B (~31GB with bfloat16)
- **Training Speed:** ~2-4 hours for 500 steps
- **Memory Usage:** ~35-40GB / 192GB (~20%)
- **Training Time (10k steps):** ~40-80 hours (overnight/weekend)

---

## 🛠️ Troubleshooting

### Check GPU Status

```bash
# ROCm check
rocminfo | grep -i gpu

# PyTorch check
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Check Disk Space

```bash
df -h
```

### Check Running Processes

```bash
# If training is running
ps aux | grep python

# Check GPU usage
nvidia-smi  # If NVIDIA
# Or ROCm equivalent
rocm-smi
```

### Monitor Training Progress

```bash
# If running in background
tail -f training.log

# Check checkpoint directory
ls -lh qwen2.5-32b-law-finetuned/checkpoint-*/
```

---

## 💰 Cost Management

**Current Cost:** $1.99/hour

- **500 steps:** ~4 hours = ~$8
- **10,000 steps:** ~80 hours = ~$160

**Tip:** Run training overnight and destroy droplet when done to save costs.

### Stop Droplet (Pause Billing)

```bash
# From DigitalOcean dashboard: Power Off
# Or via API/CLI
doctl compute droplet power-off <droplet-id>
```

### Destroy Droplet (Stop All Billing)

**⚠️ Warning:** This permanently deletes the droplet. Backup checkpoints first!

```bash
# Backup checkpoints first!
tar -czf checkpoints_backup.tar.gz qwen2.5-32b-law-finetuned/

# Download backup
scp root@134.199.195.11:/root/checkpoints_backup.tar.gz ./

# Then destroy from DigitalOcean dashboard
```

---

## 📝 Quick Reference

| Command | Purpose |
|---------|---------|
| `ssh root@134.199.195.11` | Connect to droplet |
| `python3 finetune_qwen_law_single_gpu.py` | Run training |
| `nohup python3 ... &` | Run in background |
| `tail -f training.log` | Monitor progress |
| `ls checkpoint-*/` | Check saved checkpoints |

---

## ✅ Verification Checklist

- [ ] Connected to droplet via SSH
- [ ] GPU detected (192GB MI300X)
- [ ] PyTorch installed and CUDA available
- [ ] Training script downloaded
- [ ] Training started (check logs)
- [ ] Checkpoints saving (checkpoint-50, checkpoint-100, etc.)

---

## 🚨 Important Notes

1. **Single GPU:** Script uses `device_map="cuda:0"` - will use only GPU 0
2. **Training Time:** Much slower than 8 GPUs, but more reliable
3. **Cost:** ~$1.99/hour - monitor usage
4. **Checkpoints:** Saved every 50 steps in `qwen2.5-32b-law-finetuned/checkpoint-*/`
5. **Base Model:** Will download ~62GB on first run (cached after)

---

Ready to train! 🚀

