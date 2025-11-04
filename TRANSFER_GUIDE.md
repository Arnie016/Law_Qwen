# Transfer Guide - Quick Reference

## Quick Transfer (Fastest Method)

### On your LOCAL machine:

```bash
# Make script executable
chmod +x quick_transfer.sh

# Run transfer (replace NEW_SERVER_IP)
./quick_transfer.sh NEW_SERVER_IP
```

## Manual Transfer Steps

### 1. Create Backup on Current Server

```bash
# SSH into current server
ssh -i ~/.ssh/id_ed25519 root@134.199.192.60

# Inside container
docker exec -it rocm /bin/bash
cd /root
tar -czf law_backup.tar.gz qwen2.5-32b-law-finetuned/
tar -czf eval_results.tar.gz legal_eval_results.csv
exit

# Back on host
cd /root
docker cp rocm:/root/law_backup.tar.gz ./
docker cp rocm:/root/eval_results.tar.gz ./
```

### 2. Transfer Files

```bash
# From LOCAL machine
scp -i ~/.ssh/id_ed25519 \
    root@134.199.192.60:/root/law_backup.tar.gz \
    root@NEW_SERVER_IP:/root/

scp -i ~/.ssh/id_ed25519 \
    root@134.199.192.60:/root/eval_results.tar.gz \
    root@NEW_SERVER_IP:/root/

# Transfer scripts
scp -i ~/.ssh/id_ed25519 \
    -r /Users/hema/Desktop/AMD/*.py \
    root@NEW_SERVER_IP:/root/scripts/
```

### 3. Setup New Server

```bash
# SSH into NEW server
ssh -i ~/.ssh/id_ed25519 root@NEW_SERVER_IP

# Extract
cd /root
tar -xzf law_backup.tar.gz
tar -xzf eval_results.tar.gz

# Copy into container
docker cp qwen2.5-32b-law-finetuned rocm:/qwen2.5-32b-law-finetuned
docker cp legal_eval_results.csv rocm:/root/

# Verify
docker exec -it rocm /bin/bash
ls -lh /qwen2.5-32b-law-finetuned/checkpoint-*/
```

## What Gets Transferred

### ✅ Small Files (Quick Transfer):
- **Checkpoint-500**: ~100-500MB (LoRA adapters only)
- **Evaluation results**: CSV file (~50KB)
- **Scripts**: All Python/Markdown files

### ⚠️ Large Files (Optional):
- **Base Model**: ~62GB (can re-download or transfer cache)
- **Hugging Face Cache**: ~60GB+ (optional, speeds up setup)

## Transfer Base Model Cache (Optional)

If you want to avoid re-downloading the base model:

```bash
# On CURRENT server
cd /root
tar -czf huggingface_cache.tar.gz .cache/huggingface/

# Transfer (this takes time, ~60GB)
scp -i ~/.ssh/id_ed25519 \
    root@134.199.192.60:/root/huggingface_cache.tar.gz \
    root@NEW_SERVER_IP:/root/

# On NEW server
cd /root
tar -xzf huggingface_cache.tar.gz
# Cache now in /root/.cache/huggingface/
```

## Verify Everything Works

```bash
# On NEW server, inside container
docker exec -it rocm /bin/bash

# Test loading model
python3 << 'PYTHON'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

checkpoints = sorted(glob.glob("/qwen2.5-32b-law-finetuned/checkpoint-*"))
if checkpoints:
    latest = checkpoints[-1]
    model = PeftModel.from_pretrained(base_model, latest)
    print("✅ Model loaded successfully!")
else:
    print("❌ No checkpoint found")
PYTHON
```

## File Sizes

| Item | Size | Transfer Time |
|------|------|---------------|
| Checkpoint-500 | ~100-500MB | ~1-5 min |
| Eval Results | ~50KB | <1 sec |
| Scripts | ~5MB | <1 sec |
| Base Model Cache | ~62GB | ~30-60 min |
| Total (minimal) | ~500MB | ~5 min |
| Total (with cache) | ~62GB | ~1 hour |

## Quick Commands

```bash
# Transfer everything (minimal)
./quick_transfer.sh NEW_SERVER_IP

# Transfer with base model cache (full)
# Follow manual steps above for cache transfer
```

## Troubleshooting

**Problem:** Docker container not found
```bash
# Check container name
docker ps
# Use correct name: docker exec -it CONTAINER_NAME /bin/bash
```

**Problem:** Permission denied
```bash
# Make sure SSH key is correct
ssh -i ~/.ssh/id_ed25519 root@NEW_SERVER_IP
```

**Problem:** Checkpoint not found
```bash
# Check if backup was created
docker exec rocm ls -lh /root/qwen2.5-32b-law-finetuned/
```

Ready to transfer! Use `./quick_transfer.sh NEW_SERVER_IP`

