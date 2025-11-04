#!/bin/bash
# Transfer Everything to New Server
# Guide for migrating from current server to new 8-node GPU instance

echo "=========================================="
echo "SERVER MIGRATION GUIDE"
echo "=========================================="

cat << 'EOF'

STEP 1: Export Model Checkpoints from Current Server
====================================================

# On CURRENT server (134.199.192.60)
docker exec -it rocm /bin/bash

# List all checkpoints
ls -lh /qwen2.5-32b-law-finetuned/checkpoint-*/

# Create backup directory
mkdir -p /root/backup
cd /root/backup

# Option A: Copy entire checkpoint directory
cp -r /qwen2.5-32b-law-finetuned ./qwen2.5-32b-law-finetuned-backup

# Option B: Copy just latest checkpoint (smaller)
cp -r /qwen2.5-32b-law-finetuned/checkpoint-500 ./checkpoint-500-backup

# Also backup tokenizer
cp -r /qwen2.5-32b-law-finetuned/*.json ./ 2>/dev/null || true
cp -r /qwen2.5-32b-law-finetuned/*.txt ./ 2>/dev/null || true

# Exit container
exit

# From host, compress backup
cd /root
tar -czf law_finetuning_backup.tar.gz backup/

# Check size
du -sh law_finetuning_backup.tar.gz


STEP 2: Transfer Files to New Server
=====================================

# On LOCAL machine (your Mac)
# Replace NEW_SERVER_IP with your new server IP

# Transfer backup
scp -i ~/.ssh/id_ed25519 \
    root@134.199.192.60:/root/law_finetuning_backup.tar.gz \
    root@NEW_SERVER_IP:/root/

# Transfer Git repo (if you want scripts)
scp -i ~/.ssh/id_ed25519 \
    -r /Users/hema/Desktop/AMD/* \
    root@NEW_SERVER_IP:/root/amd_project/


STEP 3: Setup New Server
==========================

# SSH into NEW server
ssh -i ~/.ssh/id_ed25519 root@NEW_SERVER_IP

# Extract backup
cd /root
tar -xzf law_finetuning_backup.tar.gz

# Enter Docker container (same as before)
docker exec -it rocm /bin/bash

# Copy checkpoint to container
docker cp /root/backup/checkpoint-500-backup rocm:/qwen2.5-32b-law-finetuned/checkpoint-500

# Or copy entire directory
docker cp /root/backup/qwen2.5-32b-law-finetuned-backup rocm:/qwen2.5-32b-law-finetuned


STEP 4: Verify Installation
============================

# Inside container
ls -lh /qwen2.5-32b-law-finetuned/checkpoint-500/

# Test loading model
python3 << 'PYTHON'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob

# Load base model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load checkpoint
checkpoints = sorted(glob.glob("/qwen2.5-32b-law-finetuned/checkpoint-*"))
if checkpoints:
    latest = checkpoints[-1]
    print(f"Loading checkpoint: {latest}")
    model = PeftModel.from_pretrained(base_model, latest)
    print("✅ Model loaded successfully!")
else:
    print("❌ No checkpoint found")
PYTHON


STEP 5: Transfer Hugging Face Cache (Optional)
==============================================

# If you want to transfer downloaded models to avoid re-downloading

# On CURRENT server (from host)
cd /root
tar -czf huggingface_cache.tar.gz .cache/huggingface/

# Transfer (this will be large, ~60GB+)
scp -i ~/.ssh/id_ed25519 \
    root@134.199.192.60:/root/huggingface_cache.tar.gz \
    root@NEW_SERVER_IP:/root/

# On NEW server
cd /root
tar -xzf huggingface_cache.tar.gz
# Cache will be in /root/.cache/huggingface/


STEP 6: Transfer Evaluation Results
=====================================

# On CURRENT server
scp -i ~/.ssh/id_ed25519 \
    root@134.199.192.60:/root/legal_eval_results.csv \
    root@NEW_SERVER_IP:/root/

# Or from inside container
docker exec rocm cat /root/legal_eval_results.csv > legal_eval_results.csv
scp -i ~/.ssh/id_ed25519 legal_eval_results.csv root@NEW_SERVER_IP:/root/


QUICK TRANSFER COMMAND (All-in-One)
====================================

# On LOCAL machine, create this script:

#!/bin/bash
OLD_SERVER="134.199.192.60"
NEW_SERVER="NEW_SERVER_IP"  # Replace with actual IP
SSH_KEY="~/.ssh/id_ed25519"

echo "Transferring files..."

# 1. Create backup on old server
ssh -i $SSH_KEY root@$OLD_SERVER << 'SSH'
docker exec rocm bash -c "cd /root && tar -czf law_backup.tar.gz qwen2.5-32b-law-finetuned/ 2>/dev/null || echo 'No checkpoint dir'"
docker exec rocm bash -c "cd /root && tar -czf eval_results.tar.gz legal_eval_results.csv 2>/dev/null || echo 'No eval results'"
exit
SSH

# 2. Transfer backups
scp -i $SSH_KEY root@$OLD_SERVER:/root/law_backup.tar.gz root@$NEW_SERVER:/root/
scp -i $SSH_KEY root@$OLD_SERVER:/root/eval_results.tar.gz root@$NEW_SERVER:/root/

# 3. Transfer scripts
scp -i $SSH_KEY -r /Users/hema/Desktop/AMD/*.py root@$NEW_SERVER:/root/scripts/
scp -i $SSH_KEY -r /Users/hema/Desktop/AMD/*.md root@$NEW_SERVER:/root/scripts/
scp -i $SSH_KEY -r /Users/hema/Desktop/AMD/*.sh root@$NEW_SERVER:/root/scripts/

# 4. Extract on new server
ssh -i $SSH_KEY root@$NEW_SERVER << 'SSH'
cd /root
tar -xzf law_backup.tar.gz
tar -xzf eval_results.tar.gz
docker cp qwen2.5-32b-law-finetuned rocm:/qwen2.5-32b-law-finetuned
docker cp legal_eval_results.csv rocm:/root/
echo "✅ Transfer complete!"
SSH

echo "Done!"


IMPORTANT NOTES:
================

1. Model Checkpoints:
   - Latest checkpoint: checkpoint-500
   - Location: /qwen2.5-32b-law-finetuned/checkpoint-500/
   - Size: ~100-500MB (LoRA adapters only)

2. Base Model:
   - Will need to download again OR transfer cache
   - Location: ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/
   - Size: ~62GB

3. Evaluation Results:
   - CSV file: legal_eval_results.csv
   - Contains all comparison data

4. Scripts:
   - All Python scripts are in your local AMD folder
   - Transfer those to new server

5. Docker Container:
   - New server should have same ROCm container setup
   - Same commands: docker exec -it rocm /bin/bash

EOF

echo ""
echo "=========================================="
echo "Ready to transfer!"
echo "=========================================="

