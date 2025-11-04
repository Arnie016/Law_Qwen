# Checkpoint Location & Download Guide

## Checkpoint Status

### ❌ Checkpoint NOT in GitHub
- Checkpoints are **excluded** in `.gitignore` (line 23: `checkpoint-*/`)
- Exception: `!models/checkpoints/checkpoint-*/` (allows if in `models/checkpoints/`)
- **Current status:** Checkpoint is on server, NOT in GitHub

---

## Where is Checkpoint-500?

### Option 1: On Original Server
```
Location: /qwen2.5-32b-law-finetuned/checkpoint-500/
Server: 134.199.192.60 (old server)
Status: Probably still there
```

### Option 2: In GitHub (if uploaded)
```
Location: models/checkpoints/checkpoint-500/
Status: Check if exists: git ls-files models/checkpoints/
```

### Option 3: On New Server (if transferred)
```
Location: /root/qwen2.5-32b-law-finetuned/checkpoint-500/
Server: 129.212.184.211 (new ROCm droplet)
Status: Check if transferred
```

---

## How to Download Checkpoint to Container

### If on Old Server:

```bash
# From NEW server, download from old server
ssh root@134.199.192.60 "docker exec rocm tar -czf - /qwen2.5-32b-law-finetuned/checkpoint-500" | \
docker exec -i rocm bash -c "cd /root && tar -xzf -"
```

### If in GitHub (models/checkpoints):

```bash
# Inside container
cd /root
git clone https://github.com/Arnie016/Law_Qwen.git temp_repo
cp -r temp_repo/models/checkpoints/checkpoint-500 ./qwen2.5-32b-law-finetuned/
rm -rf temp_repo
```

### If Uploaded to Hugging Face:

```bash
# Inside container
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Arnie016/qwen2.5-32b-law-finetuned", 
                  local_dir="./qwen2.5-32b-law-finetuned")
```

---

## Check If Checkpoint Exists

```bash
# Inside container
ls -lh /root/qwen2.5-32b-law-finetuned/checkpoint-500/ 2>/dev/null || \
ls -lh /qwen2.5-32b-law-finetuned/checkpoint-500/ 2>/dev/null || \
echo "Checkpoint not found"
```

---

## Recommended: Start Fresh GRPO

Since checkpoint might not exist, **starting fresh GRPO is fine**:
- GRPO will train base model
- Still effective (RL doesn't need pre-training)
- Faster setup (no checkpoint download needed)

**The script handles this gracefully - it will start from base model if checkpoint missing.**

