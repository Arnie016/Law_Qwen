# Add Model Weights to GitHub

## Step 1: Install Git LFS

```bash
# Mac
brew install git-lfs

# Linux
sudo apt-get install git-lfs

# Verify
git lfs version
```

## Step 2: Setup Git LFS in Repository

```bash
cd /Users/hema/Desktop/AMD

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "models/checkpoints/**/*.pt"
git lfs track "models/checkpoints/**/*.bin"
git lfs track "models/checkpoints/**/*.safetensors"
git lfs track "models/checkpoints/**/*.json"

# Commit .gitattributes
git add .gitattributes
git commit -m "Setup Git LFS for model weights"
```

## Step 3: Download Weights from Server

```bash
# Run download script
./download_weights.sh

# Or manually:
ssh -i ~/.ssh/id_ed25519 root@134.199.192.60 "docker exec rocm tar -czf - /qwen2.5-32b-law-finetuned/checkpoint-500" > models/checkpoints/checkpoint-500.tar.gz
cd models/checkpoints
tar -xzf checkpoint-500.tar.gz
rm checkpoint-500.tar.gz
cd ../..
```

## Step 4: Add and Push Weights

```bash
# Add weights
git add models/checkpoints/

# Commit
git commit -m "Add model weights (checkpoint-500)"

# Push (this will upload via Git LFS)
git push
```

## Alternative: Upload to Hugging Face Instead

If Git LFS is problematic, upload to Hugging Face:

```bash
# On server
cd scripts/utils
./upload_weights_to_hf.sh Arnie016/qwen2.5-32b-law-finetuned YOUR_HF_TOKEN
```

Then add link in README pointing to Hugging Face model.

