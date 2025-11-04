#!/bin/bash
# Download Model Weights from Server and Add to GitHub
# Run this from your LOCAL machine

set -e

SERVER_IP="${1:-134.199.192.60}"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=========================================="
echo "DOWNLOADING MODEL WEIGHTS FROM SERVER"
echo "=========================================="
echo "Server: $SERVER_IP"
echo ""

# Create checkpoints directory
mkdir -p models/checkpoints

# Download checkpoint from server
echo "[1/3] Downloading checkpoint from server..."
ssh -i "$SSH_KEY" root@$SERVER_IP << 'SSH'
docker exec rocm bash -c "cd /root && tar -czf checkpoint-500.tar.gz qwen2.5-32b-law-finetuned/checkpoint-500/ 2>/dev/null || echo 'Checkpoint not found'"
docker cp rocm:/root/checkpoint-500.tar.gz /root/
echo "Checkpoint prepared"
SSH

scp -i "$SSH_KEY" root@$SERVER_IP:/root/checkpoint-500.tar.gz ./models/checkpoints/

# Extract
echo "[2/3] Extracting checkpoint..."
cd models/checkpoints
tar -xzf checkpoint-500.tar.gz
rm checkpoint-500.tar.gz
cd ../..

# Check what we got
echo "[3/3] Checking files..."
ls -lh models/checkpoints/qwen2.5-32b-law-finetuned/checkpoint-500/ | head -10

echo ""
echo "=========================================="
echo "MODEL WEIGHTS DOWNLOADED"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Install Git LFS: brew install git-lfs"
echo "2. Setup Git LFS: git lfs install"
echo "3. Track model files: git lfs track 'models/checkpoints/**/*.pt'"
echo "4. Add and commit: git add models/ && git commit -m 'Add model weights'"
echo "5. Push: git push"
echo ""

