#!/bin/bash
# Upload Model Weights to Hugging Face Hub
# Run this on the SERVER (inside Docker container)

set -e

REPO_NAME="${1:-Arnie016/qwen2.5-32b-law-finetuned}"
HF_TOKEN="${2:-}"

if [ -z "$HF_TOKEN" ]; then
    echo "=========================================="
    echo "UPLOAD MODEL WEIGHTS TO HUGGING FACE"
    echo "=========================================="
    echo ""
    echo "Usage: $0 REPO_NAME HF_TOKEN"
    echo ""
    echo "Example:"
    echo "  $0 Arnie016/qwen2.5-32b-law-finetuned hf_xxxxxxxxxxxxx"
    echo ""
    echo "Steps:"
    echo "1. Create repo: https://huggingface.co/new"
    echo "   Name: qwen2.5-32b-law-finetuned"
    echo "   Type: Model"
    echo ""
    echo "2. Get token: https://huggingface.co/settings/tokens"
    echo "   Create token with 'write' permission"
    echo ""
    echo "3. Run this script with your token"
    exit 1
fi

echo "=========================================="
echo "Uploading to: $REPO_NAME"
echo "=========================================="

# Check if inside Docker
if [ ! -f /.dockerenv ]; then
    echo "⚠️  Not inside Docker container!"
    echo "Run: docker exec -it rocm /bin/bash"
    exit 1
fi

# Install dependencies
echo "[1/5] Installing dependencies..."
pip install huggingface_hub --quiet

# Check checkpoint exists
echo "[2/5] Checking checkpoint..."
if [ ! -d "/qwen2.5-32b-law-finetuned/checkpoint-500" ]; then
    echo "❌ Checkpoint not found!"
    echo "Expected: /qwen2.5-32b-law-finetuned/checkpoint-500"
    exit 1
fi
echo "✅ Checkpoint found"

# Upload using Python
echo "[3/5] Uploading model..."
python3 << PYTHON
from huggingface_hub import HfApi, upload_folder
import os

repo_id = "$REPO_NAME"
token = "$HF_TOKEN"
checkpoint_path = "/qwen2.5-32b-law-finetuned/checkpoint-500"

print(f"Uploading {checkpoint_path} to {repo_id}...")

# Upload entire checkpoint folder
upload_folder(
    folder_path=checkpoint_path,
    repo_id=repo_id,
    token=token,
    commit_message="Upload checkpoint-500 LoRA adapters"
)

print("✅ Upload complete!")
PYTHON

echo "[4/5] Verifying upload..."
python3 << PYTHON
from huggingface_hub import HfApi
api = HfApi(token="$HF_TOKEN")
files = api.list_repo_files("$REPO_NAME")
print("Files in repo:")
for f in files[:10]:
    print(f"  - {f}")
PYTHON

echo ""
echo "[5/5] ✅ SUCCESS!"
echo ""
echo "Model weights uploaded to:"
echo "  https://huggingface.co/$REPO_NAME"
echo ""
echo "To download later:"
echo "  from peft import PeftModel"
echo "  model = PeftModel.from_pretrained(base_model, \"$REPO_NAME\")"
echo ""

