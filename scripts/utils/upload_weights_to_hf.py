#!/usr/bin/env python3
"""
Upload Model Weights to Hugging Face Hub
Model weights are NOT in GitHub (too large), upload to Hugging Face instead
"""
import os
import sys

print("=" * 80)
print("MODEL WEIGHTS: NOT IN GITHUB")
print("=" * 80)

print("""
WHY MODEL WEIGHTS ARE NOT IN GITHUB:
====================================

✅ Model weights are EXCLUDED (in .gitignore):
   - *.pt, *.bin, *.safetensors files
   - checkpoint-*/ directories
   - Too large for GitHub (100MB+ limit)

📍 Where are your model weights?
   - On server: /qwen2.5-32b-law-finetuned/checkpoint-500/
   - Size: ~100-500MB (LoRA adapters only)
   - Base model: ~62GB (on server cache)

OPTIONS TO SAVE MODEL WEIGHTS:
===============================

OPTION 1: Upload to Hugging Face Hub (Recommended)
---------------------------------------------------

This uploads your fine-tuned LoRA adapters to Hugging Face:

1. Create Hugging Face account: https://huggingface.co/join
2. Create repository: https://huggingface.co/new
   - Name: Arnie016/qwen2.5-32b-law-finetuned
   - Type: Model
   - Visibility: Private (or Public)

3. On server, upload model:

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

# Install huggingface_hub if needed
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login
# Enter your token: https://huggingface.co/settings/tokens

# Upload checkpoint
python3 << 'PYTHON'
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "/qwen2.5-32b-law-finetuned/checkpoint-500")

# Merge and save (optional - uploads merged model)
print("Uploading to Hugging Face...")
model.push_to_hub("Arnie016/qwen2.5-32b-law-finetuned", token="YOUR_TOKEN")

# Or upload just LoRA adapter
model.save_pretrained("./temp_lora")
from huggingface_hub import upload_folder
upload_folder(
    folder_path="./temp_lora",
    repo_id="Arnie016/qwen2.5-32b-law-finetuned",
    token="YOUR_TOKEN"
)
print("✅ Uploaded!")
PYTHON
```

OPTION 2: Use Git LFS (Advanced)
----------------------------------

If you really want weights in GitHub:

```bash
# Install git-lfs
brew install git-lfs  # Mac
# or: apt-get install git-lfs  # Linux

# Initialize
git lfs install
git lfs track "*.pt"
git lfs track "*.bin"
git lfs track "*.safetensors"

# Add and commit
git add .gitattributes
git add checkpoint-500/
git commit -m "Add model weights"
git push
```

⚠️  WARNING: Git LFS costs money for large files on GitHub!

OPTION 3: Transfer Separately (For New Server)
------------------------------------------------

Use the transfer script:
  ./quick_transfer.sh NEW_SERVER_IP

This transfers checkpoint-500 directly (not via GitHub).

OPTION 4: Store on Server Only
-------------------------------

Keep weights on server, only push code to GitHub:
  ✅ Code: GitHub
  ✅ Weights: Server only
  ✅ Transfer weights when needed: Use transfer script

RECOMMENDED APPROACH:
=====================

✅ Code & Scripts: GitHub (already done!)
✅ Model Weights: Hugging Face Hub (best option)
✅ Evaluation Results: GitHub (can add CSV if small)

This way:
- Code is versioned on GitHub
- Weights are on Hugging Face (free, versioned)
- Easy to share and download later
""")

print("\n" + "=" * 80)
print("CURRENT STATUS")
print("=" * 80)
print("""
✅ Code & Scripts: On GitHub (https://github.com/Arnie016/Law_Qwen)
❌ Model Weights: On server only (/qwen2.5-32b-law-finetuned/checkpoint-500/)
📊 Evaluation Results: On server only (legal_eval_results.csv)

Next step: Upload weights to Hugging Face Hub!
""")

