# Server Setup Guide - Download Everything on Server

Yes, you can download datasets, code, and models all on the AMD DevCloud server.

## Workspace Organization

Create a clean workspace structure:

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

# Create workspace
mkdir -p ~/workspace/{models,datasets,code,outputs}
cd ~/workspace
```

## 1. Download Models

Models download automatically via Hugging Face, but you can pre-download:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download models to specific directory
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir ~/workspace/models/Qwen2.5-14B
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ~/workspace/models/SDXL
```

Or use Python:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# Model downloads to ~/.cache/huggingface/ by default
# You can change cache dir with HF_HOME env var
```

## 2. Download Datasets

### Hugging Face Datasets

```bash
pip install datasets

python3 << EOF
from datasets import load_dataset

# Download dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
# Saves to ~/.cache/huggingface/datasets/
EOF
```

### Custom Datasets

```bash
# Download via wget/curl
cd ~/workspace/datasets
wget https://example.com/dataset.zip
unzip dataset.zip

# Or use git
git clone https://github.com/user/dataset-repo.git
```

## 3. Download Code/Repositories

```bash
cd ~/workspace/code

# Clone repositories
git clone https://github.com/huggingface/transformers.git
git clone https://github.com/hpcaitech/Open-Sora.git
git clone https://github.com/your-repo/your-project.git

# Install in development mode
cd transformers
pip install -e .
```

## 4. Set Environment Variables

```bash
# Set cache directories (optional, to organize better)
export HF_HOME=~/workspace/models/huggingface
export TRANSFORMERS_CACHE=~/workspace/models/huggingface
export HF_DATASETS_CACHE=~/workspace/datasets/huggingface

# Add to ~/.bashrc to persist
echo 'export HF_HOME=~/workspace/models/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=~/workspace/models/huggingface' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=~/workspace/datasets/huggingface' >> ~/.bashrc
```

## 5. Check Disk Space

```bash
# Check available space
df -h

# Check what's using space
du -sh ~/workspace/*

# Check model cache size
du -sh ~/.cache/huggingface/
```

## 6. Example: Full Setup Workflow

```bash
# 1. Create workspace
mkdir -p ~/workspace/{models,datasets,code,outputs}
cd ~/workspace

# 2. Install everything
pip install transformers diffusers accelerate datasets huggingface_hub

# 3. Download a model (example)
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-14B-Instruct"
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="~/workspace/models"  # Custom cache location
)
print("Model downloaded!")
EOF

# 4. Download a dataset
python3 << EOF
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="~/workspace/datasets")
print("Dataset downloaded!")
EOF

# 5. Clone code repository
cd ~/workspace/code
git clone https://github.com/huggingface/transformers.git
```

## 7. Organize Projects

```bash
# Create project structure
mkdir -p ~/workspace/projects/my_project/{models,data,scripts,outputs}
cd ~/workspace/projects/my_project

# Create requirements.txt
cat > requirements.txt << EOF
torch>=2.0.0
transformers>=4.35.0
diffusers>=0.21.0
accelerate>=0.24.0
EOF

# Install project dependencies
pip install -r requirements.txt
```

## Storage Locations

- **Models**: `~/.cache/huggingface/hub/` (default) or `~/workspace/models/`
- **Datasets**: `~/.cache/huggingface/datasets/` (default) or `~/workspace/datasets/`
- **Code**: `~/workspace/code/` or `~/workspace/projects/`
- **Outputs**: `~/workspace/outputs/`

## Notes

- Everything downloads to the server (not your local machine)
- Models can be 10-140GB each, so monitor disk space
- Datasets can also be large (GBs to TBs)
- Use `nohup` for long downloads: `nohup python3 download_model.py &`
- Check space before downloading: `df -h`


