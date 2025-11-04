#!/bin/bash
# Setup workspace on AMD DevCloud server
# Run inside ROCm container: docker exec -it rocm /bin/bash

echo "=========================================="
echo "Setting up workspace on server"
echo "=========================================="

# Create directory structure
echo "Creating workspace directories..."
mkdir -p ~/workspace/{models,datasets,code,outputs,projects}
mkdir -p ~/workspace/models/{huggingface,custom}
mkdir -p ~/workspace/datasets/{huggingface,custom}

# Set environment variables
echo ""
echo "Setting environment variables..."
export HF_HOME=~/workspace/models/huggingface
export TRANSFORMERS_CACHE=~/workspace/models/huggingface
export HF_DATASETS_CACHE=~/workspace/datasets/huggingface

# Add to .bashrc
if ! grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Hugging Face cache directories" >> ~/.bashrc
    echo "export HF_HOME=~/workspace/models/huggingface" >> ~/.bashrc
    echo "export TRANSFORMERS_CACHE=~/workspace/models/huggingface" >> ~/.bashrc
    echo "export HF_DATASETS_CACHE=~/workspace/datasets/huggingface" >> ~/.bashrc
fi

# Install essential packages
echo ""
echo "Installing essential packages..."
pip install transformers diffusers accelerate datasets huggingface_hub --quiet

# Check disk space
echo ""
echo "=========================================="
echo "Disk Space Check"
echo "=========================================="
df -h | head -3

# Check current workspace
echo ""
echo "=========================================="
echo "Workspace Structure"
echo "=========================================="
tree -L 2 ~/workspace 2>/dev/null || ls -la ~/workspace

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Workspace locations:"
echo "  Models:    ~/workspace/models/"
echo "  Datasets:  ~/workspace/datasets/"
echo "  Code:      ~/workspace/code/"
echo "  Outputs:   ~/workspace/outputs/"
echo ""
echo "To use custom cache locations, run:"
echo "  source ~/.bashrc"
echo ""
echo "Or set manually:"
echo "  export HF_HOME=~/workspace/models/huggingface"
echo "  export TRANSFORMERS_CACHE=~/workspace/models/huggingface"
echo "  export HF_DATASETS_CACHE=~/workspace/datasets/huggingface"


