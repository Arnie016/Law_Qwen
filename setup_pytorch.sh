#!/bin/bash
# Install PyTorch with ROCm support inside ROCm container
# Run this inside the container: docker exec -it rocm /bin/bash

echo "=========================================="
echo "Installing PyTorch with ROCm Support"
echo "=========================================="

# Check Python version
echo "Python version:"
python3 --version

# Check pip version
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with ROCm support
# Check ROCm version first
echo ""
echo "Checking ROCm version..."
if command -v rocminfo &> /dev/null; then
    rocm_version=$(rocminfo | grep -i "rocm version" | head -1 || echo "unknown")
    echo "ROCm: $rocm_version"
else
    echo "ROCm version check failed"
fi

echo ""
echo "Installing PyTorch (ROCm 6.0 compatible)..."
# Install PyTorch with ROCm 6.0 support
# Adjust version based on actual ROCm version in container
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Alternative if above doesn't work:
# pip3 install torch torchvision torchaudio

echo ""
echo "Verifying installation..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
EOF

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="


