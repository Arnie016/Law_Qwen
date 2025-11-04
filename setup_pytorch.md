# Install PyTorch in ROCm Container

Your ROCm container doesn't have PyTorch installed. Here's how to install it.

## Quick Install

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

# Install PyTorch with ROCm support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

If that doesn't work, try:

```bash
# Generic install (may work)
pip3 install torch torchvision torchaudio
```

## Verify Installation

```bash
python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} - {props.total_memory/1e9:.2f} GB")
EOF
```

## Check ROCm Version First

```bash
# Check ROCm version to get correct PyTorch
rocminfo | grep -i "rocm version"

# Or check what's available
python3 -m pip index versions torch
```

## Alternative: Check Container Image

```bash
# See what's in the container
cat /etc/os-release
python3 --version
pip3 list | grep torch
```

## Full Setup Script

Run the setup script:

```bash
# Copy script to container first, or paste contents
bash setup_pytorch.sh
```

## Expected Output After Install

```
PyTorch version: 2.x.x+rocm6.0
CUDA available: True
GPU count: 8
GPU 0: MI300X (or similar)
  Memory: 192.00 GB
...
```

## Troubleshooting

If installation fails:

1. **Check pip version**: `pip3 --version`
2. **Upgrade pip**: `python3 -m pip install --upgrade pip`
3. **Try different ROCm version**: Change `rocm6.0` to `rocm5.7` or `rocm5.6`
4. **Check container image**: The container may have a specific PyTorch pre-installed in a different location


