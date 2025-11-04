#!/bin/bash
# Quick setup script for DigitalOcean GPU Droplet
# Run this on your droplet after SSH connection

echo "=========================================="
echo "Single GPU Droplet Setup"
echo "=========================================="

# Check GPU
echo "Checking GPU..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} - {props.total_memory/1e9:.1f} GB")
else:
    print("❌ No GPU detected")
EOF

# Create directory
mkdir -p /root/scripts/training
cd /root/scripts/training

# Download training script
echo "Downloading training script..."
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py

chmod +x finetune_qwen_law_single_gpu.py

echo "✅ Setup complete!"
echo ""
echo "To start training:"
echo "  cd /root/scripts/training"
echo "  python3 finetune_qwen_law_single_gpu.py"
echo ""
echo "Or run in background:"
echo "  nohup python3 finetune_qwen_law_single_gpu.py > training.log 2>&1 &"
echo "  tail -f training.log"

