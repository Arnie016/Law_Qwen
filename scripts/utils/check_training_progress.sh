#!/bin/bash
# Check training progress - Quick commands

echo "=" * 60
echo "TRAINING PROGRESS CHECK"
echo "=" * 60

# Check latest checkpoint
echo ""
echo "📁 Latest Checkpoint:"
ls -d /qwen2.5-32b-law-finetuned/checkpoint-* 2>/dev/null | sort -V | tail -1

# Check all checkpoints
echo ""
echo "📊 All Checkpoints:"
ls -d /qwen2.5-32b-law-finetuned/checkpoint-* 2>/dev/null | sort -V | tail -5

# Check checkpoint size
echo ""
echo "💾 Checkpoint Size:"
LATEST=$(ls -d /qwen2.5-32b-law-finetuned/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ ! -z "$LATEST" ]; then
    du -sh "$LATEST"
fi

# Check training logs (if saved)
echo ""
echo "📝 Training Logs:"
if [ -f "/qwen2.5-32b-law-finetuned/training.log" ]; then
    tail -20 /qwen2.5-32b-law-finetuned/training.log
else
    echo "No training.log found"
fi

# Check GPU usage
echo ""
echo "🎮 GPU Usage:"
rocm-smi --showuse 2>/dev/null || nvidia-smi 2>/dev/null || echo "GPU info not available"

# Check if training process is running
echo ""
echo "🔍 Training Process:"
ps aux | grep -i "python.*train" | grep -v grep || echo "No training process found"

# Check disk space
echo ""
echo "💿 Disk Space:"
df -h / | tail -1

