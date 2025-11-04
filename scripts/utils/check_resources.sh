#!/bin/bash
# Check if safe to download another model

echo "=" * 60
echo "SYSTEM RESOURCE CHECK"
echo "=" * 60

# Check disk space
echo ""
echo "💿 Disk Space:"
df -h / | tail -1

# Check GPU memory usage
echo ""
echo "🎮 GPU Memory Usage:"
rocm-smi --showmeminfo | head -20

# Check current model sizes
echo ""
echo "📦 Current Model Sizes:"
du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null | head -5 | sort -h

# Check training checkpoint size
echo ""
echo "📁 Training Checkpoint Size:"
if [ -d "/qwen2.5-32b-law-finetuned" ]; then
    du -sh /qwen2.5-32b-law-finetuned/
fi

# Check available space
echo ""
echo "✅ Available Space Analysis:"
AVAILABLE=$(df / | tail -1 | awk '{print $4}')
echo "Available: $AVAILABLE"
echo ""
echo "Large models require:"
echo "- Qwen 2.5 32B: ~66GB"
echo "- Qwen 2.5 72B: ~144GB"
echo "- DeepSeek-V3 671B: ~400GB+"
