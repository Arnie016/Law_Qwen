#!/bin/bash
# Check if law model training is complete

echo "=" * 60
echo "CHECKING LAW MODEL TRAINING STATUS"
echo "=" * 60

# Check latest checkpoint
echo ""
echo "📁 Latest Checkpoint:"
LATEST=$(ls -d /qwen2.5-32b-law-finetuned/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ ! -z "$LATEST" ]; then
    echo "✅ Found: $LATEST"
    STEP=$(echo $LATEST | grep -o '[0-9]*$')
    echo "   Step: $STEP / 500"
    
    # Check if training completed (500 steps)
    if [ "$STEP" -ge 500 ]; then
        echo "   ✅ TRAINING COMPLETE! (500/500 steps)"
    else
        echo "   ⏳ Training in progress ($STEP/500)"
        PERCENT=$((STEP * 100 / 500))
        echo "   Progress: $PERCENT%"
    fi
else
    echo "❌ No checkpoints found"
fi

# Check all checkpoints
echo ""
echo "📊 All Checkpoints:"
ls -d /qwen2.5-32b-law-finetuned/checkpoint-* 2>/dev/null | sort -V | tail -5

# Check if training is still running
echo ""
echo "🔍 Training Process:"
ps aux | grep -i "python.*train" | grep -v grep || echo "No training process found (may have completed)"

# Check for final model
echo ""
echo "💾 Final Model:"
if [ -f "/qwen2.5-32b-law-finetuned/adapter_config.json" ]; then
    echo "✅ Final model files found"
    ls -lh /qwen2.5-32b-law-finetuned/*.json 2>/dev/null | head -3
else
    echo "⚠️  No final model files yet (checkpoints only)"
fi

# Check GPU usage
echo ""
echo "🎮 GPU Usage:"
rocm-smi --showuse 2>/dev/null | head -15 || echo "Cannot check GPU"

