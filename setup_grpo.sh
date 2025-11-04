#!/bin/bash
# GRPO Training Quick Start Script
# Installs dependencies and runs GRPO training

set -e

echo "=========================================="
echo "GRPO Training Setup"
echo "=========================================="

# Check if inside Docker container
if [ ! -f /.dockerenv ]; then
    echo "⚠️  Not inside Docker container."
    echo "Run: docker exec -it rocm /bin/bash"
    exit 1
fi

echo ""
echo "Step 1: Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
pip install --no-deps trl peft accelerate --quiet

echo "✅ Unsloth installed"

echo ""
echo "Step 2: Verifying installation..."
python3 -c "from unsloth import FastLanguageModel; print('✅ Unsloth OK')"
python3 -c "from trl import GRPOConfig, GRPOTrainer; print('✅ TRL OK')"

echo ""
echo "Step 3: Ready to run GRPO training!"
echo ""
echo "Run:"
echo "  python3 /root/unsloth_grpo_prompt_injection.py"
echo ""
echo "Or use tmux to keep session alive:"
echo "  tmux new -s grpo"
echo "  python3 /root/unsloth_grpo_prompt_injection.py"
echo "  # Detach: Ctrl+B, then D"
echo "  # Reattach: tmux attach -t grpo"
echo ""

