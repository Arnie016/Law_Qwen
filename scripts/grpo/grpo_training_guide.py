#!/usr/bin/env python3
"""
GRPO Training Setup Guide
How to run GRPO fine-tuning for prompt injection game
"""
import subprocess
import sys

print("=" * 80)
print("GRPO TRAINING SETUP GUIDE")
print("=" * 80)

print("""
STEP 1: Install Unsloth (if not already installed)
====================================================

Inside ROCm container:
  docker exec -it rocm /bin/bash
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps trl peft accelerate

Check if installed:
  python3 -c "from unsloth import FastLanguageModel; print('✅ Unsloth installed')"

STEP 2: Prepare GRPO Training Script
=====================================

You already have: unsloth_grpo_prompt_injection.py

This script:
- Loads Qwen 2.5 32B (or can use smaller model for testing)
- Sets up LoRA adapters
- Generates game prompts
- Configures GRPO training
- Trains with reward function

STEP 3: Run GRPO Training
==========================

Option A: Use 32B (slower but better)
  python3 unsloth_grpo_prompt_injection.py

Option B: Test with smaller model first (faster)
  Edit script: change model_name to "Qwen/Qwen2.5-7B-Instruct" or "Qwen/Qwen2.5-14B-Instruct"

STEP 4: Monitor Training
=========================

During training, you'll see:
- Model loading progress
- LoRA adapter setup
- Game prompt generation
- GRPO training steps:
  * For each prompt: generates 4 responses
  * Calculates reward for each response
  * Updates model based on rewards
- Training metrics (loss, rewards, etc.)

STEP 5: Check Results
=====================

After training:
- Model saved to: ./qwen2.5-32b-prompt-injection-grpo
- Checkpoints saved periodically
- Can resume from checkpoint if interrupted

STEP 6: Test Trained Model
===========================

Load and test:
  python3 test_grpo_model.py

Or use in your game code to see if it learned better strategies.
""")

print("\n" + "=" * 80)
print("QUICK START COMMANDS")
print("=" * 80)
print("""
# 1. Enter container
docker exec -it rocm /bin/bash

# 2. Check if unsloth installed
python3 -c "import unsloth; print('OK')" || pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. Run GRPO training (use tmux to keep session alive)
tmux new -s grpo
python3 /root/unsloth_grpo_prompt_injection.py

# 4. Detach from tmux: Ctrl+B, then D
# 5. Reattach later: tmux attach -t grpo

# 6. Monitor GPU usage:
watch -n 1 nvidia-smi  # or rocm-smi for AMD
""")

