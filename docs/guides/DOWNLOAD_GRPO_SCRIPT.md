# Quick Fix: Download GRPO Script

## Download the script

```bash
# You're in /root/scripts/training
# Download to correct location
cd /root/scripts
mkdir -p grpo
cd grpo

# Download script from GitHub
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py

# Or download to current directory
cd /root/scripts/training
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py
```

## Then run it

```bash
# Make sure venv is activated
source /root/venv/bin/activate

# Run script
python grpo_legal_training.py
```

---

## Quick One-Liner

```bash
# Download and run
cd /root/scripts/training && \
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/grpo_legal_training.py && \
source /root/venv/bin/activate && \
python grpo_legal_training.py
```

