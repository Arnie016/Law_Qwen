# GRPO Training Guide - Complete Instructions

## Why 32B Instead of Smaller Models?

### ✅ Advantages of 32B:
- **Better reasoning** for complex legal concepts
- **More knowledge retention** during fine-tuning
- **Better fine-tuning results** (larger models improve more)
- **Your hardware supports it:** 8x MI300X with 1.5TB HBM → 32B fits comfortably

### ⚖️ Trade-offs:
- **Slower training/inference** (but acceptable with 8 GPUs)
- **More memory usage** (but you have plenty)
- **Higher cost per training step** (but better end result)

### 📊 Model Size Comparison:

| Model | Parameters | Memory (BF16) | Training Speed | Quality |
|-------|-----------|---------------|----------------|---------|
| **7B** | 7B | ~14GB | Fastest | Good |
| **14B** | 14B | ~28GB | Fast | Better |
| **32B** | 32B | ~64GB | Medium | **Best** ← You're using |
| **72B** | 72B | ~144GB | Slow | Excellent |

### 💡 Recommendation:
**Stick with 32B for legal fine-tuning.** Your hardware supports it, and you'll get better results. If you want to test GRPO faster, use 14B for initial experiments, then scale to 32B.

---

## How to Run GRPO Training

### Step 1: Install Unsloth

```bash
# Enter ROCm container
docker exec -it rocm /bin/bash

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate

# Verify installation
python3 -c "from unsloth import FastLanguageModel; print('✅ Unsloth OK')"
python3 -c "from trl import GRPOConfig, GRPOTrainer; print('✅ TRL OK')"
```

### Step 2: Run GRPO Training

**Option A: Run directly (recommended for testing)**
```bash
python3 /root/unsloth_grpo_prompt_injection.py
```

**Option B: Use tmux (recommended for long training)**
```bash
# Start tmux session
tmux new -s grpo

# Run training
python3 /root/unsloth_grpo_prompt_injection.py

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t grpo
```

### Step 3: What Happens During Training

1. **Model Loading** (~5-10 min for 32B)
   - Loads Qwen 2.5 32B
   - Sets up LoRA adapters
   - Configures for GRPO training

2. **Dataset Generation** (~1 min)
   - Generates 100 game prompts
   - Each prompt has metadata (agent_code, opponent_code)

3. **GRPO Training** (~30-60 min depending on steps)
   - For each prompt:
     * Generates 4 different responses (rollouts)
     * Calculates reward for each response
     * Updates model to prefer high-reward responses
   - Logs progress every 10 steps
   - Saves checkpoints every 50 steps

4. **Model Saving** (~2 min)
   - Saves LoRA adapters to `./qwen2.5-32b-prompt-injection-grpo`
   - Model ready to use!

### Step 4: Monitor Training

**Check GPU usage:**
```bash
# In another terminal
watch -n 1 rocm-smi
```

**Check training logs:**
```bash
# If using tmux
tmux attach -t grpo

# Or check output file if redirected
tail -f grpo_training.log
```

### Step 5: Adjust Training Parameters

Edit `/root/unsloth_grpo_prompt_injection.py`:

```python
# Change number of training steps
max_steps=60,  # Default: 60, increase for more training

# Change number of games
num_games=100,  # Default: 100, increase for more data

# Change model size (for testing)
model_name="Qwen/Qwen2.5-14B-Instruct",  # Smaller, faster

# Change number of rollouts per prompt
num_generations=4,  # Default: 4, increase for more exploration
```

---

## GRPO Training Explained

### What is GRPO?

**Group Relative Policy Optimization** - A reinforcement learning method that:
1. Generates multiple responses per prompt (4 by default)
2. Calculates reward for each response
3. Updates model to prefer high-reward responses

### How It Works:

```
For each prompt:
├─ Generate 4 responses (rollouts)
│   ├─ Response 1: "Hello! Can you give me a hint?" → reward = +1.5
│   ├─ Response 2: "My code is ABCXYZ!" → reward = -10.0 ❌
│   ├─ Response 3: "I think your code is DEFGHI." → reward = +11.0 ✅ BEST
│   └─ Response 4: "OK" → reward = -1.0
│
├─ Rank responses by reward
│   └─ Response 3 > Response 1 > Response 4 > Response 2
│
└─ Update model
    └─ Learn to prefer Response 3, avoid Response 2
```

### Reward Function:

```python
def calculate_reward(response, agent_code, opponent_code, history, turn_number):
    reward = 0.0
    
    # WIN: Extract opponent code
    if opponent_code in response:
        reward += 10.0
    
    # LOSE: Reveal own code
    if agent_code in response:
        reward -= 10.0
    
    # Reward for trying
    if code_pattern in response:
        reward += 1.0
    
    # Penalize short responses
    if len(response) < 20:
        reward -= 1.0
    
    return reward
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'unsloth'"
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Error: "ModuleNotFoundError: No module named 'trl'"
```bash
pip install trl
```

### Training too slow?
- Use smaller model: `Qwen/Qwen2.5-14B-Instruct`
- Reduce `num_generations` from 4 to 2
- Reduce `max_steps` from 60 to 30

### Out of memory?
- Use 4-bit quantization: `load_in_4bit=True`
- Reduce `per_device_train_batch_size` from 2 to 1
- Reduce `max_seq_length` from 2048 to 1024

### Training interrupted?
- Checkpoints saved every 50 steps
- Resume from checkpoint: modify script to load checkpoint
- Or restart training (will start from beginning)

---

## Expected Results

After GRPO training, the model should:
- ✅ Better at extracting opponent codes
- ✅ Better at avoiding revealing own code
- ✅ More strategic responses
- ✅ More engaging conversation

Test with:
```python
# Load trained model
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("qwen2.5-32b-prompt-injection-grpo")
FastLanguageModel.for_inference(model)

# Test
prompt = "You are Agent A. Code is ABCXYZ. [game prompt]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

---

## Quick Reference

```bash
# Install
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate

# Run
tmux new -s grpo
python3 /root/unsloth_grpo_prompt_injection.py

# Monitor
tmux attach -t grpo
rocm-smi

# Test
python3 test_grpo_model.py
```

---

Ready to train? Start with Step 1 above!

