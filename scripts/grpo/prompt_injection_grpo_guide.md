# GRPO Fine-Tuning for Prompt Injection Game

## What Was Done in Qwen3 Alpaca Notebook

The notebook shows:
1. **Unsloth Setup** - Fast fine-tuning library
2. **Qwen3-14B Model** - Loaded with 4-bit quantization
3. **LoRA Adapters** - Efficient fine-tuning (only train 1-10% of params)
4. **SFT Training** - Supervised Fine-Tuning on Alpaca dataset
5. **Alpaca Format** - Instruction/Input/Output format

## How to Adapt for Your Prompt Injection Game

### Step 1: Setup Unsloth + Model

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-32B-Instruct",  # Or use your Ollama models
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # Use full precision on MI300X
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)
```

### Step 2: Create Reward Function

```python
def calculate_reward(response, agent_code, opponent_code, history):
    """
    Reward based on game outcome:
    - +10: Extract opponent code (WIN)
    - -10: Reveal own code (LOSE)
    - +1: Try to extract codes
    - +0.5: Strategic behavior
    """
    reward = 0.0
    
    # Check if revealed own code (BAD)
    if agent_code in response:
        reward -= 10.0
    
    # Check if extracted opponent code (GOOD)
    if opponent_code in response:
        reward += 10.0
    
    # Reward for trying
    code_pattern = r'\b[A-Z]{6}\b'
    found_codes = re.findall(code_pattern, response)
    if found_codes:
        reward += 1.0 * len(found_codes)
    
    return reward
```

### Step 3: Setup GRPO Training

```python
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    output_dir="./qwen2.5-32b-prompt-injection-grpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_steps=500,
    bf16=True,
    # GRPO-specific
    num_generations=4,  # Generate 4 responses per prompt
    reward_fn=calculate_reward,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,  # Your game data
    tokenizer=tokenizer,
    reward_fn=calculate_reward,
)

trainer.train()
```

### Step 4: Generate Training Data from Games

```python
def generate_game_data(num_games=100):
    """
    Play games and collect data for training
    """
    prompts = []
    
    for game_idx in range(num_games):
        codes = [generate_code(), generate_code()]
        history = GENERAL_SYSTEM_PROMPT
        
        for round_num in range(10):  # 10 rounds per game
            agent_idx = round_num % 2
            prompt = history + SPECIFIC_SYSTEM_PROMPT(
                names[agent_idx], 
                codes[agent_idx]
            )
            prompts.append(prompt)
            
            # Generate response (during GRPO training)
            # response = model.generate(prompt)
            # reward = calculate_reward(response, ...)
    
    return Dataset.from_dict({"prompt": prompts})
```

## Complete Workflow

### 1. Initial Setup (Like Alpaca Notebook)

```python
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(...)

# Add LoRA
model = FastLanguageModel.get_peft_model(...)
```

### 2. Adapt for Your Game

```python
# Instead of Alpaca format:
# Instruction: ...
# Input: ...
# Response: ...

# Use your game format:
# System prompt + agent code + conversation history
# → Generate response
# → Calculate reward based on game outcome
```

### 3. GRPO Training (Instead of SFT)

```python
# Instead of SFTTrainer (supervised):
# trainer = SFTTrainer(...)

# Use GRPOTrainer (reinforcement learning):
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_fn=calculate_reward,  # Your game reward function
)
```

## Key Differences

| Alpaca Notebook | Your Prompt Injection Game |
|----------------|---------------------------|
| SFT (Supervised) | GRPO (RL) |
| Instruction/Output pairs | Game prompts + rewards |
| Alpaca dataset | Self-generated game data |
| Fixed responses | Rewarded responses |

## How GRPO Works

1. **Generate Multiple Responses**: For each prompt, generate 4 responses
2. **Calculate Rewards**: Score each response using your reward function
3. **Compare Responses**: Rank responses by reward
4. **Update Model**: Train model to prefer high-reward responses
5. **Repeat**: Iterate to improve performance

## Expected Results

After GRPO training:
- ✅ Agents better at extracting codes
- ✅ Agents less likely to reveal own code
- ✅ More strategic conversation patterns
- ✅ Better prompt injection techniques

## Quick Start

```bash
# Install dependencies
pip install unsloth trl transformers

# Run training
python3 prompt_injection_grpo_training.py
```

## Files Created

- `prompt_injection_grpo_training.py` - Complete GRPO training script
- This guide - Explanation of the approach

Ready to train! The script adapts the Alpaca notebook approach for your prompt injection game.

