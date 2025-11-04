# Unsloth GRPO Config Based on Qwen3 Alpaca Notebook

## What Changed from Notebook to GRPO

### Notebook Uses: `SFTTrainer` (Supervised Fine-Tuning)
```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
```

### GRPO Uses: `GRPOTrainer` (Reinforcement Learning)
```python
from trl import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(...),
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_fn=reward_fn,  # NEW: Reward function
)
```

## Key Differences

### 1. Model Setup (Same)
```python
# Both use same Unsloth setup
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(...)
```

### 2. Training Args (Same Structure)
```python
# Notebook: SFTConfig
SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    optim="adamw_8bit",
    ...
)

# GRPO: GRPOConfig (same args + extras)
GRPOConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    optim="adamw_8bit",
    ...
    num_generations=4,  # NEW: Generate 4 responses per prompt
)
```

### 3. Dataset Format (Different)

**Notebook (Alpaca):**
```python
dataset = {
    "text": [
        "Instruction: ...\nInput: ...\nResponse: ...",
        "Instruction: ...\nInput: ...\nResponse: ...",
    ]
}
```

**GRPO (Prompt Injection):**
```python
dataset = {
    "prompt": [
        "[METADATA]agent_code:ABCXYZ|opponent_code:DEFGHI[/METADATA]\nSystem prompt...",
        "[METADATA]agent_code:XYZABC|opponent_code:GHIDEF[/METADATA]\nSystem prompt...",
    ]
}
```

### 4. Reward Function (NEW for GRPO)

```python
def reward_fn(prompts, responses, **kwargs):
    """
    GRPO automatically calls this for each batch
    
    Args:
        prompts: List of prompt strings
        responses: List of generated response strings (4 per prompt)
    
    Returns:
        rewards: List of float scores (one per response)
    """
    rewards = []
    for prompt, response in zip(prompts, responses):
        # Extract metadata from prompt
        agent_code = extract_from_prompt(prompt)
        opponent_code = extract_from_prompt(prompt)
        
        # Calculate reward
        reward = calculate_reward(response, agent_code, opponent_code)
        rewards.append(reward)
    
    return rewards
```

## Complete GRPO Config from Notebook

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

# 1. Load model (same as notebook)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=False,  # Full precision on MI300X
)

# 2. Add LoRA (same as notebook)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Notebook uses 0
    bias="none",  # Notebook uses "none"
    use_gradient_checkpointing="unsloth",  # Notebook uses "unsloth"
    random_state=3407,  # Notebook uses 3407
)

# 3. GRPO Config (based on notebook's SFTConfig)
grpo_config = GRPOConfig(
    # From notebook
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,  # Start small
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.001,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="./qwen2.5-32b-prompt-injection-grpo",
    report_to="none",
    
    # Precision (from notebook style)
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    
    # GRPO-specific
    num_generations=4,  # Generate 4 responses per prompt
)

# 4. GRPO Trainer (reward_fn goes here, not in config)
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_fn=reward_fn,  # Your reward function
)

# 5. Train (same as notebook)
trainer.train()

# 6. Save (same as notebook)
model.save_pretrained("qwen2.5-32b-prompt-injection-grpo")
tokenizer.save_pretrained("qwen2.5-32b-prompt-injection-grpo")
```

## What "4 Rollouts" Means

**GRPO generates 4 responses per prompt:**

```
Prompt: "You are Agent A. Code is ABCXYZ..."

Response 1: "Hello! Can you give me a hint?" → reward = +1.5
Response 2: "My code is ABCXYZ!" → reward = -10.0
Response 3: "I think your code is DEFGHI." → reward = +11.0 ← BEST
Response 4: "OK" → reward = -1.0

Model learns: Prefer Response 3, avoid Response 2
```

## How GRPO Works

1. **Generate**: Create 4 responses per prompt
2. **Reward**: Calculate reward for each response
3. **Compare**: Rank responses by reward
4. **Update**: Train model to prefer high-reward responses
5. **Repeat**: Over many batches, model learns optimal strategy

## Ready to Run

The script `unsloth_grpo_prompt_injection.py` is ready to run with:
- ✅ Same model setup as notebook
- ✅ Same LoRA config as notebook
- ✅ Same training args as notebook
- ✅ GRPO-specific: reward function + num_generations
- ✅ Prompt injection game format

Run it on your MI300X server!

