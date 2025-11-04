# Unsloth RL Fine-Tuning Guide (RLHF/DPO)

## What is RL Fine-Tuning?

**RLHF (Reinforcement Learning from Human Feedback):**
- Train model to prefer better responses
- Uses human preferences (good vs bad answers)
- Makes model more aligned with human values

**DPO (Direct Preference Optimization):**
- Simpler alternative to RLHF
- Directly optimizes for preferred responses
- No need for separate reward model

## Unsloth Benefits

✅ **3-5x faster** than standard fine-tuning
✅ **Uses less memory** (optimized for efficiency)
✅ **Easy to use** (simple API)
✅ **Works with RLHF/DPO** out of the box

## Installation

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

## DPO Fine-Tuning (Recommended - Easier)

### Step 1: Prepare Preference Dataset

```python
from datasets import load_dataset

# Load preference dataset (chosen vs rejected responses)
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

# Format: prompt, chosen, rejected
# Example:
# {
#   "prompt": "What is machine learning?",
#   "chosen": "Machine learning is... (good answer)",
#   "rejected": "Machine learning is... (bad answer)"
# }
```

### Step 2: Fine-Tune with Unsloth DPO

```python
from unsloth import FastLanguageModel
import torch

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # Use full precision on MI300X
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Prepare dataset
from unsloth import is_bfloat16_supported
from trl import DPOTrainer
from transformers import TrainingArguments

# Format dataset for DPO
def format_dpo_dataset(examples):
    return {
        "prompt": examples["prompt"],
        "chosen": examples["chosen"],
        "rejected": examples["rejected"],
    }

formatted_dataset = dataset.map(format_dpo_dataset)

# DPO Training arguments
dpo_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="./qwen2.5-32b-dpo-finetuned",
)

# DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Can use original model as reference
    args=dpo_args,
    beta=0.1,  # DPO temperature parameter
    train_dataset=formatted_dataset["train"],
    tokenizer=tokenizer,
    max_length=2048,
    max_prompt_length=1024,
)

# Train
print("Starting DPO fine-tuning...")
dpo_trainer.train()

# Save
model.save_pretrained("./qwen2.5-32b-dpo-finetuned")
```

## RLHF Fine-Tuning (Full Pipeline)

### Step 1: Supervised Fine-Tuning (SFT)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    max_seq_length=2048,
    dtype=torch.bfloat16,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# SFT Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=2048,
    dataset_text_field="text",
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        output_dir="./qwen2.5-32b-sft",
    ),
)

trainer.train()
model.save_pretrained("./qwen2.5-32b-sft")
```

### Step 2: Reward Model Training

```python
# Train reward model (rates responses)
from unsloth import FastLanguageModel

# Load model for reward model
reward_model, reward_tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    max_seq_length=2048,
    dtype=torch.bfloat16,
)

# Add LoRA
reward_model = FastLanguageModel.get_peft_model(
    reward_model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Train reward model (simplified - full implementation is more complex)
# Reward model learns to score responses (higher = better)
```

### Step 3: PPO Training (RLHF)

```python
from trl import PPOTrainer, PPOConfig

# PPO Configuration
ppo_config = PPOConfig(
    model_name="./qwen2.5-32b-sft",
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
)

# PPO Trainer (requires reward model)
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
)

# PPO Training loop (more complex, requires generation + reward)
# This is simplified - full PPO needs generation loop
```

## Complete DPO Example (Ready to Run)

```python
#!/usr/bin/env python3
"""
Unsloth DPO Fine-Tuning - Complete Example
"""
from unsloth import FastLanguageModel
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

print("=" * 60)
print("Unsloth DPO Fine-Tuning")
print("=" * 60)

# Load model with Unsloth
print("\n1. Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Add LoRA
print("\n2. Adding LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Load preference dataset
print("\n3. Loading preference dataset...")
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train[:1000]")

# Format dataset
def format_dpo(examples):
    return {
        "prompt": examples["prompt"],
        "chosen": examples["chosen"],
        "rejected": examples["rejected"],
    }

formatted_dataset = dataset.map(format_dpo, batched=True)

# DPO Training arguments
print("\n4. Setting up DPO training...")
from unsloth import is_bfloat16_supported

dpo_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="./qwen2.5-32b-dpo-finetuned",
)

# DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    beta=0.1,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    max_length=2048,
    max_prompt_length=1024,
)

# Train
print("\n5. Starting DPO training...")
dpo_trainer.train()

# Save
print("\n6. Saving model...")
model.save_pretrained("./qwen2.5-32b-dpo-finetuned")
print("✅ DPO fine-tuning complete!")

# Test
print("\n7. Testing model...")
FastLanguageModel.for_inference(model)
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Key Differences: DPO vs RLHF

### DPO (Easier)
- ✅ No reward model needed
- ✅ Direct optimization
- ✅ Simpler to implement
- ✅ Faster training

### RLHF (Full Pipeline)
- ⚠️ Requires reward model
- ⚠️ More complex (PPO)
- ⚠️ More compute needed
- ✅ More control

## Quick Start Command

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate

# Run DPO training
python3 unsloth_dpo_finetune.py
```

## Expected Benefits

After DPO/RLHF fine-tuning:
- ✅ Better response quality
- ✅ More helpful responses
- ✅ Better alignment with preferences
- ✅ Reduced harmful outputs

## Resources

- **Unsloth Docs:** https://unsloth.ai
- **DPO Paper:** Direct Preference Optimization
- **RLHF Paper:** Training Language Models with Human Feedback

Ready to start? Use DPO first (easier), then try full RLHF if needed!

