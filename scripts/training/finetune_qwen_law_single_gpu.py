#!/usr/bin/env python3
"""
Single-GPU Fine-tune Qwen 2.5 32B on law dataset
Optimized for 1 GPU (MI300X: 192GB HBM)
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

print("=" * 60)
print("Fine-Tuning Qwen 2.5 32B on Law Dataset (Single GPU)")
print("=" * 60)

# Check GPU
print(f"\nGPU Info:")
print(f"  Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Model
model_name = "Qwen/Qwen2.5-32B-Instruct"

print(f"\n1. Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Single GPU: explicitly use GPU 0
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # Single GPU explicit
    trust_remote_code=True
)
print("✅ Model loaded on single GPU")

# Enable training
model.train()
for param in model.parameters():
    param.requires_grad = False  # Freeze base model

# Dataset
print("\n2. Loading law dataset...")
dataset = load_dataset("lamblamb/pile_of_law_subset")
print(f"✅ Dataset loaded: {len(dataset['train'])} examples")

# Format dataset
print("\n3. Formatting dataset...")
def format_instruction(examples):
    texts = []
    for text in examples['text']:
        instruction = f"<|im_start|>user\nAnalyze this legal text:\n{text[:500]}<|im_end|>\n<|im_start|>assistant\nThis legal text discusses legal matters...<|im_end|>"
        texts.append(instruction)
    return {"text": texts}

formatted_dataset = dataset.map(format_instruction, batched=True)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
print("✅ Dataset formatted")

# LoRA setup
print("\n4. Setting up LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Enable gradients for LoRA parameters
model.train()
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

print("✅ LoRA configured")

# Training args - optimized for single GPU
training_args = TrainingArguments(
    output_dir="./qwen2.5-32b-law-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Single GPU: keep at 1
    gradient_accumulation_steps=16,  # Compensate with gradient accumulation
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    warmup_steps=50,
    max_steps=10000,  # IMPORTANT: Increased from 500 to 10,000 for improvement
    save_steps=500,  # Save checkpoint every 500 steps (10 checkpoints total)
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,  # Saves memory
    # Single GPU optimizations
    dataloader_num_workers=4,  # Can increase for single GPU
    remove_unused_columns=False,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
print("\n5. Starting training...")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Max steps: {training_args.max_steps}")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# Save
print("\n6. Saving model...")
model.save_pretrained("./qwen2.5-32b-law-finetuned")
print("✅ Fine-tuning complete!")

