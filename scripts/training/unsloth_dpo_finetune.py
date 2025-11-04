#!/usr/bin/env python3
"""
Unsloth DPO Fine-Tuning - Ready to Run
"""
from unsloth import FastLanguageModel, is_bfloat16_supported
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
    load_in_4bit=False,  # Use full precision on MI300X
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

