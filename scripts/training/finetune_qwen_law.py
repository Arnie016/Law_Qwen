#!/usr/bin/env python3
"""
Fine-tune Qwen 2.5 32B on law dataset
Quick start script
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

def main():
    print("=" * 60)
    print("Fine-Tuning Qwen 2.5 32B on Law Dataset")
    print("=" * 60)
    
    # Model
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    
    print(f"\n1. Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ Model loaded")
    
    # Dataset
    print("\n2. Loading law dataset...")
    dataset = load_dataset("lamblamb/pile_of_law_subset")
    print(f"✅ Dataset loaded: {len(dataset['train'])} examples")
    
    # Format dataset
    print("\n3. Formatting dataset...")
    def format_instruction(examples):
        texts = []
        for text in examples['text']:
            # Create instruction format
            instruction = f"<|im_start|>user\nAnalyze this legal text:\n{text[:500]}<|im_end|>\n<|im_start|>assistant\nThis legal text discusses...<|im_end|>"
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
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print("✅ LoRA configured")
    
    # Training args
    training_args = TrainingArguments(
        output_dir="./qwen2.5-32b-law-finetuned",
        num_train_epochs=1,  # Start with 1 epoch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        warmup_steps=50,
        max_steps=500,  # Quick test
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    print("\n5. Starting training...")
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
    print(f"Model saved to: ./qwen2.5-32b-law-finetuned")

if __name__ == "__main__":
    main()

