# Fix Gradient Error - Debug and Fix

The error "element 0 of tensors does not require grad" means LoRA parameters aren't trainable.

## Debug: Check What's Trainable

```bash
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# Setup LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable params: {trainable_params:,}")
print(f"Total params: {total_params:,}")
print(f"Trainable%: {100 * trainable_params / total_params:.2f}%")

# Check if any LoRA params are trainable
lora_params = [n for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
print(f"LoRA trainable params: {len(lora_params)}")

if len(lora_params) == 0:
    print("❌ ERROR: No LoRA parameters are trainable!")
    print("Manually enabling...")
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            print(f"  Enabled: {name}")
else:
    print("✅ LoRA parameters are trainable")
EOF
```

## Fixed Script (Proper Gradient Setup)

```bash
python3 << EOF
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
print("Fine-Tuning Qwen 2.5 32B on Law Dataset")
print("=" * 60)

# Model
model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# FREEZE ALL PARAMETERS FIRST
for param in model.parameters():
    param.requires_grad = False

print("✅ Model loaded (all params frozen)")

# Dataset
dataset = load_dataset("lamblamb/pile_of_law_subset")
print(f"✅ Dataset: {len(dataset['train'])} examples")

# Format dataset
def format_instruction(examples):
    texts = []
    for text in examples['text']:
        instruction = f"<|im_start|>user\nAnalyze this legal text:\n{text[:500]}<|im_end|>\n<|im_start|>assistant\nThis legal text discusses legal matters...<|im_end|>"
        texts.append(instruction)
    return {"text": texts}

formatted_dataset = dataset.map(format_instruction, batched=True)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

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

model = get_peft_model(model, lora_config)

# CRITICAL: Explicitly enable gradients for LoRA
model.train()
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
        print(f"  Enabled: {name}")
    else:
        param.requires_grad = False

# Verify trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Training args
training_args = TrainingArguments(
    output_dir="./qwen2.5-32b-law-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    warmup_steps=50,
    max_steps=500,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=False,  # Try disabling this
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

print("\n5. Starting training...")
trainer.train()

print("\n6. Saving model...")
model.save_pretrained("./qwen2.5-32b-law-finetuned")
print("✅ Fine-tuning complete!")
EOF
```

## Key Changes

1. **Freeze all params first** - `param.requires_grad = False` for all
2. **Apply LoRA** - Then add LoRA adapters
3. **Enable LoRA gradients** - Explicitly set `requires_grad=True` for LoRA params
4. **Disable gradient checkpointing** - Try `gradient_checkpointing=False` (might help)
5. **Verify trainable params** - Check that LoRA params are actually trainable

Run the debug script first to see if LoRA params are trainable. If not, the fixed script explicitly enables them.

