# Fine-Tune Qwen 2.5 32B on Law Dataset

Complete guide for fine-tuning Qwen 2.5 32B on legal datasets using your 8x MI300X GPUs.

## Your Setup

- **Model:** Qwen 2.5 32B (19GB, already installed via Ollama)
- **Dataset:** `pile-of-law/pile-of-law` (legal corpus)
- **GPUs:** 8x MI300X (1.5TB GPU memory)
- **Disk:** 1.5TB available

---

## Method 1: Fine-Tune with Transformers (Recommended)

**Best for: Full control, custom training loops**

### Step 1: Install Dependencies

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

pip install transformers datasets accelerate peft bitsandbytes torch
```

### Step 2: Download Law Dataset

```bash
python3 << EOF
from datasets import load_dataset

print("Downloading Pile of Law dataset...")
# Use streaming for large dataset
dataset = load_dataset("pile-of-law/pile-of-law", "all", streaming=True)

# For fine-tuning, you'll want a subset
# Let's get a sample
dataset_subset = load_dataset("lamblamb/pile_of_law_subset")
print(f"Subset examples: {len(dataset_subset['train'])}")
EOF
```

### Step 3: Prepare Dataset

```python
# prepare_law_dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("lamblamb/pile_of_law_subset")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

# Prepare for fine-tuning
def format_for_training(examples):
    # Format as instruction-following
    texts = []
    for text in examples['text']:
        # Create instruction format
        formatted = f"<|im_start|>user\nAnalyze this legal document:\n{text[:1000]}<|im_end|>\n<|im_start|>assistant\n[Legal analysis here]<|im_end|>"
        texts.append(formatted)
    
    return {"text": texts}

# Format dataset
formatted_dataset = dataset.map(format_for_training, batched=True)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
tokenized_dataset.save_to_disk("./law_dataset_tokenized")
```

### Step 4: Fine-Tune with LoRA (Efficient)

```python
# fine_tune_qwen_law.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch

# Load model
model_name = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Uses all 8 GPUs
    trust_remote_code=True
)

# Setup LoRA for efficient fine-tuning
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

# Load dataset
dataset = load_from_disk("./law_dataset_tokenized")

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen2.5-32b-law-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,  # Use bfloat16 for MI300X
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    warmup_steps=100,
    max_steps=1000,  # Adjust based on dataset size
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

# Train
print("Starting fine-tuning...")
trainer.train()

# Save
model.save_pretrained("./qwen2.5-32b-law-finetuned")
```

### Step 5: Run Fine-Tuning

```bash
python3 fine_tune_qwen_law.py
```

---

## Method 2: Convert Ollama Model to Transformers

**If you want to fine-tune the Ollama model:**

```bash
# Ollama models are in GGUF format
# Convert to transformers format for fine-tuning
# (This is complex - Method 1 is easier)
```

---

## Method 3: Fine-Tune with Unsloth (Fastest)

**Optimized for speed:**

```bash
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git

python3 << EOF
from unsloth import FastLanguageModel
import torch

# Load model
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

# Load dataset
from datasets import load_dataset
dataset = load_dataset("lamblamb/pile_of_law_subset")

# Format dataset
def format_instruction(examples):
    inputs = f"<|im_start|>user\n{examples['text'][:500]}<|im_end|>\n<|im_start|>assistant\n"
    outputs = f"[Legal analysis]<|im_end|>"
    return {"text": inputs + outputs}

dataset = dataset.map(format_instruction)

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    max_seq_length=2048,
    dataset_text_field="text",
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        output_dir="./outputs",
        optim="adamw_torch",
    ),
)

trainer.train()
EOF
```

---

## Quick Start Script

```bash
#!/bin/bash
# Quick fine-tuning script

echo "Step 1: Installing dependencies..."
pip install transformers datasets accelerate peft bitsandbytes

echo "Step 2: Downloading law dataset..."
python3 << EOF
from datasets import load_dataset
dataset = load_dataset("lamblamb/pile_of_law_subset")
print(f"Dataset ready: {len(dataset['train'])} examples")
EOF

echo "Step 3: Starting fine-tuning..."
# Run the fine-tuning script above
```

---

## Input/Output Format

**Before Fine-Tuning:**
```
Input: "What is contract law?"
Output: [Generic answer]
```

**After Fine-Tuning on Law Data:**
```
Input: "What is contract law?"
Output: "Contract law governs agreements between parties. Key elements include 
offer, acceptance, and consideration. In Smith v. Jones (2023), the court 
held that..."
```

---

## Recommended Approach

**Use Method 1 (Transformers + LoRA):**
- ✅ Full control
- ✅ Efficient (LoRA)
- ✅ Uses all 8 GPUs
- ✅ Works with your setup

**Start with:**
1. Download `lamblamb/pile_of_law_subset` (smaller, manageable)
2. Use LoRA for efficient fine-tuning
3. Fine-tune for 1-3 epochs
4. Test on legal questions

---

## Expected Training Time

- **Dataset:** 10K-100K examples
- **GPUs:** 8x MI300X
- **Time:** ~2-6 hours (depending on dataset size)

Ready to start? Run the setup commands above!

