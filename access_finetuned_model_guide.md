# How to Access Your Fine-Tuned Model

## Quick Commands

### 1. Check Available Checkpoints

```bash
# List all checkpoints
ls -d /qwen2.5-32b-law-finetuned/checkpoint-* 2>/dev/null | sort -V

# Latest checkpoint
ls -d /qwen2.5-32b-law-finetuned/checkpoint-* 2>/dev/null | sort -V | tail -1

# Count checkpoints
ls -d /qwen2.5-32b-law-finetuned/checkpoint-* 2>/dev/null | wc -l
```

### 2. Load and Use Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    trust_remote_code=True
)

# Load LoRA adapter (from checkpoint)
model = PeftModel.from_pretrained(
    base_model,
    "/qwen2.5-32b-law-finetuned/checkpoint-350"  # Use latest checkpoint
)

# Use the model
prompt = "What is contract law?"
formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 3. Simple Usage Script

```python
#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    trust_remote_code=True
)

# Load latest checkpoint
import glob
checkpoints = sorted(glob.glob("/qwen2.5-32b-law-finetuned/checkpoint-*"))
latest = checkpoints[-1] if checkpoints else None

if latest:
    print(f"Loading adapter from: {latest}")
    model = PeftModel.from_pretrained(base_model, latest)
else:
    print("No checkpoint found, using base model")
    model = base_model

# Test
prompt = "What is contract law?"
formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Where Are Your Checkpoints?

**Location:** `/qwen2.5-32b-law-finetuned/checkpoint-*`

**Structure:**
```
/qwen2.5-32b-law-finetuned/
├── checkpoint-50/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_state.json
├── checkpoint-100/
│   └── ...
├── checkpoint-150/
│   └── ...
└── checkpoint-200/
    └── ...
```

## What Each Checkpoint Contains

- **adapter_model.bin**: LoRA weights (~100MB)
- **adapter_config.json**: LoRA configuration
- **training_state.json**: Training progress (loss, step, etc.)

## Loading Different Checkpoints

```python
# Load specific checkpoint
model = PeftModel.from_pretrained(
    base_model,
    "/qwen2.5-32b-law-finetuned/checkpoint-350"  # Specific checkpoint
)

# Load latest checkpoint
import glob
checkpoints = sorted(glob.glob("/qwen2.5-32b-law-finetuned/checkpoint-*"))
latest = checkpoints[-1]
model = PeftModel.from_pretrained(base_model, latest)
```

## After Training Completes

If training finishes and saves final model:

```python
# Load final model (after training completes)
model = PeftModel.from_pretrained(
    base_model,
    "/qwen2.5-32b-law-finetuned"  # Final saved location
)
```

## Quick Test Commands

```bash
# Check checkpoint status
ls -lh /qwen2.5-32b-law-finetuned/checkpoint-* | tail -5

# Run test script
python3 access_finetuned_model.py
```

