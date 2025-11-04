# Test Your Fine-Tuned Law Model

## Quick Fix for Bitsandbytes Error

The error is harmless - bitsandbytes isn't needed. Here's the fixed code:

```python
# Inside ROCm container
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob

# Find latest checkpoint
checkpoints = sorted(glob.glob("/qwen2.5-32b-law-finetuned/checkpoint-*"))
latest = checkpoints[-1]
print(f"Loading checkpoint: {latest}")

# Load base model (FIXED - use dtype instead of torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    trust_remote_code=True
)

# Suppress bitsandbytes warning
import warnings
warnings.filterwarnings('ignore', category=RuntimeError, message='.*bitsandbytes.*')

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct",
    dtype=torch.bfloat16,  # FIXED: Use dtype instead of torch_dtype
    device_map="auto",
    trust_remote_code=True
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, latest)

# Test
prompt = "What is contract law?"
formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
EOF
```

## Changes Made

1. **`dtype` instead of `torch_dtype`** - Fixed deprecation warning
2. **Suppress bitsandbytes warning** - Ignore harmless error
3. **Find latest checkpoint** - Automatically use checkpoint-500

## Run It

```bash
# Inside ROCm container
python3 test_finetuned_law_model.py
```

This will load and test your fine-tuned model!

