# Quick Load Script - Copy-Paste This

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob
import os
import sys

# Suppress bitsandbytes errors
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

class SuppressBitsandbytes:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.stderr

# Find checkpoint
checkpoints = sorted(glob.glob("/qwen2.5-32b-law-finetuned/checkpoint-*"))
latest = checkpoints[-1]
print(f"Loading: {latest}")

# Load model (suppress bitsandbytes errors)
with SuppressBitsandbytes():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)

with SuppressBitsandbytes():
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

model = PeftModel.from_pretrained(base_model, latest)

# Test
prompt = "What is contract law?"
formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*60)
print("RESPONSE:")
print("="*60)
print(response)
```

