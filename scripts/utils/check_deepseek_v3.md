# Check if DeepSeek-V3 is Installed

DeepSeek-V3 is a 671B parameter MoE (Mixture of Experts) model with ~37B active parameters.

## Quick Check

**Inside ROCm container or host:**

```bash
python3 << EOF
import os
from pathlib import Path

# Check Hugging Face cache
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

deepseek_found = False
if hf_cache.exists():
    for item in hf_cache.iterdir():
        if "deepseek" in item.name.lower() and "v3" in item.name.lower():
            print(f"✅ Found: {item.name}")
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            print(f"   Size: {size / 1e12:.2f} TB")
            deepseek_found = True

if not deepseek_found:
    print("❌ DeepSeek-V3 not found")
    
    # Try to check if it can be loaded
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            "deepseek-ai/DeepSeek-V3",
            trust_remote_code=True
        )
        print("✅ Model available on Hugging Face (not downloaded yet)")
    except Exception as e:
        print(f"❌ Error: {e}")
EOF
```

## Or Use the Check Script

```bash
python3 check_deepseek_v3.py
```

## Check Ollama Models

```bash
# If using Ollama
ollama list | grep deepseek

# Or check files
ls -lh ~/.ollama/models/ | grep deepseek
```

## Download DeepSeek-V3 (if not installed)

**Warning: This is HUGE (~671B parameters, many TBs)**

```bash
python3 << EOF
from transformers import AutoModel, AutoTokenizer
import torch

print("Downloading DeepSeek-V3 (this is VERY large, will take time)...")
print("Model: 671B parameters, ~37B active (MoE)")

model_name = "deepseek-ai/DeepSeek-V3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Uses all 8 GPUs
)

print("✅ DeepSeek-V3 downloaded!")
EOF
```

## Model Info

- **Model:** DeepSeek-V3
- **Parameters:** 671B total, ~37B active (MoE)
- **Size:** Very large (multiple TBs)
- **Architecture:** Mixture of Experts
- **Link:** https://hf.co/deepseek-ai/DeepSeek-V3

## Check Disk Space First

```bash
df -h
```

DeepSeek-V3 requires significant disk space. Check available space before downloading!

