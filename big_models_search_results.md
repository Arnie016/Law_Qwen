# Big Models Search Results - Ready to Download

Hugging Face MCP search results for large models that work on your 8x MI300X setup.

## Top Large Language Models (70B+)

### 1. Qwen 2.5 72B Series

**Qwen/Qwen2.5-72B-Instruct** ⭐ **RECOMMENDED**
- **Downloads:** 221.2K | **Likes:** 868
- **Task:** text-generation
- **Size:** ~140GB
- **Link:** https://hf.co/Qwen/Qwen2.5-72B-Instruct
- **Download:**
```bash
python3 download_model.py Qwen/Qwen2.5-72B-Instruct
```

**Qwen/Qwen2.5-VL-72B-Instruct** (Multimodal - Image + Text)
- **Downloads:** 615.8K | **Likes:** 556
- **Task:** image-text-to-text
- **Size:** ~140GB
- **Link:** https://hf.co/Qwen/Qwen2.5-VL-72B-Instruct
- **Special:** Can process images + text

**Qwen/Qwen2.5-72B-Instruct-AWQ** (Quantized, smaller)
- **Downloads:** 269.9K | **Likes:** 72
- **Size:** ~40GB (quantized)
- **Link:** https://hf.co/Qwen/Qwen2.5-72B-Instruct-AWQ

---

### 2. Llama 3.3 70B Series

**meta-llama/Llama-3.3-70B-Instruct** ⭐ **RECOMMENDED**
- **Downloads:** 755.3K | **Likes:** 2,551
- **Task:** text-generation
- **Size:** ~140GB
- **Link:** https://hf.co/meta-llama/Llama-3.3-70B-Instruct
- **Note:** Requires Hugging Face token (get from https://huggingface.co/settings/tokens)
- **Download:**
```bash
# First: huggingface-cli login (enter your token)
python3 download_model.py meta-llama/Llama-3.3-70B-Instruct
```

**meta-llama/Llama-3.1-70B-Instruct**
- **Downloads:** 723.2K | **Likes:** 856
- **Size:** ~140GB
- **Link:** https://hf.co/meta-llama/Llama-3.1-70B-Instruct

**meta-llama/Meta-Llama-3-70B-Instruct**
- **Downloads:** 477.7K | **Likes:** 1,496
- **Size:** ~140GB
- **Link:** https://hf.co/meta-llama/Meta-Llama-3-70B-Instruct

---

### 3. Mixtral 8x22B (Mixture of Experts)

**mistralai/Mixtral-8x22B-Instruct-v0.1** ⭐ **RECOMMENDED**
- **Downloads:** 45.1K | **Likes:** 737
- **Task:** text-generation
- **Size:** ~90GB
- **Link:** https://hf.co/mistralai/Mixtral-8x22B-Instruct-v0.1
- **Special:** MoE model (8 experts, only 2 active per token)
- **Download:**
```bash
python3 download_model.py mistralai/Mixtral-8x22B-Instruct-v0.1
```

**mistralai/Mixtral-8x22B-v0.1** (Base model)
- **Downloads:** 2.2K | **Likes:** 231
- **Size:** ~90GB
- **Link:** https://hf.co/mistralai/Mixtral-8x22B-v0.1

---

## Quick Download Commands

### Setup (one-time)
```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

# Setup workspace
bash setup_workspace.sh

# Install dependencies
pip install transformers accelerate huggingface_hub
```

### Download Qwen 2.5 72B (No auth needed)
```bash
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-72B-Instruct"
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={i: "180GB" for i in range(8)}
)
print("✅ Model downloaded!")
EOF
```

### Download Llama 3.3 70B (Requires auth)
```bash
# First: Get token from https://huggingface.co/settings/tokens
huggingface-cli login

# Then download
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.3-70B-Instruct"
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={i: "180GB" for i in range(8)}
)
print("✅ Model downloaded!")
EOF
```

### Download Mixtral 8x22B
```bash
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={i: "180GB" for i in range(8)}
)
print("✅ Model downloaded!")
EOF
```

---

## Model Comparison

| Model | Parameters | Size | Downloads | Auth Required | Speed |
|-------|------------|------|-----------|----------------|-------|
| **Qwen 2.5 72B** | 72B | ~140GB | 221K | ❌ No | Fast |
| **Llama 3.3 70B** | 70B | ~140GB | 755K | ✅ Yes | Fast |
| **Mixtral 8x22B** | 8x22B | ~90GB | 45K | ❌ No | Very Fast (MoE) |

---

## Recommendations

1. **Start with Qwen 2.5 72B** - No auth needed, excellent quality
2. **Try Mixtral 8x22B** - Faster inference (MoE), smaller download
3. **Llama 3.3 70B** - If you can get Hugging Face token, very popular

All models will automatically use all 8 GPUs with `device_map="auto"`.


