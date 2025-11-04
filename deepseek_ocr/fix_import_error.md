# Fix DeepSeek-OCR Import Error

The error `cannot import name 'LlamaFlashAttention2'` means transformers version is too old or incompatible.

## Solution 1: Update Transformers (Recommended)

```bash
# Upgrade transformers to latest version
pip install --upgrade transformers

# Also install flash-attention if needed
pip install flash-attn --no-build-isolation
```

## Solution 2: Install Specific Version

```bash
# Try transformers 4.40.0 or newer
pip install transformers>=4.40.0

# Or latest
pip install transformers>=4.45.0
```

## Solution 3: Check Current Version

```bash
python3 -c "import transformers; print(transformers.__version__)"
```

If it's < 4.40.0, upgrade it.

## Solution 4: Install Flash Attention (If Still Fails)

```bash
# Flash attention for ROCm (may need specific build)
pip install flash-attn --no-build-isolation

# Or try without flash attention
# The model might work without it, just slower
```

## Complete Fix Command

```bash
# Inside ROCm container
pip install --upgrade transformers>=4.40.0
pip install flash-attn --no-build-isolation || echo "Flash-attn install failed, continuing without it"
```

## Then Retry

```bash
python3 << EOF
from transformers import AutoModel, AutoProcessor
import torch

model_name = "deepseek-ai/DeepSeek-OCR"
print(f"Downloading {model_name}...")

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("✅ Model downloaded!")
EOF
```


