# DeepSeek-OCR Dependencies Installer

Quick fix for missing dependencies. Run this inside ROCm container.

## Install All Required Dependencies

```bash
pip install transformers accelerate pillow addict matplotlib einops easydict
```

## Or Install One by One

```bash
pip install addict
pip install matplotlib
pip install einops
pip install easydict
```

## Complete Dependency List

```bash
pip install torch transformers accelerate pillow addict matplotlib einops easydict tokenizers numpy fastapi uvicorn pydantic python-multipart
```

## After Installing

Try downloading again:

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


