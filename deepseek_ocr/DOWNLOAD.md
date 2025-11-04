# Download DeepSeek-OCR on AMD DevCloud

Quick guide to download and test DeepSeek-OCR on your server.

## Quick Download

**Inside ROCm container:**

```bash
# Enter container
docker exec -it rocm /bin/bash

# Navigate to deepseek_ocr directory
cd /path/to/deepseek_ocr

# Run download script
bash download_model.sh
```

Or manually:

```bash
# Install dependencies
pip install transformers accelerate pillow

# Download and test
python3 quick_test.py
```

## Manual Download

```bash
python3 << EOF
import torch
from transformers import AutoModel, AutoProcessor

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

## Model Info

- **Model:** `deepseek-ai/DeepSeek-OCR`
- **Size:** ~7GB
- **Parameters:** 3.3B
- **Downloads:** 2.1M+
- **Cache location:** `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR`

## After Download

1. **Test the model:**
   ```bash
   python3 quick_test.py
   ```

2. **Run the API:**
   ```bash
   python3 app.py
   ```

3. **Test API endpoint:**
   ```bash
   curl -X POST "http://localhost:8000/ocr" \
     -F "file=@test_image.png"
   ```

## Troubleshooting

- **Out of memory:** Model uses ~7GB, should fit on MI300X easily
- **Download slow:** Model is ~7GB, be patient
- **Import errors:** Make sure transformers is installed: `pip install transformers`


