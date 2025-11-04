# Real Model Downloads for AMD MI300X

Models that actually work with ROCm + real parameters.

## 1. Stable Diffusion 2.1 (Text-to-Image) - **RECOMMENDED**

**Status**: ✅ Works out of the box with PyTorch/ROCm

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

# Install
pip install diffusers transformers accelerate torch torchvision

# Download & run
python3 << EOF
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

image = pipe("a cat walking on the street").images[0]
image.save("output.png")
print("Done! Saved to output.png")
EOF
```

**Model size**: ~5GB  
**Parameters**: 890M  
**Works**: ✅ Yes

---

## 2. Qwen 2.5 3B (Language Model) - **RECOMMENDED**

**Status**: ✅ Works with transformers

```bash
pip install transformers accelerate torch

python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
EOF
```

**Model size**: ~6GB  
**Parameters**: 3B  
**Works**: ✅ Yes

---

## 3. Phi-2 (Microsoft) - Small, Fast

**Status**: ✅ Works, lightweight

```bash
pip install transformers accelerate torch

python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
EOF
```

**Model size**: ~5GB  
**Parameters**: 2.7B  
**Works**: ✅ Yes

---

## 4. LayoutLMv3 (Document Understanding)

**Status**: ✅ Works for document analysis

```bash
pip install transformers torch pillow

python3 << EOF
from transformers import AutoModelForTokenClassification, AutoProcessor
from PIL import Image
import torch

model_name = "microsoft/layoutlmv3-base"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
model = model.to("cuda")

# Example usage
image = Image.open("document.png").convert("RGB")
words = ["Hello", "World"]
boxes = [[0, 0, 100, 50], [100, 0, 200, 50]]

encoding = processor(image, words, boxes=boxes, return_tensors="pt")
encoding = {k: v.to("cuda") for k, v in encoding.items()}

with torch.no_grad():
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1)

print(f"Predictions: {predictions}")
EOF
```

**Model size**: ~330MB  
**Parameters**: 133M  
**Works**: ✅ Yes

---

## 5. Tesseract OCR (Not a download, but works)

**Status**: ✅ System package

```bash
# Install Tesseract
apt-get update && apt-get install -y tesseract-ocr

# Use via Python
pip install pytesseract pillow

python3 << EOF
import pytesseract
from PIL import Image

image = Image.open("document.png")
text = pytesseract.image_to_string(image)
print(text)
EOF
```

**Works**: ✅ Yes, no download needed

---

## 6. AnimateDiff (Text-to-Video) - **EXPERIMENTAL**

**Status**: ⚠️ May need adjustments

```bash
pip install diffusers transformers accelerate torch

python3 << EOF
from diffusers import AnimateDiffPipeline, MotionAdapter
import torch

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained(
    "frankjoshua/toonyou_beta6",
    motion_adapter=adapter,
    torch_dtype=torch.bfloat16
).to("cuda")

video = pipe("a cat walking", num_frames=16).frames
# Save video frames
EOF
```

**Model size**: ~2-3GB  
**Works**: ⚠️ May need ROCm compatibility testing

---

## Quick Test: Verify GPU Works

```bash
# Test GPU access
python3 << EOF
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
```

---

## Recommended Starting Order

1. **Stable Diffusion 2.1** - Easiest, most reliable
2. **Qwen 2.5 3B** - Good language model, fast
3. **LayoutLMv3** - If you need document analysis
4. **Phi-2** - If you want something smaller/faster

---

## Notes

- All models use `torch_dtype=torch.bfloat16` for efficiency on MI300X
- Models download automatically on first use (via Hugging Face)
- Total download size: ~15-20GB for all models
- MI300X has 192GB HBM, so plenty of space

