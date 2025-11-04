# Popular Hugging Face Datasets You Can Download

All datasets download automatically when you use them. Here are the most popular ones:

## Text/Language Datasets

### 1. WikiText (902K downloads) ⭐ **RECOMMENDED**

**Great for language modeling:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(f"Training examples: {len(dataset['train'])}")
print(f"Sample: {dataset['train'][0]['text'][:200]}")
EOF
```

**Dataset:** `Salesforce/wikitext`
- **Downloads:** 902K
- **Size:** ~30MB
- **Link:** https://hf.co/datasets/Salesforce/wikitext

---

### 2. C4 (621K downloads) ⭐ **HUGE**

**Colossal Cleaned Common Crawl - massive text dataset:**

```bash
python3 << EOF
from datasets import load_dataset

# Download a small subset first
dataset = load_dataset("allenai/c4", "en", streaming=True)
# Use streaming=True for large datasets
for sample in dataset['train'].take(5):
    print(sample['text'][:200])
EOF
```

**Dataset:** `allenai/c4`
- **Downloads:** 621K
- **Size:** ~750GB (full dataset)
- **Link:** https://hf.co/datasets/allenai/c4

---

### 3. GSM8K (435K downloads) - Math Problems

**Math word problems:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("openai/gsm8k")
print(f"Examples: {len(dataset['train'])}")
print(f"Sample: {dataset['train'][0]}")
EOF
```

**Dataset:** `openai/gsm8k`
- **Downloads:** 435K
- **Size:** ~10MB
- **Link:** https://hf.co/datasets/openai/gsm8k

---

## Code/Programming Datasets

### 4. SWE-bench (900K downloads)

**Software engineering tasks:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
print(f"Examples: {len(dataset['test'])}")
EOF
```

**Dataset:** `princeton-nlp/SWE-bench_Verified`
- **Downloads:** 900K
- **Link:** https://hf.co/datasets/princeton-nlp/SWE-bench_Verified

---

### 5. SWE-Gym (1.2M downloads)

**Python code repository dataset:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("SWE-Gym/SWE-Gym")
print(f"Examples: {len(dataset['train'])}")
EOF
```

**Dataset:** `SWE-Gym/SWE-Gym`
- **Downloads:** 1.2M
- **Link:** https://hf.co/datasets/SWE-Gym/SWE-Gym

---

## Image Datasets

### 6. HuggingFace Documentation Images (2M downloads)

**Images from HF docs:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("huggingface/documentation-images")
print(f"Images: {len(dataset['train'])}")
EOF
```

**Dataset:** `huggingface/documentation-images`
- **Downloads:** 2M
- **Link:** https://hf.co/datasets/huggingface/documentation-images

---

### 7. Objaverse (595K downloads) - 3D Objects

**Massive 3D object dataset:**

```bash
python3 << EOF
from datasets import load_dataset

# This is huge (800K+ 3D objects)
dataset = load_dataset("allenai/objaverse")
EOF
```

**Dataset:** `allenai/objaverse`
- **Downloads:** 595K
- **Size:** Very large
- **Link:** https://hf.co/datasets/allenai/objaverse

---

## Multimodal (Image + Text)

### 8. Leopard-Instruct (470K downloads)

**Large instruction-tuning dataset:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("wyu1/Leopard-Instruct")
print(f"Examples: {len(dataset['train'])}")
EOF
```

**Dataset:** `wyu1/Leopard-Instruct`
- **Downloads:** 470K
- **925K instances** with images + text
- **Link:** https://hf.co/datasets/wyu1/Leopard-Instruct

---

## Robotics Datasets

### 9. NVIDIA GR00T Robotics (1.9M downloads)

**Robotics simulation data:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim")
EOF
```

**Dataset:** `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`
- **Downloads:** 1.9M
- **Link:** https://hf.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

---

## Time Series Datasets

### 10. GIFT-Eval Pretrain (1.1M downloads)

**Time series forecasting:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("Salesforce/GiftEvalPretrain")
EOF
```

**Dataset:** `Salesforce/GiftEvalPretrain`
- **Downloads:** 1.1M
- **4.5 million time series**
- **Link:** https://hf.co/datasets/Salesforce/GiftEvalPretrain

---

## Quick Setup

**Install datasets library:**

```bash
pip install datasets
```

**Download any dataset:**

```bash
python3 << EOF
from datasets import load_dataset

# Replace "dataset_name" with any dataset above
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Check what's in it
print(dataset)
print(f"Examples: {len(dataset['train'])}")
print(f"Sample: {dataset['train'][0]}")
EOF
```

---

## Dataset Categories

| Category | Examples | Use Case |
|----------|----------|----------|
| **Text** | WikiText, C4, GSM8K | Language modeling, fine-tuning |
| **Code** | SWE-bench, SWE-Gym | Code generation, debugging |
| **Images** | Documentation-images, Objaverse | Image models, 3D |
| **Multimodal** | Leopard-Instruct | Vision-language models |
| **Robotics** | GR00T | Robot training |
| **Time Series** | GIFT-Eval | Forecasting |

---

## Notes

- Datasets download automatically on first use
- Large datasets (like C4) use `streaming=True` to avoid downloading everything
- Cache location: `~/.cache/huggingface/datasets/`
- Check disk space before downloading large datasets

**Start with:** WikiText or GSM8K (small, fast to download)

