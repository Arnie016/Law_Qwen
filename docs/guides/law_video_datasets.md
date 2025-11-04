# Best Law & Video Datasets from Hugging Face

## 🏛️ BEST LAW DATASETS

### 1. Pile of Law (Legal Documents) ⭐ **RECOMMENDED**

**Massive legal corpus:**

```bash
python3 << EOF
from datasets import load_dataset

# Pile of Law - subset
dataset = load_dataset("pile-of-law/pile-of-law", "all", streaming=True)
# Use streaming=True for large datasets

# Or legal advice subset
dataset = load_dataset("lilacai/lilac-pile-of-law-r-legaladvice")
EOF
```

**Dataset:** `pile-of-law/pile-of-law`
- **Size:** Very large (legal documents from multiple sources)
- **Link:** https://hf.co/datasets/pile-of-law/pile-of-law

---

### 2. Canadian Legal Data (280 downloads)

**Canadian legal documents:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("refugee-law-lab/canadian-legal-data")
print(f"Examples: {len(dataset['train'])}")
EOF
```

**Dataset:** `refugee-law-lab/canadian-legal-data`
- **Downloads:** 280
- **Size:** 100K-1M examples
- **Languages:** English, French
- **Link:** https://hf.co/datasets/refugee-law-lab/canadian-legal-data

---

### 3. Legal Writing Corpus (11 downloads)

**Structured legal case solutions:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("FloKarlFriedKassel/Corpus_for_legal_writing_in_civil_law")
print(f"Examples: {len(dataset['train'])}")
EOF
```

**Dataset:** `FloKarlFriedKassel/Corpus_for_legal_writing_in_civil_law`
- **Downloads:** 11
- **613 structured legal case solutions**
- **Link:** https://hf.co/datasets/FloKarlFriedKassel/Corpus_for_legal_writing_in_civil_law

---

## 🎥 BEST VIDEO DATASETS

### 1. LLaVA-Video-178K ⭐ **RECOMMENDED**

**Large video-text dataset:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("lmms-lab/LLaVA-Video-178K")
print(f"Examples: {len(dataset['train'])}")
print(f"Sample keys: {dataset['train'][0].keys()}")
EOF
```

**Dataset:** `lmms-lab/LLaVA-Video-178K`
- **Downloads:** 62.7K | **Likes:** 175
- **Size:** 178K video-text pairs
- **Use:** Video understanding, video-language models
- **Link:** https://hf.co/datasets/lmms-lab/LLaVA-Video-178K

---

### 2. Video-MME (35.9K downloads)

**Video multimodal evaluation:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("lmms-lab/Video-MME")
print(f"Examples: {len(dataset['test'])}")
EOF
```

**Dataset:** `lmms-lab/Video-MME`
- **Downloads:** 35.9K | **Likes:** 55
- **Use:** Video model evaluation
- **Link:** https://hf.co/datasets/lmms-lab/Video-MME

---

### 3. ByteDance Synthetic Videos (28.4K downloads)

**CGI synthetic videos:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("kevinzzz8866/ByteDance_Synthetic_Videos")
print(f"Examples: {len(dataset['train'])}")
EOF
```

**Dataset:** `kevinzzz8866/ByteDance_Synthetic_Videos`
- **Downloads:** 28.4K
- **Size:** 10K-100K videos
- **Use:** Video synthesis, physical fidelity
- **Link:** https://hf.co/datasets/kevinzzz8866/ByteDance_Synthetic_Videos

---

### 4. Facebook PE-Video (6.8K downloads) ⭐ **LARGE**

**1 million diverse videos:**

```bash
python3 << EOF
from datasets import load_dataset

# Large dataset - use streaming
dataset = load_dataset("facebook/PE-Video", streaming=True)
# Or download subset
dataset = load_dataset("facebook/PE-Video")
EOF
```

**Dataset:** `facebook/PE-Video`
- **Downloads:** 6.8K | **Likes:** 36
- **Size:** 1 million videos, 120K+ annotated clips
- **Use:** Large-scale video training
- **Link:** https://hf.co/datasets/facebook/PE-Video

---

### 5. VideoChat-Flash Training Data (7.5K downloads)

**Video chat training data:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("OpenGVLab/VideoChat-Flash-Training-Data")
EOF
```

**Dataset:** `OpenGVLab/VideoChat-Flash-Training-Data`
- **Downloads:** 7.5K | **Likes:** 13
- **Use:** Video-language model training
- **Link:** https://hf.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data

---

## Quick Download Commands

### Law Datasets

```bash
pip install datasets

# Pile of Law (if available)
python3 << EOF
from datasets import load_dataset
dataset = load_dataset("lilacai/lilac-pile-of-law-r-legaladvice")
print(dataset)
EOF

# Canadian Legal Data
python3 << EOF
from datasets import load_dataset
dataset = load_dataset("refugee-law-lab/canadian-legal-data")
print(f"Examples: {len(dataset['train'])}")
EOF
```

### Video Datasets

```bash
# LLaVA-Video-178K (best quality)
python3 << EOF
from datasets import load_dataset
dataset = load_dataset("lmms-lab/LLaVA-Video-178K")
print(f"Examples: {len(dataset['train'])}")
EOF

# Video-MME (evaluation)
python3 << EOF
from datasets import load_dataset
dataset = load_dataset("lmms-lab/Video-MME")
print(dataset)
EOF
```

---

## Summary

| Category | Best Dataset | Downloads | Size |
|----------|-------------|-----------|------|
| **Law** | Pile of Law | N/A | Very large |
| **Law** | Canadian Legal Data | 280 | 100K-1M |
| **Video** | **LLaVA-Video-178K** | 62.7K | 178K pairs |
| **Video** | **Facebook PE-Video** | 6.8K | 1M videos |
| **Video** | Video-MME | 35.9K | Evaluation set |

---

## Recommendations

**For Law:**
1. **Pile of Law** - Most comprehensive
2. **Canadian Legal Data** - Well-structured, bilingual

**For Video:**
1. **LLaVA-Video-178K** - Best for training video-language models
2. **Facebook PE-Video** - Largest (1M videos)
3. **Video-MME** - For evaluation/testing

**Note:** Video datasets are large. Check disk space before downloading!

