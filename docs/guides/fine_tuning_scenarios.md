# Fine-Tuning Scenarios: Law & Video on 8x MI300X

You have **1.5TB disk space** and **8x MI300X GPUs** - perfect for large-scale fine-tuning!

## 📊 Your Resources

- **Disk Space:** 1.5TB available
- **GPU Memory:** 1.5TB total (8x 192GB)
- **Can handle:** Large datasets and models

---

## 🏛️ LAW FINE-TUNING SCENARIOS

### 1. Legal Document Classification

**Goal:** Classify legal documents by type (contract, case law, regulation, etc.)

**Dataset:** `pile-of-law/pile-of-law`
**Model:** `pile-of-law/legalbert-large-1.7M-2` or `Qwen/Qwen2.5-14B-Instruct`

**Input:**
```
Document: "In the matter of Smith v. Jones, the court held that..."
```

**Output:**
```
{
  "classification": "case_law",
  "confidence": 0.95,
  "subtype": "appellate_decision"
}
```

**Code:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "pile-of-law/legalbert-large-1.7M-2",
    num_labels=10  # 10 document types
)

# Load dataset
dataset = load_dataset("pile-of-law/pile-of-law", "all", streaming=True)
```

---

### 2. Legal Question Answering

**Goal:** Answer legal questions from case documents

**Dataset:** `pile-of-law/pile-of-law` + custom QA pairs
**Model:** `Qwen/Qwen2.5-14B-Instruct`

**Input:**
```
Question: "What was the court's ruling in Smith v. Jones?"
Context: "In Smith v. Jones, the court held that..."
```

**Output:**
```
{
  "answer": "The court ruled in favor of Smith, holding that the contract was valid.",
  "confidence": 0.92,
  "source": "Smith v. Jones, 2023 WL 123456"
}
```

---

### 3. Contract Analysis & Extraction

**Goal:** Extract key terms from contracts (parties, dates, amounts, clauses)

**Dataset:** `pile-of-law/pile-of-law` (contract subset)
**Model:** `Qwen/Qwen2.5-14B-Instruct`

**Input:**
```
Contract: "This Agreement is entered into on January 1, 2024, between Company A and Company B..."
```

**Output:**
```json
{
  "parties": ["Company A", "Company B"],
  "date": "2024-01-01",
  "key_terms": {
    "termination_clause": "30 days notice",
    "payment_terms": "Net 30",
    "jurisdiction": "New York"
  }
}
```

---

### 4. Legal Document Summarization

**Goal:** Generate concise summaries of long legal documents

**Dataset:** `pile-of-law/pile-of-law`
**Model:** `Qwen/Qwen2.5-14B-Instruct` or `Llama-3.1-70B-Instruct`

**Input:**
```
Long legal document (5000 words): "In the matter of..."
```

**Output:**
```
Summary: "This case involves a dispute over contract interpretation. 
The court found that the contract language was ambiguous and remanded 
for further proceedings. Key holding: Ambiguous terms require extrinsic 
evidence."
```

---

### 5. Legal Citation Generation

**Goal:** Generate proper legal citations from case summaries

**Dataset:** `pile-of-law/pile-of-law`
**Model:** `Qwen/Qwen2.5-14B-Instruct`

**Input:**
```
Case summary: "Smith v. Jones, decided in 2023, involved contract dispute..."
```

**Output:**
```
Citation: "Smith v. Jones, 456 F.3d 789 (9th Cir. 2023)"
```

---

### 6. Legal Risk Assessment

**Goal:** Assess legal risk of contracts/documents

**Dataset:** `pile-of-law/pile-of-law` + labeled risk data
**Model:** `Qwen/Qwen2.5-14B-Instruct`

**Input:**
```
Contract: "This agreement..."
```

**Output:**
```json
{
  "risk_level": "medium",
  "risk_score": 0.65,
  "risks": [
    "Unclear termination clause",
    "No dispute resolution mechanism",
    "Jurisdiction may be unfavorable"
  ],
  "recommendations": [
    "Add explicit termination notice period",
    "Include arbitration clause"
  ]
}
```

---

### 7. Case Law Similarity Search

**Goal:** Find similar cases based on legal issues

**Dataset:** `pile-of-law/pile-of-law`
**Model:** Fine-tuned embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`)

**Input:**
```
Query case: "Dispute over employment contract non-compete clause"
```

**Output:**
```
Similar cases:
1. Johnson v. Corp (0.92 similarity) - Non-compete enforcement
2. Williams v. Tech Co (0.88 similarity) - Employment contract dispute
3. Brown v. Employer (0.85 similarity) - Restrictive covenant case
```

---

### 8. Legal Translation (English ↔ Legal Language)

**Goal:** Translate between plain English and formal legal language

**Dataset:** `pile-of-law/pile-of-law` + parallel translations
**Model:** `Qwen/Qwen2.5-14B-Instruct`

**Input:**
```
Plain English: "Company A agrees to pay Company B $100,000"
```

**Output:**
```
Legal Language: "Company A hereby covenants and agrees to remit payment 
in the amount of One Hundred Thousand Dollars ($100,000.00) to Company B 
pursuant to the terms herein."
```

---

### 9. Legal Document Generation

**Goal:** Generate legal documents (contracts, motions, briefs) from templates

**Dataset:** `pile-of-law/pile-of-law` + document templates
**Model:** `Qwen/Qwen2.5-14B-Instruct`

**Input:**
```
Template: "Employment Agreement"
Parameters: {
  "employer": "Tech Corp",
  "employee": "John Doe",
  "salary": "$150,000",
  "start_date": "2024-01-01"
}
```

**Output:**
```
Generated Employment Agreement:
"This Employment Agreement is entered into on January 1, 2024..."
[Full legal document generated]
```

---

### 10. Multi-Jurisdiction Legal Analysis

**Goal:** Compare legal approaches across jurisdictions

**Dataset:** `pile-of-law/pile-of-law` + `refugee-law-lab/canadian-legal-data`
**Model:** `Qwen/Qwen2.5-14B-Instruct` or `Llama-3.1-70B-Instruct`

**Input:**
```
Legal issue: "Non-compete clause enforcement"
Jurisdictions: ["US", "Canada"]
```

**Output:**
```json
{
  "us_approach": "Courts generally enforce if reasonable in scope and duration",
  "canadian_approach": "More restrictive, requires clear consideration",
  "differences": [
    "US: Presumed valid if reasonable",
    "Canada: Requires explicit benefit to employee"
  ],
  "recommendation": "Consider Canadian standards for international contracts"
}
```

---

## 🎥 VIDEO FINE-TUNING SCENARIOS

### 1. Video Captioning (Video → Text)

**Goal:** Generate captions describing video content

**Dataset:** `lmms-lab/LLaVA-Video-178K`
**Model:** `Qwen/Qwen2.5-VL-72B-Instruct` or fine-tune `Qwen/Qwen2.5-14B-Instruct`

**Input:**
```
Video frames: [frame1.jpg, frame2.jpg, ..., frame25.jpg]
```

**Output:**
```
Caption: "A person is walking through a busy city street, 
carrying a shopping bag, while cars pass by in the background."
```

---

### 2. Video Question Answering

**Goal:** Answer questions about video content

**Dataset:** `lmms-lab/LLaVA-Video-178K` or `lmms-lab/Video-MME`
**Model:** `Qwen/Qwen2.5-VL-72B-Instruct`

**Input:**
```
Video: [video frames]
Question: "What is the person doing in the video?"
```

**Output:**
```
Answer: "The person is cooking pasta in a kitchen. They are stirring 
a pot on the stove and adding ingredients."
```

---

### 3. Video Action Classification

**Goal:** Classify actions in videos (walking, cooking, dancing, etc.)

**Dataset:** `facebook/PE-Video` or `lmms-lab/Video-MME`
**Model:** Fine-tune vision transformer (ViT) or `Qwen/Qwen2.5-VL-72B-Instruct`

**Input:**
```
Video: [25 frames of a person dancing]
```

**Output:**
```json
{
  "primary_action": "dancing",
  "confidence": 0.94,
  "secondary_actions": ["moving arms", "stepping"],
  "attributes": {
    "dance_style": "hip-hop",
    "intensity": "high"
  }
}
```

---

### 4. Video Summarization

**Goal:** Create short summaries of long videos

**Dataset:** `lmms-lab/LLaVA-Video-178K`
**Model:** `Qwen/Qwen2.5-VL-72B-Instruct` or `Llama-3.1-70B-Instruct`

**Input:**
```
Long video: 5-minute video (300 frames)
```

**Output:**
```
Summary: "This video shows a cooking tutorial for making pasta. 
The chef demonstrates: (1) boiling water, (2) adding pasta, 
(3) preparing sauce, (4) combining ingredients. Key tip: Save pasta 
water for sauce."
```

---

### 5. Video-to-Story Generation

**Goal:** Generate narrative stories from video content

**Dataset:** `lmms-lab/LLaVA-Video-178K`
**Model:** `Qwen/Qwen2.5-14B-Instruct` or `Llama-3.1-70B-Instruct`

**Input:**
```
Video: [person walking through park]
```

**Output:**
```
Story: "On a sunny afternoon, Sarah decided to take a walk through 
the neighborhood park. She noticed the vibrant autumn leaves falling 
gently from the trees. A gentle breeze rustled through the branches 
as she continued her peaceful stroll, lost in thought about the day ahead."
```

---

### 6. Video Scene Detection

**Goal:** Detect scene changes and segment videos

**Dataset:** `facebook/PE-Video` or `lmms-lab/Video-MME`
**Model:** Fine-tune temporal model (e.g., VideoMAE)

**Input:**
```
Video: [continuous video stream]
```

**Output:**
```json
{
  "scenes": [
    {
      "start": 0,
      "end": 30,
      "description": "Introduction scene - person entering room"
    },
    {
      "start": 30,
      "end": 60,
      "description": "Cooking scene - preparing ingredients"
    },
    {
      "start": 60,
      "end": 90,
      "description": "Final scene - presenting finished dish"
    }
  ]
}
```

---

### 7. Video Emotion Recognition

**Goal:** Detect emotions from video (facial expressions, body language)

**Dataset:** `lmms-lab/LLaVA-Video-178K` + emotion labels
**Model:** Fine-tune vision-language model

**Input:**
```
Video: [person's face showing emotion]
```

**Output:**
```json
{
  "primary_emotion": "happy",
  "confidence": 0.89,
  "secondary_emotions": ["excited", "content"],
  "intensity": 0.85,
  "facial_expressions": ["smiling", "eyes_crinkled"]
}
```

---

### 8. Video Object Tracking

**Goal:** Track objects across video frames

**Dataset:** `lmms-lab/Video-MME` or custom tracking dataset
**Model:** Fine-tune object detection model (YOLO, DETR)

**Input:**
```
Video: [frames with multiple objects]
```

**Output:**
```json
{
  "objects": [
    {
      "id": 1,
      "class": "person",
      "track": [
        {"frame": 0, "bbox": [100, 200, 150, 300]},
        {"frame": 1, "bbox": [110, 205, 155, 305]},
        {"frame": 2, "bbox": [120, 210, 160, 310]}
      ]
    },
    {
      "id": 2,
      "class": "car",
      "track": [...]
    }
  ]
}
```

---

### 9. Video Style Transfer

**Goal:** Transfer artistic style to videos

**Dataset:** `facebook/PE-Video` + style examples
**Model:** Fine-tune diffusion model (e.g., Stable Video Diffusion)

**Input:**
```
Original video: [normal video]
Style reference: "anime style" or "oil painting style"
```

**Output:**
```
Stylized video: [video with artistic style applied]
```

---

### 10. Video-to-Text Retrieval

**Goal:** Find videos based on text queries

**Dataset:** `lmms-lab/LLaVA-Video-178K`
**Model:** Fine-tune video-text embedding model

**Input:**
```
Query: "A person cooking pasta in a kitchen"
```

**Output:**
```
Retrieved videos:
1. video_123.mp4 (similarity: 0.95)
2. video_456.mp4 (similarity: 0.88)
3. video_789.mp4 (similarity: 0.82)
```

---

## 💻 Fine-Tuning Setup for 8 GPUs

### Multi-GPU Training Setup

```python
import torch
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator

# Use all 8 GPUs
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,  # or bf16 for MI300X
    dataloader_num_workers=8,
    ddp_find_unused_parameters=False,
    # Multi-GPU settings
    multi_gpu=True,
    # Use all GPUs
    local_rank=-1,
)

# Model automatically uses all GPUs with device_map="auto"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically splits across 8 GPUs
)
```

---

## 📊 Dataset Recommendations for Your Setup

**With 1.5TB disk + 8 GPUs, you can:**

✅ **Law:**
- `pile-of-law/pile-of-law` (large, but manageable)
- `refugee-law-lab/canadian-legal-data` (smaller)

✅ **Video:**
- `lmms-lab/LLaVA-Video-178K` (178K pairs - good size)
- `facebook/PE-Video` (1M videos - use streaming)
- `datahiveai/Tiktok-Videos` (smaller TikTok dataset)

---

## Quick Start: Law Fine-Tuning

```bash
python3 << EOF
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
import torch

# Load dataset
dataset = load_dataset("pile-of-law/pile-of-law", "all", streaming=True)

# Load model (uses all 8 GPUs)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Fine-tune...
EOF
```

---

## Quick Start: Video Fine-Tuning

```bash
python3 << EOF
from datasets import load_dataset
from transformers import AutoModelForVision2Seq
import torch

# Load video dataset
dataset = load_dataset("lmms-lab/LLaVA-Video-178K")

# Load vision-language model
model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Fine-tune...
EOF
```

---

## Summary

**You can download:**
- ✅ `pile-of-law/pile-of-law` (law)
- ✅ `lmms-lab/LLaVA-Video-178K` (video)
- ✅ `facebook/PE-Video` (use streaming)

**Best for fine-tuning:**
- **Law:** `Qwen/Qwen2.5-14B-Instruct` or `Llama-3.1-70B-Instruct`
- **Video:** `Qwen/Qwen2.5-VL-72B-Instruct`

**With 8 GPUs:** Models automatically split across all GPUs with `device_map="auto"`!

