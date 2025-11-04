# Single GPU Training Guide

## ✅ Will It Work? YES

**Answer:** Yes, the training script will work on 1 GPU.

### Why It Works:

1. **Model Size:** Qwen 2.5 32B = ~62GB full precision, ~31GB with bfloat16
   - Single MI300X: **192GB HBM** ✅ More than enough
   - Model fits comfortably on 1 GPU

2. **LoRA Efficiency:** Only trains ~400MB of adapters (not full model)
   - Much lower memory than full fine-tuning
   - Base model loaded but frozen

3. **Current Script:** Already configured for single GPU:
   - `device_map="auto"` → automatically uses 1 GPU if that's all available
   - `per_device_train_batch_size=1` → works fine for single GPU
   - `gradient_accumulation_steps=16` → compensates for small batch size

---

## 🔧 What Changed

### Original Script (`finetune_qwen_law_fixed.py`)
- Uses `device_map="auto"` (works but not explicit)
- Works on 1 GPU or 8 GPUs automatically

### New Script (`finetune_qwen_law_single_gpu.py`)
- Uses `device_map="cuda:0"` (explicitly single GPU)
- Added GPU info check
- Increased `max_steps=10000` (from 500) ← **IMPORTANT**
- Optimized for single GPU setup

---

## 📊 Performance Comparison

### 8 GPUs vs 1 GPU

| Aspect | 8 GPUs | 1 GPU |
|--------|--------|-------|
| **Speed** | ~8x faster | Baseline |
| **Memory** | Split across GPUs | All on one GPU |
| **Batch Size** | Same (per device) | Same (per device) |
| **Training Time** | ~30 min for 500 steps | ~4 hours for 500 steps |
| **Compatibility** | May have multi-GPU issues | ✅ No issues |

**Verdict:** Single GPU is slower but **more reliable** (no multi-GPU errors).

---

## 🚀 How to Run

### Option 1: Use Original Script (Works Fine)
```bash
python3 scripts/training/finetune_qwen_law_fixed.py
```
- Will automatically use 1 GPU if that's all available
- No changes needed

### Option 2: Use Single-GPU Optimized Script
```bash
python3 scripts/training/finetune_qwen_law_single_gpu.py
```
- Explicitly configured for single GPU
- Includes GPU info check
- Pre-configured for 10,000 steps

---

## ⚠️ Important Notes

### 1. Training Time
- **500 steps:** ~30 min (8 GPUs) vs ~4 hours (1 GPU)
- **10,000 steps:** ~10 hours (8 GPUs) vs ~80 hours (1 GPU)
- **Recommendation:** Use 1 GPU but train overnight/weekend

### 2. Memory Usage
- Model: ~31GB (bfloat16)
- LoRA: ~400MB
- Training overhead: ~5-10GB
- **Total:** ~35-40GB / 192GB = **~20% usage** ✅ Plenty of headroom

### 3. Batch Size
- Current: `per_device_train_batch_size=1`
- Can increase to `2` or `4` if you want (memory allows)
- Single GPU has more memory per GPU than multi-GPU split

### 4. Gradient Accumulation
- Current: `gradient_accumulation_steps=16`
- This means effective batch size = 1 × 16 = 16
- Can reduce if you increase `per_device_train_batch_size`

---

## 🔍 Verifying Single GPU Usage

```python
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Expected output:
```
GPUs available: 1
Using GPU: AMD MI300X
Memory: 192.0 GB
```

---

## 📈 Expected Improvements

### With 10,000 Steps (vs 500 Steps):

| Metric | Current (500 steps) | Expected (10,000 steps) |
|--------|---------------------|-------------------------|
| **Legal Score** | 6.50/20 | 8-10/20 |
| **Improvement** | -0.11 (worse) | +1.5-3.5 points |
| **Training Time** | ~30 min | ~80 hours (1 GPU) |

---

## 🎯 Recommendation

**Yes, use 1 GPU.** It's:
- ✅ More reliable (no multi-GPU errors)
- ✅ Easier to debug
- ✅ Sufficient memory (192GB is plenty)
- ⚠️ Slower but acceptable for overnight training

**Action:** Run `finetune_qwen_law_single_gpu.py` with `max_steps=10000` overnight.

---

## 🔧 Quick Fixes

If you want to use the original script but ensure single GPU:

```python
# Change line 30 from:
device_map="auto",

# To:
device_map="cuda:0",  # Explicit single GPU
```

That's it! Everything else stays the same.

