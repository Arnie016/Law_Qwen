# Using Unsloth Notebooks with AMD GPUs

## ✅ Yes, You Can Use Unsloth Notebooks!

**Unsloth supports AMD GPUs/ROCm** - Your setup is compatible!

---

## 🎯 What You Need to Know

### Compatibility

**✅ Unsloth Works with AMD:**
- ROCm support included
- MI300X GPUs supported
- Optimized for AMD hardware

**⚠️ But Note:**
- Unsloth notebooks are often designed for CUDA/NVIDIA
- Some notebooks may need minor modifications
- Our script already handles ROCm-specific issues

---

## 📝 Can You Use Unsloth Notebooks Directly?

### Option 1: Use Unsloth Notebooks (Recommended for Learning)

**Pros:**
- ✅ Well-documented examples
- ✅ Community support
- ✅ Multiple model examples
- ✅ Step-by-step tutorials

**Cons:**
- ⚠️ May need ROCm adjustments
- ⚠️ Designed for CUDA (but works on ROCm)
- ⚠️ May use bitsandbytes (disable on ROCm)

**How to Use:**
1. Download notebook from Unsloth GitHub
2. Upload to your Jupyter: http://129.212.184.211
3. Run cells - should work with ROCm!
4. If bitsandbytes error → disable it (see below)

---

## 🔧 Modifications Needed for ROCm

### 1. Disable Bitsandbytes (If Needed)

**Unsloth notebooks often use:**
```python
load_in_4bit=True  # Uses bitsandbytes
```

**For ROCm, change to:**
```python
load_in_4bit=False  # Disable bitsandbytes
torch_dtype=torch.bfloat16  # Use full precision instead
```

### 2. Use Standard Optimizer

**Unsloth notebooks may use:**
```python
optim="adamw_8bit"  # Uses bitsandbytes
```

**For ROCm, change to:**
```python
optim="adamw_torch"  # Standard PyTorch optimizer
```

### 3. Check GPU Detection

**Add this check:**
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
# Should show True on ROCm
```

---

## 📊 Comparison: Unsloth Notebook vs Our Script

| Aspect | Unsloth Notebook | Our Script |
|--------|------------------|------------|
| **Model** | Various (8B-72B) | Qwen 2.5 32B |
| **GPU Support** | CUDA + ROCm | ROCm optimized |
| **Bitsandbytes** | Often enabled | Disabled |
| **Optimizer** | May use 8-bit | Standard AdamW |
| **Reward Function** | Examples provided | Legal-specific |
| **ROCm Fixes** | Manual changes needed | Pre-configured |

---

## 🎯 When to Use Each

### Use Unsloth Notebooks If:
- ✅ Learning GRPO/SFT concepts
- ✅ Testing different models
- ✅ Quick prototyping
- ✅ Following tutorials

### Use Our Script If:
- ✅ Legal domain fine-tuning
- ✅ Production training
- ✅ ROCm-specific optimizations
- ✅ Already have our setup

---

## 🚀 Quick Start: Using Unsloth Notebook

### Step 1: Download Notebook

**From Unsloth GitHub:**
- https://github.com/unslothai/unsloth
- Or search for "Qwen GRPO notebook"

**Example notebooks:**
- Qwen3-14B GRPO notebook
- Llama3-8B GRPO notebook
- Mistral-7B GRPO notebook

### Step 2: Upload to Jupyter

1. Open Jupyter: http://129.212.184.211
2. Click "Upload"
3. Upload the `.ipynb` file
4. Open it

### Step 3: Modify for ROCm

**Find and replace:**
```python
# Find:
load_in_4bit=True

# Replace with:
load_in_4bit=False
torch_dtype=torch.bfloat16
```

**Find and replace:**
```python
# Find:
optim="adamw_8bit"

# Replace with:
optim="adamw_torch"
```

### Step 4: Run Cells

- Execute cells one by one
- Should work with your AMD GPU!

---

## ⚠️ Common Issues with Unsloth Notebooks on ROCm

### Issue 1: Bitsandbytes Error

**Error:**
```
bitsandbytes not found / ROCm binary not found
```

**Fix:**
```python
# Disable bitsandbytes
load_in_4bit=False
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
```

### Issue 2: 8-bit Optimizer Error

**Error:**
```
adamw_8bit not available
```

**Fix:**
```python
# Use standard optimizer
optim="adamw_torch"
```

### Issue 3: GPU Not Detected

**Error:**
```
CUDA not available
```

**Fix:**
```python
# Check ROCm PyTorch
import torch
print(torch.cuda.is_available())  # Should be True
# If False, check Docker container
```

---

## 💡 Recommendation

### For Your Legal Fine-Tuning Project:

**Use Our Script** because:
1. ✅ Already ROCm-optimized
2. ✅ Legal-specific reward function
3. ✅ Proper dataset handling
4. ✅ All fixes already applied

**Use Unsloth Notebooks** for:
1. ✅ Learning GRPO concepts
2. ✅ Testing different models
3. ✅ Quick experiments
4. ✅ Understanding best practices

---

## 🔗 Links

**Unsloth Documentation:**
- AMD/ROCm Guide: https://docs.unsloth.ai/get-started/install-and-update/amd
- Notebooks: https://github.com/unslothai/unsloth/tree/main/notebooks

**Our Scripts:**
- GRPO Legal Training: `scripts/grpo/grpo_legal_training.py`
- Comparison Guide: `docs/guides/GRPO_SCRIPT_VS_UNSLOTH_NOTEBOOKS.md`

---

## ✅ Bottom Line

**Yes, you can use Unsloth notebooks!** They work with AMD GPUs, but you may need to:
1. Disable bitsandbytes (`load_in_4bit=False`)
2. Use standard optimizer (`optim="adamw_torch"`)
3. Check GPU detection

**Our script is already optimized for ROCm**, so if you're doing legal fine-tuning, stick with our script. But Unsloth notebooks are great for learning and experimentation!

