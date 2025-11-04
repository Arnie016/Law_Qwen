# GRPO Script Review - Potential Issues

## Issues Found and Fixed

### ✅ FIXED: Reward Function Signature

**Problem:** GRPO API expects `reward_fn(prompts, responses, **kwargs)` not `(response, prompt)`

**Fix:** Created wrapper function that matches GRPO API

### ✅ FIXED: Reward Function Location

**Problem:** Had `reward_fn` in both GRPOConfig and GRPOTrainer

**Fix:** Removed from GRPOConfig, kept only in GRPOTrainer (correct location)

---

## Remaining Potential Issues

### 1. Dataset Format ⚠️

GRPO might expect specific dataset format. Current script formats as:
```python
{"prompt": ["prompt1", "prompt2", ...]}
```

This should work, but if it doesn't, might need:
```python
{"text": ["prompt1", "prompt2", ...]}  # Some versions expect "text"
```

### 2. Checkpoint Loading ⚠️

Script tries to load checkpoint-500, but:
- Might not exist in container
- Might need different path
- Script handles this gracefully (falls back to base model)

### 3. Unsloth Compatibility ⚠️

GRPO might need specific Unsloth version or TRL version. If errors occur:
- Check TRL version: `pip show trl`
- May need: `pip install trl>=0.8.0`

### 4. Memory Usage ⚠️

32B model + 4 generations per prompt = high memory usage
- Script uses LoRA (efficient)
- But might need to reduce batch size if OOM errors

---

## Testing Before Full Run

Run a quick test first:

```bash
# Test with small dataset and few steps
# Modify script temporarily:
max_steps=10  # Test with 10 steps first
per_device_train_batch_size=1  # Smaller batch
```

---

## If Script Fails

### Error: "reward_fn signature mismatch"
- Fix: Already fixed in updated script

### Error: "Dataset format incorrect"
- Fix: Try changing `"prompt"` to `"text"` in dataset

### Error: "GRPOConfig doesn't accept num_generations"
- Fix: Remove from config, set in trainer instead

### Error: "Out of memory"
- Fix: Reduce batch size or num_generations

---

## Updated Script Location

Fixed script: `scripts/grpo/grpo_legal_training.py`

**Changes:**
- ✅ Fixed reward function signature
- ✅ Moved reward_fn to GRPOTrainer only
- ✅ Added proper wrapper function

---

## Recommendation

**Try the script - it should work now.** If errors occur:

1. Test with small `max_steps=10` first
2. Check error messages
3. Fix based on actual API (TRL version might differ)

The script structure is correct, just API details might need adjustment based on TRL version.

