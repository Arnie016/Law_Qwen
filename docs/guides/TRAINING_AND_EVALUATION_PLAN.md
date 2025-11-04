# Training & Evaluation Plan

## What We're Doing

### 1. NEW Fine-Tuning (10,000 steps)
- **Current:** Starting fresh training run
- **Steps:** 10,000 (vs 500 previously)
- **Goal:** Actually improve legal reasoning
- **Expected:** +2-5 points improvement

### 2. Re-Evaluation After Training
- **Compare:** NEW fine-tuned (10k) vs Base model
- **Also Compare:** NEW fine-tuned (10k) vs OLD fine-tuned (500 steps)
- **Includes:** t-statistic tests for statistical significance

---

## Evaluation Comparison Plan

### Comparison 1: NEW vs Base
```
Base Model (untrained)
  ↓
NEW Fine-Tuned (10,000 steps)
  ↓
Run evaluation script
```

### Comparison 2: NEW vs OLD
```
OLD Fine-Tuned (500 steps) - Score: 6.50/20
  ↓
NEW Fine-Tuned (10,000 steps) - Expected: 8-10/20
  ↓
Run comparison script
```

---

## The Evaluation Script Already Includes:

✅ **t-statistic tests** (lines 418-423)
✅ **Statistical significance** (|t|>2 & |Δ|>2)
✅ **Mean difference** calculation
✅ **Standard deviation** of differences
✅ **Win/loss counting**

---

## After Training Completes (~80 hours):

### Step 1: Run Standard Evaluation (NEW vs Base)
```bash
cd /root/scripts/evaluation
python3 eval_legal_models_scientific.py \
    --base_model Qwen/Qwen2.5-32B-Instruct \
    --fine_tuned_model ./qwen2.5-32b-law-finetuned
```

### Step 2: Run Three-Way Comparison (Optional)
```bash
# Compare: Base vs OLD (500) vs NEW (10k)
python3 compare_all_models.py \
    --base_model Qwen/Qwen2.5-32B-Instruct \
    --old_model ./qwen2.5-32b-law-finetuned-500steps \
    --new_model ./qwen2.5-32b-law-finetuned
```

---

## Expected Results

### Current (500 steps):
- Base: 6.61/20
- Fine-Tuned: 6.50/20
- Difference: -0.11 ❌
- t-statistic: Not significant

### Expected (10,000 steps):
- Base: 6.61/20
- Fine-Tuned: 8-10/20 ✅
- Difference: +1.5-3.5 points
- t-statistic: Should be significant (|t|>2)

---

## Summary

**Yes, we're fine-tuning again** → NEW 10k-step training

**Yes, we'll run t-statistic tests again** → After training completes

**The evaluation script already includes t-tests** → Just run it after training

---

## Quick Commands After Training

```bash
# 1. Evaluation (NEW vs Base)
cd /root/scripts/evaluation
python3 eval_legal_models_scientific.py

# 2. Check if improvement is significant
grep "t-statistic" legal_eval_results.csv
grep "Significant" legal_eval_results.csv
```

---

**TL;DR:** 
- Train 10k steps → Takes ~80 hours
- Run evaluation → Compares NEW vs Base (includes t-tests)
- Compare results → See if improvement is statistically significant

