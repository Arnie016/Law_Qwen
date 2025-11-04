# Fast RL Fine-Tuning Plan (After 500 Steps SFT)

## Strategy: SFT (500 steps) → RL Fine-Tuning (GRPO)

**Why RL instead of more SFT:**
- ✅ Faster (500-1000 RL steps vs 10,000 SFT steps)
- ✅ Better alignment (rewards guide improvement)
- ✅ More efficient (focused on better responses)
- ✅ Cheaper (~$10-20 vs $160)

---

## Plan

### Step 1: Current Status
- ✅ Base model: 6.61/20
- ✅ SFT 500 steps: 6.50/20 (not better)
- ⏭️ Skip more SFT steps

### Step 2: RL Fine-Tuning (GRPO)
- Start from checkpoint-500 (500-step SFT)
- Train 500-1000 GRPO steps
- Use legal reasoning reward function
- Expected time: ~4-8 hours
- Expected improvement: +2-4 points

---

## Why RL Will Work Better

### Problem with SFT:
- Model learns format, not reasoning
- Generic legal text ≠ legal Q&A
- No guidance on what's "good"

### Solution with RL:
- Reward function guides improvement
- Focuses on legal reasoning quality
- Learns what makes good legal answers
- Faster convergence

---

## Quick Setup for GRPO Legal Training

### Option 1: Use Existing GRPO Script (Modify for Legal)

```bash
# Download GRPO script
cd /root/scripts/grpo
curl -o grpo_legal_training.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/grpo/unsloth_grpo_prompt_injection.py

# Modify reward function for legal reasoning
```

### Option 2: Use LegalBench Dataset with GRPO

Better dataset = better results

```bash
# Use LegalBench instead of generic legal text
from datasets import load_dataset
dataset = load_dataset("nguha/legalbench", "all")
```

---

## Cost Comparison

| Method | Steps | Time | Cost |
|--------|-------|------|------|
| SFT 10k | 10,000 | ~80 hours | ~$160 |
| RL GRPO | 500-1000 | ~4-8 hours | ~$10-20 |
| **Savings** | | | **~$140** |

---

## Expected Results

### Current:
- Base: 6.61/20
- SFT 500: 6.50/20 ❌

### After RL:
- Base: 6.61/20
- SFT 500 + RL 500: 8-10/20 ✅
- Improvement: +1.5-3.5 points

---

## Quick Start Commands

```bash
# 1. Load checkpoint-500
cd /root/scripts/training
# Model should be at: ./qwen2.5-32b-law-finetuned/checkpoint-500/

# 2. Run GRPO training
cd /root/scripts/grpo
python3 grpo_legal_training.py
```

---

## Recommendation

**YES, switch to RL after 500 steps.**

Reasons:
1. ✅ SFT not improving (6.50 vs 6.61)
2. ✅ RL can target legal reasoning directly
3. ✅ Much cheaper (~$20 vs $160)
4. ✅ Faster (hours vs days)
5. ✅ Better alignment with rewards

**Action:** Use GRPO with legal reward function, train 500-1000 steps.

