# Critical Rules for Legal LLM Fine-Tuning

## 🚨 CORE PRINCIPLES

### Rule 1: Reasoning ≠ Legal Knowledge
**CRITICAL:** Strong reasoning models (O3, GPT-OSS, Qwen) do NOT automatically have legal knowledge.
- Reasoning models can solve logic puzzles but don't know UCC § 2-205
- Legal models need BOTH: legal facts + legal reasoning
- Don't assume reasoning models will perform well on legal tasks without fine-tuning

### Rule 2: Legal Knowledge Must Come First
**ORDER MATTERS:** Build legal knowledge BEFORE improving reasoning.
1. First: SFT on legal datasets (10K+ steps) → Build knowledge
2. Then: GRPO/RL → Improve reasoning quality
- GRPO on weak knowledge base = only improves formatting, not understanding
- 500 steps is insufficient (need 10K+ for meaningful improvement)

### Rule 3: Dataset Size Matters
**USE FULL DATASETS:**
- `pile-of-law/pile-of-law` (full) > `lamblamb/pile_of_law_subset` (subset)
- Full Pile of Law = millions of examples vs 41K subset
- LegalBench = Q&A format (better than raw text)
- Streaming mode for large datasets

---

## 🛠️ TECHNICAL RULES

### Rule 4: Unsloth GRPO API Requirements
**CRITICAL:** Unsloth has specific API requirements:
- `reward_funcs` (plural) - must be a LIST, even if single function
- Use `*args, **kwargs` in reward function to handle any calling pattern
- Unsloth may call with `prompts`/`responses` OR `inputs`/`completions`
- Always handle both naming conventions

```python
# CORRECT:
def grpo_reward_fn(*args, **kwargs):
    prompts = kwargs.get('prompts') or kwargs.get('inputs') or (args[0] if args else [])
    responses = kwargs.get('responses') or kwargs.get('completions') or (args[1] if len(args) > 1 else [])
    # ... rest of function

trainer = GRPOTrainer(
    reward_funcs=[grpo_reward_fn],  # Must be LIST
)

# WRONG:
trainer = GRPOTrainer(
    reward_fn=grpo_reward_fn,  # ❌ Wrong parameter name
)
```

### Rule 5: Bitsandbytes on ROCm
**DISABLE BITSANDBYTES on ROCm:**
- ROCm doesn't support bitsandbytes well
- Use `optim="adamw_torch"` (standard PyTorch optimizer)
- Set environment variables: `BITSANDBYTES_NOWELCOME=1`, `DISABLE_BITSANDBYTES_AUTO_INSTALL=1`
- Use full precision (bf16/fp16), not 8-bit

```python
# CORRECT:
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
grpo_config = GRPOConfig(
    optim="adamw_torch",  # Standard optimizer
    bf16=True,  # Full precision
)

# WRONG:
# Don't use load_in_4bit=True or 8-bit optimizers on ROCm
```

### Rule 6: Docker Container Requirements
**PYTORCH ONLY IN CONTAINER:**
- DigitalOcean PyTorch (ROCm7) droplets have PyTorch INSIDE Docker container
- Always run: `docker exec -it rocm /bin/bash` before training
- Scripts must be downloaded INSIDE container, not on host

### Rule 7: Batch Size for GRPO
**MUST BE MULTIPLE OF num_generations:**
- `per_device_train_batch_size` must be multiple of `num_generations`
- Example: `num_generations=4` → batch_size must be 4, 8, 12, etc.
- Unsloth will auto-adjust if wrong, but specify correctly

```python
# CORRECT:
grpo_config = GRPOConfig(
    per_device_train_batch_size=4,  # Multiple of 4
    num_generations=4,
)

# WRONG:
per_device_train_batch_size=3,  # ❌ Not multiple of 4
```

---

## 📊 TRAINING RULES

### Rule 8: Training Steps Requirements
**MINIMUM STEPS:**
- SFT: 10,000+ steps for meaningful improvement (not 500)
- GRPO: 1,000 steps is reasonable for RL fine-tuning
- 500 steps = insufficient, barely changes model behavior

### Rule 9: Evaluation Expectations
**REALISTIC EXPECTATIONS:**
- Base Qwen 2.5 32B: ~6.5/20 on legal questions
- 500-step SFT: No improvement (may even be worse)
- 10K-step SFT: Expect 8-10/20 (+1.5-3.5 points)
- 10K SFT + GRPO: Expect 10-12/20 (+3.5-5.5 points)

### Rule 10: Reward Function Design
**REWARD FUNCTION MUST:**
- Score responses based on legal quality (not just length)
- Check for: legal terminology, citations, IRAC structure, analysis depth
- Return list of floats (one per response)
- Handle empty/malformed inputs gracefully

```python
# GOOD REWARD FUNCTION:
def legal_reward_function(response, question):
    reward = 0.0
    # Check legal terminology
    # Check citations (§, UCC, cases)
    # Check structure (IRAC)
    # Check analysis depth
    # Penalize short/generic responses
    return reward  # Float between -5 and +20
```

---

## 🎯 MODEL SELECTION RULES

### Rule 11: Model Size for Legal Tasks
**Qwen 2.5 32B is Good Choice:**
- Optimal size for 192GB GPU (fits comfortably)
- Better than GPT-OSS-20B (more capacity)
- Can't use GPT-OSS-120B (too large for GPU)
- 72B also works but slower training

### Rule 12: Legal-Specific Models Exist
**CONSIDER SPECIALIZED MODELS:**
- Unilaw-R1 (7B): Legal reasoning + knowledge
- ChatLaw: Legal LLM with RAG
- But: Qwen 2.5 32B fine-tuned properly can match them

---

## 📁 FILE ORGANIZATION RULES

### Rule 13: Repository Structure
**ORGANIZE BY FUNCTION:**
```
scripts/
├── training/          # SFT scripts
├── grpo/             # RL scripts
├── evaluation/        # Evaluation scripts
└── utils/             # Utility scripts

docs/
├── guides/            # How-to guides
└── analysis/          # Analysis docs

models/
└── checkpoints/       # Model weights (Git LFS)
```

---

## 🔍 DEBUGGING RULES

### Rule 14: Check GPU Availability
**ALWAYS VERIFY GPU:**
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
# Must show GPU=True before training
```

### Rule 15: Check Dataset Format
**VERIFY DATASET STRUCTURE:**
- Legal datasets should have `text` or `question`/`answer` fields
- GRPO needs `prompt` field (formatted)
- Always check: `dataset["train"][0].keys()`

### Rule 16: Error Handling
**COMMON ERRORS:**
- `TypeError: missing 1 required positional argument` → Check reward function signature
- `bitsandbytes error` → Disable bitsandbytes, use `optim="adamw_torch"`
- `GPU not detected` → Check PyTorch ROCm installation, use Docker container
- `File not found` → Scripts must be in container, not on host

---

## 📝 DOCUMENTATION RULES

### Rule 17: Always Document Results
**RECORD:**
- Training steps completed
- Evaluation scores (base vs fine-tuned)
- Dataset used
- Configuration parameters
- Issues encountered and fixes

### Rule 18: Version Control
**COMMIT FREQUENTLY:**
- Script changes
- Configuration updates
- Evaluation results
- Documentation

---

## ⚠️ COMMON MISTAKES TO AVOID

1. ❌ Using reasoning models expecting legal knowledge
2. ❌ Running scripts on host instead of Docker container
3. ❌ Using bitsandbytes on ROCm
4. ❌ Using `reward_fn` instead of `reward_funcs` (plural)
5. ❌ Training only 500 steps expecting improvement
6. ❌ Using subset dataset instead of full dataset
7. ❌ Running GRPO before building legal knowledge
8. ❌ Expecting dramatic improvement from small changes
9. ❌ Not checking GPU availability before training
10. ❌ Not handling reward function argument variations

---

## ✅ BEST PRACTICES SUMMARY

1. **Knowledge First:** Build legal knowledge with 10K+ SFT steps
2. **Full Datasets:** Use complete datasets, not subsets
3. **Then Reasoning:** Apply GRPO after knowledge is built
4. **Proper APIs:** Follow Unsloth/TRL API requirements exactly
5. **ROCm Setup:** Disable bitsandbytes, use standard optimizers
6. **Container Usage:** Always work inside Docker container
7. **Realistic Goals:** Expect gradual improvement, not miracles
8. **Documentation:** Record everything for future reference

---

**Last Updated:** Based on Qwen 2.5 32B legal fine-tuning project
**Key Lesson:** Reasoning ≠ Knowledge. Legal models need both, built in correct order.

