# Model Size Comparison for Fine-Tuning

## Qwen 2.5 Model Sizes

| Model | Parameters | Memory (BF16) | Training Speed | Quality | Best For |
|-------|-----------|---------------|----------------|---------|----------|
| **Qwen 2.5 7B** | 7B | ~14GB | Fastest | Good | Quick experiments, testing |
| **Qwen 2.5 14B** | 14B | ~28GB | Fast | Better | Balanced speed/quality |
| **Qwen 2.5 32B** | 32B | ~64GB | Medium | Best | Complex tasks, production |
| **Qwen 2.5 72B** | 72B | ~144GB | Slow | Excellent | Maximum quality |

## Your Setup: 8x MI300X with 1.5TB HBM

**Memory per GPU:** ~192GB (1.5TB / 8)
**32B Model:** ~64GB per GPU → ✅ Fits comfortably
**72B Model:** ~144GB per GPU → ✅ Also fits (but slower)

## Why 32B for Legal Fine-Tuning?

1. **Legal Complexity:**
   - Legal reasoning requires nuance
   - 32B captures subtle distinctions better
   - Better at handling multiple legal concepts simultaneously

2. **Your Hardware:**
   - 8 GPUs = plenty of memory
   - No need to use smaller models for memory reasons
   - Can handle 32B without quantization

3. **Fine-Tuning Quality:**
   - Larger models learn more from fine-tuning
   - Better retention of base knowledge
   - More consistent improvements

## When to Use Smaller Models?

- **7B/14B:** Quick experiments, testing prompts, faster iteration
- **32B:** Production fine-tuning (what you're doing)
- **72B:** Maximum quality (if you have time)

## Recommendation

**Stick with 32B for legal fine-tuning.** Your hardware supports it, and you'll get better results.

If you want to test GRPO faster, you could use 14B for initial experiments, then scale to 32B.

