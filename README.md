# AMD MI300X Legal Fine-Tuning Project

Fine-tuning Qwen 2.5 32B on legal datasets using 8x AMD MI300X GPUs.

## Project Overview

- **Model**: Qwen 2.5 32B Instruct
- **Hardware**: 8x AMD MI300X (1.5TB HBM total)
- **Dataset**: pile_of_law_subset
- **Method**: LoRA fine-tuning

## Key Files

### Fine-Tuning Scripts
- `finetune_qwen_law_fixed.py` - Main fine-tuning script (500 steps)
- `eval_legal_models_scientific.py` - Comprehensive evaluation (10 legal questions)
- `compare_base_vs_finetuned.py` - Simple comparison script

### GRPO Training
- `unsloth_grpo_prompt_injection.py` - GRPO RL fine-tuning for prompt injection game
- `GRPO_TRAINING_COMPLETE_GUIDE.md` - Complete GRPO guide

### Evaluation Results
- `legal_eval_results.csv` - Evaluation results (if available)
- **Findings**: No significant improvement after 500 steps (-0.11 points)

### Documentation
- `finetune_qwen_law.md` - Fine-tuning guide
- `model_size_comparison.md` - Model size analysis
- `finetuning_analysis_recommendations.py` - Analysis of results

## Quick Start

### Fine-Tuning
```bash
docker exec -it rocm /bin/bash
python3 finetune_qwen_law_fixed.py
```

### Evaluation
```bash
python3 eval_legal_models_scientific.py
```

### GRPO Training
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python3 unsloth_grpo_prompt_injection.py
```

## Results Summary

**Evaluation Results:**
- Base Model: 6.61/20
- Fine-Tuned: 6.50/20
- Difference: -0.11 (not significant)
- Conclusion: Need more training (10,000+ steps recommended)

**Key Findings:**
- 500 steps insufficient for improvement
- Both models scored low (legal questions are hard)
- 8/10 questions had identical scores
- Need domain-specific fine-tuning or more training

## Next Steps

1. Increase training to 10,000+ steps
2. Use domain-specific datasets
3. Consider RAG approach for legal knowledge
4. Test GRPO training for prompt injection game

## Server Info

- SSH: `ssh -i ~/.ssh/id_ed25519 root@134.199.192.60`
- Docker: `docker exec -it rocm /bin/bash`
- Models: `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct`
- Checkpoints: `/qwen2.5-32b-law-finetuned/checkpoint-500/`

## License

MIT License
