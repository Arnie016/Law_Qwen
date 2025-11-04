# Legal Fine-Tuning with Qwen 2.5 32B on AMD MI300X

Fine-tuning Qwen 2.5 32B on legal datasets using 8x AMD MI300X GPUs (1.5TB HBM).

## 📁 Repository Structure

```
├── scripts/
│   ├── training/          # Fine-tuning scripts
│   ├── evaluation/        # Model evaluation scripts
│   ├── grpo/             # GRPO RL training
│   └── utils/             # Utility scripts
├── docs/
│   ├── guides/            # Documentation and guides
│   └── analysis/          # Analysis and comparisons
├── models/
│   └── checkpoints/       # Model weights (Git LFS)
└── data/
    └── evaluation/        # Evaluation results
```

## 🚀 Quick Start

### Fine-Tuning
```bash
cd scripts/training
python3 finetune_qwen_law_fixed.py
```

### Evaluation
```bash
cd scripts/evaluation
python3 eval_legal_models_scientific.py
```

### GRPO Training
```bash
cd scripts/grpo
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python3 unsloth_grpo_prompt_injection.py
```

## 📊 Results

**Evaluation Results:**
- Base Model: 6.61/20
- Fine-Tuned: 6.50/20
- Difference: -0.11 (not significant)
- **Conclusion:** Need more training (10,000+ steps recommended)

**Model Weights:**
- Checkpoint-500: LoRA adapters (~500MB)
- Available in `models/checkpoints/` (Git LFS)

## 📚 Documentation

- **Fine-Tuning Guide:** `docs/guides/finetune_qwen_law.md`
- **GRPO Training:** `docs/guides/GRPO_TRAINING_COMPLETE_GUIDE.md`
- **Transfer Guide:** `docs/guides/TRANSFER_GUIDE.md`
- **Model Analysis:** `docs/analysis/model_size_comparison.md`

## 🛠️ Setup

### Requirements
- 8x AMD MI300X GPUs (or compatible)
- Docker with ROCm support
- Python 3.10+

### Installation
```bash
# Clone repository
git clone https://github.com/Arnie016/Law_Qwen.git
cd Law_Qwen

# Install dependencies
pip install -r requirements.txt

# Setup Git LFS (for model weights)
git lfs install
```

## 📦 Model Weights

Model weights are stored using Git LFS (Large File Storage).

To download:
```bash
git lfs pull
```

Weights location: `models/checkpoints/checkpoint-500/`

## 🔗 Links

- **Repository:** https://github.com/Arnie016/Law_Qwen
- **Server:** AMD DevCloud (8x MI300X)
- **Base Model:** Qwen/Qwen2.5-32B-Instruct
- **Dataset:** pile_of_law_subset

## 📝 Key Files

- `scripts/training/finetune_qwen_law_fixed.py` - Main fine-tuning script
- `scripts/evaluation/eval_legal_models_scientific.py` - Comprehensive evaluation
- `scripts/grpo/unsloth_grpo_prompt_injection.py` - GRPO RL training
- `docs/guides/TRANSFER_GUIDE.md` - Server migration guide

## 🤝 Contributing

This is a research project. Feel free to fork and experiment!

## 📄 License

MIT License
