#!/bin/bash
# Reorganize GitHub Repository
# Organize files into folders

set -e

echo "=========================================="
echo "REORGANIZING REPOSITORY"
echo "=========================================="

# Create folder structure
mkdir -p scripts/training
mkdir -p scripts/evaluation
mkdir -p scripts/grpo
mkdir -p scripts/utils
mkdir -p docs/guides
mkdir -p docs/analysis
mkdir -p models/checkpoints
mkdir -p data/evaluation

# Move training scripts
mv finetune_qwen_law*.py scripts/training/ 2>/dev/null || true
mv finetune_qwen_law*.md scripts/training/ 2>/dev/null || true
mv finetune_video_multimodal*.py scripts/training/ 2>/dev/null || true
mv finetune_video_multimodal*.md scripts/training/ 2>/dev/null || true
mv unsloth_dpo_finetune.py scripts/training/ 2>/dev/null || true
mv unsloth_rl_finetuning_guide.md scripts/training/ 2>/dev/null || true

# Move evaluation scripts
mv eval_legal_models_scientific.py scripts/evaluation/ 2>/dev/null || true
mv comprehensive_legal_evaluation.py scripts/evaluation/ 2>/dev/null || true
mv compare_base_vs_finetuned.py scripts/evaluation/ 2>/dev/null || true
mv test_finetuned_law_model*.py scripts/evaluation/ 2>/dev/null || true
mv load_law_model.py scripts/evaluation/ 2>/dev/null || true
mv access_finetuned_model.py scripts/evaluation/ 2>/dev/null || true

# Move GRPO scripts
mv unsloth_grpo_prompt_injection.py scripts/grpo/ 2>/dev/null || true
mv prompt_injection_grpo*.py scripts/grpo/ 2>/dev/null || true
mv prompt_injection_grpo*.md scripts/grpo/ 2>/dev/null || true
mv grpo_config_from_notebook.md scripts/grpo/ 2>/dev/null || true
mv grpo_training_guide.py scripts/grpo/ 2>/dev/null || true
mv how_grpo_rewards_work*.py scripts/grpo/ 2>/dev/null || true
mv how_grpo_rewards_work*.md scripts/grpo/ 2>/dev/null || true
mv four_rollouts_explained*.py scripts/grpo/ 2>/dev/null || true
mv four_rollouts_explained*.md scripts/grpo/ 2>/dev/null || true
mv complete_reward_function_examples.py scripts/grpo/ 2>/dev/null || true
mv prompt_structure_reward_guide.md scripts/grpo/ 2>/dev/null || true
mv what_is_a_prompt*.py scripts/grpo/ 2>/dev/null || true
mv what_is_a_prompt*.md scripts/grpo/ 2>/dev/null || true

# Move utils
mv check_*.sh scripts/utils/ 2>/dev/null || true
mv check_*.py scripts/utils/ 2>/dev/null || true
mv check_*.md scripts/utils/ 2>/dev/null || true
mv download_*.py scripts/utils/ 2>/dev/null || true
mv download_*.sh scripts/utils/ 2>/dev/null || true
mv show_pile_of_law_examples*.py scripts/utils/ 2>/dev/null || true
mv quick_*.py scripts/utils/ 2>/dev/null || true
mv quick_*.md scripts/utils/ 2>/dev/null || true
mv list_ollama_models.sh scripts/utils/ 2>/dev/null || true
mv generate_video.py scripts/utils/ 2>/dev/null || true
mv setup_*.sh scripts/utils/ 2>/dev/null || true

# Move guides
mv *_guide.md docs/guides/ 2>/dev/null || true
mv GRPO_TRAINING_COMPLETE_GUIDE.md docs/guides/ 2>/dev/null || true
mv TRANSFER_GUIDE.md docs/guides/ 2>/dev/null || true
mv finetune_qwen_law.md docs/guides/ 2>/dev/null || true
mv finetune_video_multimodal_guide.md docs/guides/ 2>/dev/null || true
mv server_setup_guide.md docs/guides/ 2>/dev/null || true
mv video_generation_guide.md docs/guides/ 2>/dev/null || true
mv huggingface_datasets_guide.md docs/guides/ 2>/dev/null || true
mv tiktok_datasets_guide.md docs/guides/ 2>/dev/null || true

# Move analysis
mv finetuning_analysis_recommendations.py docs/analysis/ 2>/dev/null || true
mv model_size_comparison.md docs/analysis/ 2>/dev/null || true
mv big_models_search_results.md docs/analysis/ 2>/dev/null || true
mv reasoning_models.md docs/analysis/ 2>/dev/null || true
mv model_downloads.md docs/analysis/ 2>/dev/null || true

# Move notebooks
mv *.ipynb docs/ 2>/dev/null || true

# Move transfer scripts
mv transfer_to_new_server.sh scripts/utils/ 2>/dev/null || true
mv quick_transfer.sh scripts/utils/ 2>/dev/null || true
mv upload_weights_to_hf.* scripts/utils/ 2>/dev/null || true
mv push_to_github.sh scripts/utils/ 2>/dev/null || true
mv setup_github.sh scripts/utils/ 2>/dev/null || true

# Move datasets info
mv law_video_datasets.md docs/guides/ 2>/dev/null || true
mv fine_tuning_scenarios.md docs/guides/ 2>/dev/null || true

# Keep README and .gitignore in root
# Keep upload scripts in root for easy access

echo "✅ Files organized!"
echo ""
echo "New structure:"
tree -L 2 -d 2>/dev/null || find . -type d -maxdepth 2 | grep -v ".git" | sort

