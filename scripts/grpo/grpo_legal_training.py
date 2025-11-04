#!/usr/bin/env python3
"""
GRPO Training for Legal Reasoning Improvement
Train on checkpoint-500 (500-step SFT) with legal reasoning rewards
"""
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import os

print("=" * 60)
print("GRPO Legal Reasoning Training")
print("=" * 60)

# Check GPU
print(f"\nGPU Info:")
print(f"  Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Model path (from 500-step SFT)
CHECKPOINT_PATH = "./qwen2.5-32b-law-finetuned/checkpoint-500"
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

print(f"\n1. Loading model from checkpoint: {CHECKPOINT_PATH}")

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit=False,
)

# Load LoRA adapters from checkpoint-500
print(f"   Loading LoRA adapters from {CHECKPOINT_PATH}...")
try:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
    print("   ✅ Loaded checkpoint-500 LoRA adapters")
except Exception as e:
    print(f"   ⚠️  Could not load checkpoint: {e}")
    print("   Starting fresh GRPO training")

# Add LoRA (if not already added)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

print("✅ Model loaded")

# Legal reasoning reward function
def legal_reward_function(response, question):
    """
    Reward function for legal reasoning quality
    Higher reward for:
    - Legal terminology
    - Structured reasoning
    - Citation of statutes/cases
    - Complete answers
    """
    reward = 0.0
    
    # Length check (complete answers)
    if len(response) > 100:
        reward += 1.0
    elif len(response) > 50:
        reward += 0.5
    
    # Legal terminology
    legal_terms = [
        'statute', 'case', 'court', 'ruling', 'precedent', 'doctrine',
        'law', 'legal', 'constitutional', 'common law', 'UCC', 'FSIA',
        'ABA', 'Model Rules', 'negligence', 'liability', 'immunity'
    ]
    found_terms = sum(1 for term in legal_terms if term.lower() in response.lower())
    reward += found_terms * 0.2
    
    # Structured reasoning (has paragraphs or lists)
    if '\n\n' in response or len(response.split('.')) > 3:
        reward += 1.0
    
    # Citation patterns (statutes, cases)
    import re
    if re.search(r'§\s*\d+', response) or re.search(r'\d+-\d+', response):
        reward += 1.0  # Statute citation
    if re.search(r'v\.', response) or re.search(r'case', response, re.I):
        reward += 0.5  # Case citation
    
    # Penalize very short or non-legal responses
    if len(response) < 20:
        reward -= 1.0
    if 'legal' not in response.lower() and 'law' not in response.lower():
        reward -= 0.5
    
    return reward

# Dataset: Use LegalBench or create Q&A pairs
print("\n2. Loading dataset...")

# Option 1: LegalBench (if available)
try:
    dataset = load_dataset("nguha/legalbench", "all", streaming=True)
    print("   ✅ Using LegalBench dataset")
except:
    # Option 2: Create Q&A pairs from legal text
    print("   Using law dataset to create Q&A pairs...")
    dataset = load_dataset("lamblamb/pile_of_law_subset")
    
    # Format as Q&A pairs
    def format_qa(examples):
        questions = []
        for text in examples['text']:
            # Extract question from text (simplified)
            question = f"Analyze this legal text: {text[:300]}"
            questions.append(question)
        return {"question": questions}
    
    dataset = dataset.map(format_qa, batched=True)
    print("   ✅ Dataset formatted")

# Format for GRPO
def format_grpo(examples):
    prompts = []
    for question in examples.get('question', examples.get('text', [])):
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)
    return {"prompt": prompts}

grpo_dataset = dataset.map(format_grpo, batched=True)

# GRPO Configuration
print("\n3. Setting up GRPO config...")
grpo_config = GRPOConfig(
    output_dir="./qwen2.5-32b-law-grpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_steps=1000,  # Fast RL training: 500-1000 steps
    warmup_steps=100,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    logging_steps=10,
    save_steps=100,
    # GRPO-specific
    num_generations=4,  # Generate 4 responses per prompt
    # Note: reward_fn goes in GRPOTrainer, not GRPOConfig
)

# Reward function wrapper (GRPO will call with responses and prompts)
def grpo_reward_fn(prompts, responses, **kwargs):
    """
    GRPO reward function - called automatically by GRPOTrainer
    Args:
        prompts: List of prompt strings
        responses: List of response strings (one per prompt)
    Returns:
        rewards: List of reward floats
    """
    rewards = []
    for prompt, response in zip(prompts, responses):
        # Extract question from prompt
        try:
            question = prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]
        except:
            question = prompt  # Fallback if format doesn't match
        reward = legal_reward_function(response, question)
        rewards.append(reward)
    return rewards

# GRPO Trainer
print("\n4. Starting GRPO training...")
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=grpo_dataset["train"],
    tokenizer=tokenizer,
    reward_fn=grpo_reward_fn,  # Reward function goes here
)

print(f"   Training for {grpo_config.max_steps} steps...")
print(f"   Expected time: ~{grpo_config.max_steps * 0.01:.1f} hours")

trainer.train()

# Save
print("\n5. Saving model...")
model.save_pretrained("./qwen2.5-32b-law-grpo")
print("✅ GRPO training complete!")

print("\n" + "=" * 60)
print("Next Steps:")
print("1. Evaluate: python3 scripts/evaluation/eval_legal_models_scientific.py")
print("2. Compare: Base vs SFT-500 vs SFT-500+GRPO")
print("=" * 60)

