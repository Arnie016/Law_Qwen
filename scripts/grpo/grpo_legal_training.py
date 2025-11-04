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
checkpoint_loaded = False
try:
    from peft import PeftModel
    import os
    if os.path.exists(CHECKPOINT_PATH):
        model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
        print("   ✅ Loaded checkpoint-500 LoRA adapters")
        checkpoint_loaded = True
    else:
        print(f"   ⚠️  Checkpoint not found at {CHECKPOINT_PATH}")
        print("   Starting fresh GRPO training")
except Exception as e:
    print(f"   ⚠️  Could not load checkpoint: {e}")
    print("   Starting fresh GRPO training")

# Add LoRA (only if checkpoint not loaded)
if not checkpoint_loaded:
    print("   Adding new LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
    )
else:
    print("   Using LoRA adapters from checkpoint-500")

print("✅ Model loaded")

# Improved legal reasoning reward function
def legal_reward_function(response, question):
    """
    Enhanced reward function for legal reasoning quality
    Higher reward for:
    - Legal terminology and precision
    - Structured reasoning (IRAC format)
    - Citation of statutes/cases
    - Complete, comprehensive answers
    - Legal analysis depth
    """
    import re
    reward = 0.0
    
    # ===== COMPLETENESS (0-3 points) =====
    response_len = len(response.strip())
    if response_len > 500:
        reward += 3.0  # Comprehensive answer
    elif response_len > 300:
        reward += 2.0  # Good length
    elif response_len > 100:
        reward += 1.0  # Minimum acceptable
    elif response_len < 50:
        reward -= 2.0  # Too short
    
    # ===== LEGAL TERMINOLOGY (0-4 points) =====
    legal_terms = [
        # Core legal concepts
        'statute', 'case', 'court', 'ruling', 'precedent', 'doctrine',
        'law', 'legal', 'constitutional', 'common law', 'statutory',
        # Specific legal areas
        'UCC', 'FSIA', 'ABA', 'Model Rules', 'negligence', 'liability', 
        'immunity', 'jurisdiction', 'standing', 'causation', 'damages',
        # Legal procedures
        'motion', 'pleading', 'discovery', 'settlement', 'appeal',
        # Legal reasoning
        'analysis', 'reasoning', 'conclusion', 'holding', 'dicta'
    ]
    found_terms = sum(1 for term in legal_terms if term.lower() in response.lower())
    reward += min(found_terms * 0.3, 4.0)  # Cap at 4 points
    
    # ===== STRUCTURED REASONING (0-3 points) =====
    # Check for IRAC format (Issue, Rule, Application, Conclusion)
    irac_indicators = ['issue', 'rule', 'application', 'conclusion', 'analysis']
    irac_count = sum(1 for word in irac_indicators if word.lower() in response.lower())
    if irac_count >= 3:
        reward += 3.0  # Well-structured IRAC
    elif irac_count >= 2:
        reward += 1.5  # Some structure
    
    # Check for paragraphs (structured writing)
    if response.count('\n\n') >= 2:
        reward += 1.0  # Well-paragraphed
    elif '\n\n' in response:
        reward += 0.5
    
    # Check for lists/numbering (organized)
    if re.search(r'\d+\.\s', response) or re.search(r'[•\-\*]\s', response):
        reward += 0.5  # Organized with lists
    
    # ===== CITATIONS (0-3 points) =====
    # Statute citations (§, UCC, etc.)
    statute_patterns = [
        r'§\s*\d+',           # § 2-205
        r'UCC\s*§?\s*\d+',    # UCC § 2-205
        r'\d+-\d+',           # 2-205
        r'Section\s+\d+',     # Section 205
    ]
    has_statute = any(re.search(pattern, response, re.I) for pattern in statute_patterns)
    if has_statute:
        reward += 2.0  # Strong citation
    
    # Case citations (v., Case, etc.)
    case_patterns = [
        r'\w+\s+v\.\s+\w+',   # Smith v. Jones
        r'Case\s+[A-Z]',       # Case references
        r'[A-Z][a-z]+\s+v\.', # Case names
    ]
    has_case = any(re.search(pattern, response, re.I) for pattern in case_patterns)
    if has_case:
        reward += 1.0
    
    # ===== LEGAL ANALYSIS DEPTH (0-3 points) =====
    # Check for legal analysis keywords
    analysis_keywords = [
        'because', 'therefore', 'however', 'furthermore', 'consequently',
        'applies', 'distinguished', 'analogous', 'similar', 'different',
        'requires', 'mandates', 'prohibits', 'allows', 'permits'
    ]
    analysis_count = sum(1 for word in analysis_keywords if word.lower() in response.lower())
    reward += min(analysis_count * 0.2, 3.0)
    
    # Check for comparative analysis
    if re.search(r'(compare|distinguish|similar|different)', response, re.I):
        reward += 1.0  # Comparative analysis
    
    # Check for counterarguments
    if re.search(r'(however|although|despite|nevertheless|but)', response, re.I):
        reward += 0.5  # Shows nuanced thinking
    
    # ===== QUESTION-SPECIFIC REWARDS (0-2 points) =====
    # Reward for answering the specific question asked
    question_keywords = re.findall(r'\b\w+\b', question.lower())
    question_keywords = [w for w in question_keywords if len(w) > 4]  # Substantive words
    matched_keywords = sum(1 for kw in question_keywords if kw in response.lower())
    if matched_keywords >= 3:
        reward += 2.0  # Addresses question well
    elif matched_keywords >= 1:
        reward += 1.0
    
    # ===== PENALTIES =====
    # Penalize very short or non-legal responses
    if response_len < 20:
        reward -= 3.0  # Too short
    
    # Penalize if no legal terminology at all
    if found_terms == 0:
        reward -= 1.0
    
    # Penalize generic responses
    generic_phrases = ['i think', 'i believe', 'maybe', 'probably', 'i guess']
    if sum(1 for phrase in generic_phrases if phrase in response.lower()) > 2:
        reward -= 1.0  # Too generic
    
    # Penalize repetitive responses
    words = response.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            reward -= 1.0  # Too repetitive
    
    # ===== NORMALIZE REWARD =====
    # Cap reward between -5 and +20 (reasonable range)
    reward = max(-5.0, min(20.0, reward))
    
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
# Note: per_device_train_batch_size must be multiple of num_generations
# Unsloth will auto-adjust batch_size=2 to batch_size=4 (multiple of num_generations=4)
grpo_config = GRPOConfig(
    output_dir="./qwen2.5-32b-law-grpo",
    per_device_train_batch_size=4,  # Must be multiple of num_generations (4)
    gradient_accumulation_steps=2,  # Reduced to compensate
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
    # Note: reward_fn goes in GRPOTrainer, NOT GRPOConfig
)

# Reward function wrapper (GRPO will call with responses and prompts)
def grpo_reward_fn(*args, **kwargs):
    """
    GRPO reward function - called automatically by GRPOTrainer
    Handles any calling convention from Unsloth (positional, keyword, or mixed)
    
    Unsloth may call it as:
    - reward_func(prompts, responses)
    - reward_func(inputs, completions)
    - reward_func(prompts=..., responses=...)
    - reward_func(inputs=..., completions=...)
    """
    # Extract prompts/inputs from args or kwargs
    prompts = None
    responses = None
    
    # Try positional args first
    if len(args) >= 1:
        prompts = args[0]
    if len(args) >= 2:
        responses = args[1]
    
    # Try kwargs (handle both naming conventions)
    if prompts is None:
        prompts = kwargs.get('prompts') or kwargs.get('inputs') or kwargs.get('original_prompts')
    if responses is None:
        responses = kwargs.get('responses') or kwargs.get('completions')
    
    # Fallback: if still None, try args[0] and args[1] again
    if prompts is None and len(args) > 0:
        prompts = args[0]
    if responses is None and len(args) > 1:
        responses = args[1]
    
    # Ensure we have lists
    if prompts is None:
        prompts = []
    if responses is None:
        responses = []
    
    if not isinstance(prompts, list):
        prompts = [prompts] if prompts else []
    if not isinstance(responses, list):
        responses = [responses] if responses else []
    
    # Safety check: must have same length
    if len(prompts) != len(responses):
        # If lengths don't match, try to pair them properly
        min_len = min(len(prompts), len(responses))
        prompts = prompts[:min_len]
        responses = responses[:min_len]
    
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
# Unsloth expects reward_funcs (plural) as a LIST of functions
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=grpo_dataset["train"],
    tokenizer=tokenizer,
    reward_funcs=[grpo_reward_fn],  # Must be a list, even if single function
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

