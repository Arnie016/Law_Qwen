#!/usr/bin/env python3
"""
Compare Base Model vs GRPO Fine-Tuned Model (checkpoint-1000)
Run on server to evaluate improvements
"""
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["DISABLE_BITSANDBYTES_AUTO_INSTALL"] = "1"

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import json
from datetime import datetime

print("=" * 60)
print("Base Model vs GRPO Fine-Tuned Comparison")
print("=" * 60)

# Legal test questions
TEST_QUESTIONS = [
    "When does an offer become irrevocable under UCC § 2-205, and how does this differ from common-law options?",
    "Explain the strict scrutiny test and give two cases where it was applied.",
    "Analyze whether an accomplice who withdraws before the crime occurs is still criminally liable.",
    "Compare negligence per se and res ipsa loquitur; when does each apply?",
    "Discuss the rule against perpetuities and its modern reforms.",
    "When can the corporate veil be pierced? Give examples.",
    "Differentiate between copyright and trademark fair-use doctrines.",
    "Explain state immunity under the FSIA and its exceptions.",
    "Outline the hearsay rule and three major exceptions.",
    "Under the ABA Model Rules, when must a lawyer withdraw from representation?",
]

# Legal reward function (same as training)
def legal_reward_function(response, question):
    import re
    reward = 0.0
    
    # Completeness
    response_len = len(response.strip())
    if response_len > 500:
        reward += 3.0
    elif response_len > 300:
        reward += 2.0
    elif response_len > 100:
        reward += 1.0
    elif response_len < 50:
        reward -= 2.0
    
    # Legal terminology
    legal_terms = [
        'statute', 'case', 'court', 'ruling', 'precedent', 'doctrine',
        'law', 'legal', 'constitutional', 'common law', 'statutory',
        'UCC', 'FSIA', 'ABA', 'Model Rules', 'negligence', 'liability',
        'immunity', 'jurisdiction', 'standing', 'causation', 'damages',
    ]
    found_terms = sum(1 for term in legal_terms if term.lower() in response.lower())
    reward += min(found_terms * 0.3, 4.0)
    
    # Citations
    if re.search(r'§|UCC|FSIA|ABA', response):
        reward += 2.0
    
    # Structure (IRAC)
    if re.search(r'(issue|rule|analysis|conclusion)', response, re.I):
        reward += 3.0
    
    # Analysis depth
    analysis_keywords = ['requires', 'mandates', 'prohibits', 'allows']
    analysis_count = sum(1 for word in analysis_keywords if word.lower() in response.lower())
    reward += min(analysis_count * 0.2, 3.0)
    
    # Question-specific
    question_keywords = re.findall(r'\b\w+\b', question.lower())
    question_keywords = [w for w in question_keywords if len(w) > 4]
    matched_keywords = sum(1 for kw in question_keywords if kw in response.lower())
    if matched_keywords >= 3:
        reward += 2.0
    elif matched_keywords >= 1:
        reward += 1.0
    
    # Penalties
    if response_len < 20:
        reward -= 3.0
    if found_terms == 0:
        reward -= 1.0
    
    reward = max(-5.0, min(20.0, reward))
    return reward

def generate_response(model, tokenizer, prompt):
    """Generate response from model"""
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant response
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1].strip()
    
    return response

print("\n1. Loading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-32B-Instruct",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
print("✅ Base model loaded")

print("\n2. Loading GRPO fine-tuned model (checkpoint-1000)...")
checkpoint_path = "/root/scripts/grpo/qwen2.5-32b-law-grpo/checkpoint-1000"

if os.path.exists(checkpoint_path):
    grpo_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    print("✅ GRPO model loaded")
else:
    print(f"⚠️  Checkpoint not found at {checkpoint_path}")
    print("   Using base model for comparison")
    grpo_model = base_model

grpo_tokenizer = base_tokenizer  # Same tokenizer

print("\n3. Testing models on legal questions...")
print("=" * 60)

results = []

for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"\n📝 Question {i}/{len(TEST_QUESTIONS)}:")
    print(f"   {question[:80]}...")
    
    # Base model
    print("   [Base Model] Generating...")
    base_response = generate_response(base_model, base_tokenizer, question)
    base_reward = legal_reward_function(base_response, question)
    
    # GRPO model
    print("   [GRPO Model] Generating...")
    grpo_response = generate_response(grpo_model, grpo_tokenizer, question)
    grpo_reward = legal_reward_function(grpo_response, question)
    
    # Compare
    improvement = grpo_reward - base_reward
    
    results.append({
        "question": question,
        "base_reward": base_reward,
        "grpo_reward": grpo_reward,
        "improvement": improvement,
        "base_length": len(base_response),
        "grpo_length": len(grpo_response),
    })
    
    print(f"   ✅ Base: {base_reward:.2f} | GRPO: {grpo_reward:.2f} | Δ: {improvement:+.2f}")

print("\n" + "=" * 60)
print("4. Summary Statistics")
print("=" * 60)

base_avg = sum(r["base_reward"] for r in results) / len(results)
grpo_avg = sum(r["grpo_reward"] for r in results) / len(results)
improvement_avg = grpo_avg - base_avg

base_wins = sum(1 for r in results if r["grpo_reward"] > r["base_reward"])
grpo_wins = sum(1 for r in results if r["grpo_reward"] > r["base_reward"])
ties = sum(1 for r in results if r["grpo_reward"] == r["base_reward"])

print(f"\n📊 Average Rewards:")
print(f"   Base Model:  {base_avg:.2f} / 20")
print(f"   GRPO Model:  {grpo_avg:.2f} / 20")
print(f"   Improvement: {improvement_avg:+.2f} points ({improvement_avg/base_avg*100:+.1f}%)")

print(f"\n🏆 Wins:")
print(f"   GRPO better: {grpo_wins} / {len(results)}")
print(f"   Base better: {base_wins} / {len(results)}")
print(f"   Ties:        {ties} / {len(results)}")

print(f"\n📏 Length Comparison:")
base_len_avg = sum(r["base_length"] for r in results) / len(results)
grpo_len_avg = sum(r["grpo_length"] for r in results) / len(results)
print(f"   Base avg length:  {base_len_avg:.0f} chars")
print(f"   GRPO avg length:  {grpo_len_avg:.0f} chars")
print(f"   Difference:       {grpo_len_avg - base_len_avg:+.0f} chars")

# Save results
output_file = "/root/scripts/grpo/comparison_results.json"
with open(output_file, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "base_avg": base_avg,
        "grpo_avg": grpo_avg,
        "improvement": improvement_avg,
        "results": results,
    }, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")

print("\n" + "=" * 60)
print("✅ Comparison Complete!")
print("=" * 60)

if improvement_avg > 0:
    print(f"\n🎉 GRPO training improved model by {improvement_avg:.2f} points!")
else:
    print(f"\n⚠️  No significant improvement detected.")

