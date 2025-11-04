# Legal LLM Landscape: Reasoning vs Knowledge

## Models Already Doing Legal Fine-Tuning

### 1. Unilaw-R1 (7B) ⭐
**What:** Legal reasoning model with both knowledge AND reasoning
- **Knowledge:** 17,000 chain-of-thought legal samples
- **Reasoning:** Two-stage training (SFT + RL)
- **Result:** Outperforms similar-size models, matches DeepSeek-R1-Distill-Qwen-32B
- **Key:** Solves "insufficient legal knowledge" AND "unreliable reasoning"

### 2. ChatLaw
**What:** Legal LLM with external knowledge bases
- **Knowledge:** Integrated legal databases
- **Reasoning:** Uses RAG (Retrieval-Augmented Generation)
- **Key:** Combines model reasoning with external legal facts

### 3. Legal-BERT / Legal-RoBERTa
**What:** Legal domain adaptation
- **Knowledge:** Trained on legal corpus
- **Reasoning:** Standard transformer reasoning
- **Key:** Domain knowledge, not specialized reasoning

## The Problem You Identified

**Reasoning ≠ Knowledge:**

```
Strong Reasoning Model (O3, GPT-OSS):
├─ Can reason: "If A then B, therefore C" ✅
├─ Can solve logic puzzles ✅
└─ But: Doesn't know UCC § 2-205 ❌

Legal Knowledge Model:
├─ Knows: UCC § 2-205 details ✅
├─ Knows: Case citations ✅
└─ But: May not reason well about them ❌

Best Legal Model (Unilaw-R1):
├─ Knows: Legal facts ✅
├─ Reasons: Step-by-step legal analysis ✅
└─ Result: Both knowledge AND reasoning ✅
```

## What This Means for Your Project

**Current Approach:**
- Qwen 2.5 32B (general reasoning) + Legal dataset fine-tuning
- Problem: May learn legal format but not deep legal knowledge

**What Works (Based on Unilaw-R1):**
1. **High-Quality Legal Dataset**
   - Not just raw legal text
   - Chain-of-thought examples (17K samples)
   - Shows HOW to reason about legal problems

2. **Two-Stage Training**
   - Stage 1: SFT on legal knowledge
   - Stage 2: RL on legal reasoning quality

3. **Legal-Specific Formats**
   - IRAC (Issue, Rule, Application, Conclusion)
   - Case citations
   - Statute references

## Your Current Setup vs Best Practice

**Your Setup:**
```
Qwen 2.5 32B (general reasoning)
+ pile_of_law_subset (raw legal text)
+ 500 SFT steps (too few)
+ GRPO (optimizing format, not knowledge)
= Weak legal knowledge, decent formatting
```

**Best Practice (Unilaw-R1 approach):**
```
Base Model (general reasoning)
+ High-quality legal CoT dataset (17K samples)
+ 10K+ SFT steps (build knowledge)
+ RL on legal reasoning (improve quality)
= Strong legal knowledge + strong reasoning
```

## Recommendation

**Don't use reasoning models alone.** Instead:

1. **Focus on Legal Knowledge First:**
   - Full Pile of Law (millions of examples)
   - LegalBench Q&A format
   - Chain-of-thought legal reasoning examples

2. **Then Improve Reasoning:**
   - GRPO (what you're doing now)
   - But with better base knowledge

3. **Consider Legal-Specific Models:**
   - Start from Unilaw-R1 if available
   - Or ChatLaw with RAG

**Your insight is correct:** Better reasoning doesn't mean better legal knowledge. You need BOTH.

