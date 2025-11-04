#!/usr/bin/env python3
"""
Show examples from Pile of Law dataset using streaming (no full download)
"""
from datasets import load_dataset

print("Loading Pile of Law subset with streaming...")
print("(Only loads examples, doesn't download full dataset)")

# Use streaming to avoid downloading everything
dataset = load_dataset("lamblamb/pile_of_law_subset", streaming=True)

print("\n✅ Dataset loaded (streaming mode)")
print("\n" + "="*80)
print("EXAMPLE 1: Court Opinion/Case Law")
print("="*80)

# Get first example
example1 = next(iter(dataset["train"]))
print(f"\nText (first 1000 chars):")
print(example1["text"][:1000])
print("...")

print("\n" + "="*80)
print("EXAMPLE 2: Legal Statute/Regulation")
print("="*80)

# Skip some examples to get different types
it = iter(dataset["train"])
for _ in range(500):
    next(it)
example2 = next(it)
print(f"\nText (first 1000 chars):")
print(example2["text"][:1000])
print("...")

print("\n" + "="*80)
print("EXAMPLE 3: Contract Language")
print("="*80)

# Skip more
it = iter(dataset["train"])
for _ in range(1000):
    next(it)
example3 = next(it)
print(f"\nText (first 1000 chars):")
print(example3["text"][:1000])
print("...")

print("\n" + "="*80)
print("EXAMPLE 4: Legal Brief/Filing")
print("="*80)

# Skip more
it = iter(dataset["train"])
for _ in range(5000):
    next(it)
example4 = next(it)
print(f"\nText (first 1000 chars):")
print(example4["text"][:1000])
print("...")

print("\n" + "="*80)
print("DATASET SOURCES (from Pile of Law)")
print("="*80)
print("""
The Pile of Law contains:
- Court opinions (federal and state cases)
- Statutes (U.S. Code, Code of Federal Regulations)
- Legal contracts
- SEC filings (EDGAR)
- Patent documents
- Legal briefs and filings
- Law review articles
- State regulations
- Case law citations
- Legal precedents
""")

print("\n" + "="*80)
print("CLEANUP: Free up disk space")
print("="*80)
print("""
To free up space from partial downloads:
1. Clear Hugging Face cache:
   rm -rf ~/.cache/huggingface/datasets/lamblamb___pile_of_law_subset
   
2. Check disk space:
   df -h
""")

