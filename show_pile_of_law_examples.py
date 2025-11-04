#!/usr/bin/env python3
"""
Show examples from Pile of Law dataset
Run locally: python3 show_pile_of_law_examples.py
"""
from datasets import load_dataset

print("Loading Pile of Law subset...")
print("(This may take a minute to download if first time)")
dataset = load_dataset("lamblamb/pile_of_law_subset")

print(f"\n✅ Dataset loaded!")
print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset.get('validation', []))}")

print("\n" + "="*80)
print("EXAMPLE 1: Court Opinion/Case Law")
print("="*80)
example1 = dataset["train"][0]
print(f"\nText (first 1000 chars):")
print(example1["text"][:1000])
print("...")

print("\n" + "="*80)
print("EXAMPLE 2: Legal Statute/Regulation")
print("="*80)
example2 = dataset["train"][500]
print(f"\nText (first 1000 chars):")
print(example2["text"][:1000])
print("...")

print("\n" + "="*80)
print("EXAMPLE 3: Contract Language")
print("="*80)
example3 = dataset["train"][1000]
print(f"\nText (first 1000 chars):")
print(example3["text"][:1000])
print("...")

print("\n" + "="*80)
print("EXAMPLE 4: Legal Brief/Filing")
print("="*80)
example4 = dataset["train"][5000]
print(f"\nText (first 1000 chars):")
print(example4["text"][:1000])
print("...")

print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)
# Check text lengths
lengths = [len(ex["text"]) for ex in dataset["train"][:1000]]
print(f"Average text length: {sum(lengths)/len(lengths):.0f} characters")
print(f"Min length: {min(lengths)} characters")
print(f"Max length: {max(lengths)} characters")

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

