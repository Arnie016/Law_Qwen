#!/usr/bin/env python3
"""
Quick script to download and explore Hugging Face datasets
Usage: python3 download_dataset.py wikitext wikitext-2-raw-v1
"""
import sys
from datasets import load_dataset

if len(sys.argv) < 2:
    print("Usage: python3 download_dataset.py <dataset_name> [config]")
    print("\nPopular datasets:")
    print("  wikitext wikitext-2-raw-v1")
    print("  openai/gsm8k")
    print("  allenai/c4 en")
    print("  wyu1/Leopard-Instruct")
    sys.exit(1)

dataset_name = sys.argv[1]
config = sys.argv[2] if len(sys.argv) > 2 else None

print(f"Downloading {dataset_name}...")

try:
    if config:
        dataset = load_dataset(dataset_name, config)
    else:
        dataset = load_dataset(dataset_name)
    
    print(f"\n✅ Dataset downloaded!")
    print(f"\nDataset structure:")
    print(dataset)
    
    print(f"\nSplits: {list(dataset.keys())}")
    for split_name in dataset.keys():
        print(f"  {split_name}: {len(dataset[split_name])} examples")
    
    if len(dataset[list(dataset.keys())[0]]) > 0:
        print(f"\nSample from {list(dataset.keys())[0]}:")
        sample = dataset[list(dataset.keys())[0]][0]
        if isinstance(sample, dict):
            for key, value in sample.items():
                if isinstance(value, str):
                    print(f"  {key}: {value[:200]}...")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {sample}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
