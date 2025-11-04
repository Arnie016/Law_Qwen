#!/usr/bin/env python3
"""
Download models to server with custom cache location.
Usage: python3 download_model.py Qwen/Qwen2.5-14B-Instruct
"""
import sys
import os
from pathlib import Path

# Set custom cache directory
cache_dir = os.path.expanduser("~/workspace/models")
os.makedirs(cache_dir, exist_ok=True)

# Set environment variables
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir

if len(sys.argv) < 2:
    print("Usage: python3 download_model.py <model_name>")
    print("Example: python3 download_model.py Qwen/Qwen2.5-14B-Instruct")
    sys.exit(1)

model_name = sys.argv[1]
print(f"Downloading {model_name} to {cache_dir}...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    print("Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        cache_dir=cache_dir,
        device_map="auto"
    )
    
    print(f"✅ Model downloaded successfully to {cache_dir}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)


