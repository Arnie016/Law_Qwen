#!/usr/bin/env python3
"""
Quick test script - Copy-paste and run anywhere!
Shows system info, GPU status, and training progress
"""
import os
import sys
from pathlib import Path

print("=" * 60)
print("QUICK SYSTEM CHECK")
print("=" * 60)

# Check Python version
print(f"\n🐍 Python: {sys.version.split()[0]}")
print(f"📍 Python path: {sys.executable}")

# Check current directory
print(f"\n📂 Current directory: {os.getcwd()}")

# Check if we're in Docker/container
if os.path.exists("/.dockerenv"):
    print("🐳 Running in Docker container")
else:
    print("🖥️  Running on host system")

# Check for GPU
try:
    import torch
    print(f"\n🔥 PyTorch version: {torch.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("\n⚠️  PyTorch not installed")

# Check for training checkpoints
print("\n" + "=" * 60)
print("TRAINING CHECKPOINTS")
print("=" * 60)

checkpoint_paths = [
    "/qwen2.5-32b-law-finetuned",
    "~/qwen2.5-32b-law-finetuned",
    "./qwen2.5-32b-law-finetuned",
]

found_checkpoints = False
for path in checkpoint_paths:
    expanded_path = Path(path).expanduser()
    if expanded_path.exists():
        print(f"\n✅ Found checkpoints at: {expanded_path}")
        checkpoints = sorted(expanded_path.glob("checkpoint-*"))
        if checkpoints:
            print(f"   Latest checkpoint: {checkpoints[-1].name}")
            print(f"   Total checkpoints: {len(checkpoints)}")
            found_checkpoints = True
        else:
            print("   (No checkpoint directories found)")
        break

if not found_checkpoints:
    print("\n⚠️  No training checkpoints found")

# Check disk space
print("\n" + "=" * 60)
print("DISK SPACE")
print("=" * 60)
try:
    import shutil
    total, used, free = shutil.disk_usage("/")
    print(f"📦 Total: {total // (1024**3)} GB")
    print(f"💾 Used: {used // (1024**3)} GB")
    print(f"💿 Free: {free // (1024**3)} GB")
except:
    pass

# Check environment
print("\n" + "=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"USER: {os.environ.get('USER', 'unknown')}")
print(f"HOSTNAME: {os.environ.get('HOSTNAME', 'unknown')}")
if 'CONDA_DEFAULT_ENV' in os.environ:
    print(f"CONDA ENV: {os.environ['CONDA_DEFAULT_ENV']}")

print("\n" + "=" * 60)
print("✅ CHECK COMPLETE!")
print("=" * 60)

