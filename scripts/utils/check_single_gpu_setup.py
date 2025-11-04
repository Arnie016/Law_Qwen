#!/usr/bin/env python3
"""
Quick GPU Check & Single GPU Setup
Copy-paste this into your SSH session
"""
import torch
import sys

print("=" * 60)
print("GPU CHECK FOR SINGLE GPU TRAINING")
print("=" * 60)

# Check GPU availability
print(f"\n✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"✅ GPU count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"\n🎮 GPU {i}:")
        print(f"   Name: {props.name}")
        print(f"   Memory: {memory_gb:.1f} GB")
    
    print("\n" + "=" * 60)
    print("SINGLE GPU CONFIGURATION")
    print("=" * 60)
    
    if gpu_count == 1:
        print("✅ PERFECT! You have exactly 1 GPU.")
        print("✅ Use device_map='cuda:0' in training script")
    elif gpu_count > 1:
        print(f"⚠️  You have {gpu_count} GPUs available.")
        print("✅ Will use GPU 0 only (device_map='cuda:0')")
        print("✅ Other GPUs will be ignored")
    else:
        print("❌ No GPUs detected!")
        sys.exit(1)
    
    # Test GPU memory
    print("\n" + "=" * 60)
    print("TESTING GPU MEMORY")
    print("=" * 60)
    try:
        x = torch.randn(1000, 1000, device='cuda:0', dtype=torch.bfloat16)
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"✅ GPU 0 memory test successful!")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
    
    # Model size estimate
    print("\n" + "=" * 60)
    print("MODEL SIZE ESTIMATE")
    print("=" * 60)
    print("Qwen 2.5 32B:")
    print("   Full precision: ~62 GB")
    print("   BFloat16: ~31 GB")
    print("   LoRA adapters: ~400 MB")
    print("   Training overhead: ~5-10 GB")
    print(f"   Total needed: ~35-40 GB")
    print(f"   Available: {props.total_memory / 1e9:.1f} GB")
    
    if memory_gb >= 40:
        print("✅ SUFFICIENT MEMORY for single GPU training!")
    else:
        print("⚠️  May need to reduce batch size or use gradient checkpointing")
    
    print("\n" + "=" * 60)
    print("READY FOR SINGLE GPU TRAINING")
    print("=" * 60)
    print("Use: scripts/training/finetune_qwen_law_single_gpu.py")
    print("Or modify existing script: device_map='cuda:0'")
    
else:
    print("❌ CUDA not available - check ROCm installation")
    sys.exit(1)

