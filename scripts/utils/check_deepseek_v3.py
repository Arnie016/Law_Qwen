#!/usr/bin/env python3
"""
Check if DeepSeek-V3 is installed/downloaded
"""
import os
from pathlib import Path

def check_deepseek_v3():
    """Check if DeepSeek-V3 is installed."""
    print("=" * 60)
    print("Checking for DeepSeek-V3 (671B MoE)")
    print("=" * 60)
    
    # Check Hugging Face cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    
    print("\n1. Checking Hugging Face cache...")
    deepseek_paths = []
    
    if hf_cache.exists():
        for item in hf_cache.iterdir():
            if "deepseek" in item.name.lower() and "v3" in item.name.lower():
                deepseek_paths.append(item)
                print(f"   ✅ Found: {item.name}")
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                print(f"      Size: {size / 1e12:.2f} TB")
    
    if not deepseek_paths:
        print("   ❌ Not found in Hugging Face cache")
    
    # Check common model names
    print("\n2. Checking for DeepSeek-V3 models...")
    possible_names = [
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-V3-Chat",
        "deepseek-ai/DeepSeek-V3-Base",
    ]
    
    for model_name in possible_names:
        # Check if model files exist
        model_path = hf_cache / f"models--{model_name.replace('/', '--')}"
        if model_path.exists():
            print(f"   ✅ Found: {model_name}")
            print(f"      Path: {model_path}")
        else:
            print(f"   ❌ Not found: {model_name}")
    
    # Try to import and check
    print("\n3. Checking transformers library...")
    try:
        from transformers import AutoModel, AutoTokenizer
        
        print("   ✅ Transformers available")
        
        # Try to load model (this will check if it's downloaded)
        print("\n4. Attempting to load DeepSeek-V3...")
        try:
            model_name = "deepseek-ai/DeepSeek-V3"
            print(f"   Trying: {model_name}")
            
            # Just check config, don't load full model
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            print(f"   ✅ DeepSeek-V3 found!")
            print(f"      Model type: {config.model_type}")
            print(f"      Architecture: {config.architectures}")
            
            # Check if model files exist
            model_path = hf_cache / f"models--{model_name.replace('/', '--')}"
            if model_path.exists():
                print(f"      Files location: {model_path}")
                
                # Check size
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                print(f"      Total size: {total_size / 1e12:.2f} TB")
                
                # List shard files
                shard_files = list(model_path.rglob("*.safetensors"))
                if shard_files:
                    print(f"      Model shards: {len(shard_files)} files")
                    
        except Exception as e:
            print(f"   ❌ Not loaded: {e}")
            print(f"      Model may not be downloaded yet")
            
    except ImportError:
        print("   ❌ Transformers not installed")
    
    # Check Ollama models
    print("\n5. Checking Ollama...")
    ollama_path = Path.home() / ".ollama" / "models"
    if ollama_path.exists():
        ollama_models = list(ollama_path.rglob("*deepseek*"))
        if ollama_models:
            print(f"   ✅ Found DeepSeek models in Ollama:")
            for model in ollama_models:
                print(f"      {model}")
        else:
            print("   ❌ No DeepSeek models in Ollama")
    else:
        print("   ℹ️  Ollama models directory not found")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("To download DeepSeek-V3:")
    print("  python3 << EOF")
    print("  from transformers import AutoModel, AutoTokenizer")
    print("  model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-V3', trust_remote_code=True)")
    print("  EOF")

if __name__ == "__main__":
    check_deepseek_v3()

