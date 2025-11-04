#!/usr/bin/env python3
"""
Check transformers version and try alternative DeepSeek-OCR loading
"""
import sys

print("Checking transformers version...")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers not installed")
    sys.exit(1)

# Check if LlamaFlashAttention2 exists
print("\nChecking if LlamaFlashAttention2 is available...")
try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    print("✅ LlamaFlashAttention2 found")
except ImportError:
    print("❌ LlamaFlashAttention2 not found")
    print("\nTrying to upgrade transformers...")
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("✅ Transformers upgraded, please restart Python")
    else:
        print("❌ Upgrade failed")
        print("\nAlternative: Try installing transformers from source or use a different OCR model")


