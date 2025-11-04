#!/usr/bin/env python3
"""
Quick test script to verify GPU and download a real model.
Run this inside ROCm container to test everything works.
"""
import torch
import sys

def test_gpu():
    """Test GPU access."""
    print("=" * 50)
    print("GPU Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   ROCm: {torch.version.hip if hasattr(torch.version, 'hip') else 'Unknown'}")
    return True


def download_stable_diffusion():
    """Download and test Stable Diffusion 2.1."""
    print("\n" + "=" * 50)
    print("Downloading Stable Diffusion 2.1")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        print("Downloading model (this may take a few minutes)...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.bfloat16
        )
        pipe = pipe.to("cuda")
        
        print("✅ Model loaded successfully")
        print("   Generating test image...")
        
        image = pipe("a red apple on a white table", num_inference_steps=20).images[0]
        image.save("test_output.png")
        
        print("✅ Test image saved to test_output.png")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_qwen():
    """Download and test Qwen 2.5 3B."""
    print("\n" + "=" * 50)
    print("Downloading Qwen 2.5 3B")
    print("=" * 50)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Downloading model (this may take a few minutes)...")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print("✅ Model loaded successfully")
        print("   Testing generation...")
        
        prompt = "What is machine learning? Answer in one sentence:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("AMD MI300X Model Test Script\n")
    
    # Test GPU
    if not test_gpu():
        print("\n❌ GPU not available. Exiting.")
        sys.exit(1)
    
    # Ask which model to test
    print("\nWhich model to download and test?")
    print("1. Stable Diffusion 2.1 (text-to-image)")
    print("2. Qwen 2.5 3B (language model)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    results = []
    
    if choice in ["1", "3"]:
        results.append(("Stable Diffusion", download_stable_diffusion()))
    
    if choice in ["2", "3"]:
        results.append(("Qwen", download_qwen()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{name}: {status}")


