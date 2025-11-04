#!/usr/bin/env python3
"""
Quick test script for DeepSeek-OCR
Test the model on your server before running the API.
"""
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

def test_deepseek_ocr():
    """Test DeepSeek-OCR model."""
    print("=" * 50)
    print("DeepSeek-OCR Quick Test")
    print("=" * 50)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\nDownloading DeepSeek-OCR model (this may take a few minutes)...")
    model_name = "deepseek-ai/DeepSeek-OCR"
    
    try:
        # Load processor and model
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto"
        )
        
        model.eval()
        print("✅ Model loaded successfully")
        
        # Create a test image (or use a real image file)
        print("\nCreating test image...")
        # Create a simple test image with text
        from PIL import ImageDraw, ImageFont
        
        # Create a blank image
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((10, 10), "Hello DeepSeek-OCR!", fill='black', font=font)
        draw.text((10, 50), "This is a test image.", fill='black', font=font)
        draw.text((10, 90), "Testing OCR capabilities.", fill='black', font=font)
        
        # Save test image
        img.save("test_image.png")
        print("✅ Test image saved to test_image.png")
        
        # Run OCR
        print("\nRunning OCR...")
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            result = processor.decode(outputs[0], skip_special_tokens=True)
        
        print("\n" + "=" * 50)
        print("OCR Result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deepseek_ocr()


