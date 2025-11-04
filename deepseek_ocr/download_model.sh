#!/bin/bash
# Download DeepSeek-OCR model on AMD DevCloud server
# Run inside ROCm container: docker exec -it rocm /bin/bash

echo "=========================================="
echo "Downloading DeepSeek-OCR Model"
echo "=========================================="

# Check if we're in ROCm container
if ! python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "Error: PyTorch not installed. Install first with: pip install torch"
    exit 1
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip install --upgrade transformers>=4.40.0 --quiet
pip install accelerate pillow addict matplotlib einops easydict --quiet
# Try flash-attn, but don't fail if it doesn't install
pip install flash-attn --no-build-isolation --quiet || echo "Flash-attn install skipped (optional)"

# Download and test the model
echo ""
echo "Downloading DeepSeek-OCR model (this may take a few minutes, ~7GB)..."
echo ""

python3 << EOF
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import os

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

model_name = "deepseek-ai/DeepSeek-OCR"
print(f"\nDownloading {model_name}...")

try:
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Processor loaded")
    
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    print("✅ Model loaded successfully")
    
    # Create a test image
    print("\nCreating test image...")
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((10, 10), "Hello DeepSeek-OCR!", fill='black', font=font)
    draw.text((10, 50), "This is a test image.", fill='black', font=font)
    draw.text((10, 90), "Testing OCR capabilities.", fill='black', font=font)
    
    img.save("test_image.png")
    print("✅ Test image saved to test_image.png")
    
    # Run OCR
    print("\nRunning OCR test...")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        result = processor.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "=" * 50)
    print("OCR Result:")
    print("=" * 50)
    print(result)
    print("=" * 50)
    
    print("\n✅ DeepSeek-OCR downloaded and tested successfully!")
    print(f"Model location: ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run the API: python3 app.py"
echo "  2. Or test with: python3 quick_test.py"

