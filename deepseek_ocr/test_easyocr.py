#!/usr/bin/env python3
"""
Test EasyOCR - creates test image first
"""
from PIL import Image, ImageDraw, ImageFont
import easyocr

print("Creating test image...")
# Create a test image
img = Image.new('RGB', (400, 200), color='white')
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.load_default()
except:
    font = None

draw.text((10, 10), "Hello EasyOCR!", fill='black', font=font)
draw.text((10, 50), "This is a test image.", fill='black', font=font)
draw.text((10, 90), "Testing OCR capabilities.", fill='black', font=font)
draw.text((10, 130), "EasyOCR works!", fill='black', font=font)

img.save("test_image.png")
print("✅ Test image saved to test_image.png")

print("\nInitializing EasyOCR (this downloads models on first run)...")
reader = easyocr.Reader(['en'])

print("\nRunning OCR...")
result = reader.readtext('test_image.png')

print("\n" + "=" * 50)
print("OCR Results:")
print("=" * 50)
for detection in result:
    bbox, text, confidence = detection
    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Bounding box: {bbox}")
    print("-" * 50)

print("\n✅ EasyOCR test completed!")


