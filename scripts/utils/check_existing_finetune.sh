#!/bin/bash
# Check if fine-tuning was done before

echo "Checking for existing fine-tuned models..."

# Check the output directory
if [ -d "./qwen2.5-32b-law-finetuned" ]; then
    echo "✅ Found existing fine-tuned model directory!"
    echo ""
    echo "Contents:"
    ls -lh ./qwen2.5-32b-law-finetuned/
    echo ""
    echo "Size:"
    du -sh ./qwen2.5-32b-law-finetuned/
    echo ""
    echo "Model can be loaded with:"
    echo "from peft import PeftModel"
    echo "model = PeftModel.from_pretrained(base_model, './qwen2.5-32b-law-finetuned')"
else
    echo "❌ No existing fine-tuned model found"
    echo "This appears to be the first run"
fi

echo ""
echo "Current training status:"
echo "- Progress: 5/500 steps (1%)"
echo "- Time per step: ~24.6 seconds"
echo "- Estimated remaining: ~3 hours 23 minutes"

