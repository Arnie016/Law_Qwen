# DeepSeek-OCR FastAPI Service

OCR service with bounding box output, optimized for AMD MI300X/ROCm.

## Quick Start

### Inside ROCm Docker Container

```bash
# Enter container
docker exec -it rocm /bin/bash

# Navigate to project
cd /path/to/deepseek_ocr

# Create venv (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run service
python3 app.py
```

Service runs on `http://0.0.0.0:8000`

### Using Docker

```bash
# Build
docker build -t deepseek-ocr .

# Run (map port 8000)
docker run --rm -it --device=/dev/dri --device=/dev/kfd -p 8000:8000 deepseek-ocr
```

## API Endpoints

### `GET /`
Health check

### `GET /health`
Model status

### `POST /ocr`
Upload image, get OCR results with bounding boxes.

**Request:**
```bash
curl -X POST "http://localhost:8000/ocr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"
```

**Response:**
```json
{
  "text": "extracted text",
  "boxes": [
    {
      "x_min": 0.0,
      "y_min": 0.0,
      "x_max": 100.0,
      "y_max": 50.0,
      "text": "word",
      "confidence": 0.95
    }
  ],
  "full_text": "full extracted text"
}
```

## Model Info

- **Model:** `deepseek-ai/DeepSeek-OCR`
- **Downloads:** 2.1M+ | **Likes:** 2,424
- **Parameters:** 3.3B
- **Task:** image-text-to-text (OCR)
- **Link:** https://hf.co/deepseek-ai/DeepSeek-OCR
- **License:** MIT

## Quick Test

Test the model before running the API:

```bash
python3 quick_test.py
```

This will:
1. Download the model (~7GB)
2. Create a test image
3. Run OCR on it
4. Show the results

## Notes

- Model loads on first request (lazy loading)
- Uses bfloat16 on GPU for efficiency
- ROCm-compatible PyTorch operations
- DeepSeek-OCR is a vision-language model (multimodal)
- Supports multilingual OCR

