# Layout Analysis Pipeline

OCR → LayoutLMv3 → LLM reasoning pipeline for document analysis.

## Quick Start

```bash
# Inside ROCm container
docker exec -it rocm /bin/bash

cd /path/to/layout_pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 app.py
```

Service runs on `http://0.0.0.0:8001`

## API

### `POST /analyze`

Upload document image, get:
- Extracted text
- Bounding boxes with labels
- Layout structure (paragraphs, tables, titles)
- LLM-generated insights

**Example:**
```bash
curl -X POST "http://localhost:8001/analyze" \
  -F "file=@document.png"
```

**Response:**
```json
{
  "text": "full text...",
  "boxes": [...],
  "layout_structure": {
    "reading_order": [...],
    "paragraphs": [...],
    "tables": [...],
    "titles": [...]
  },
  "insights": [
    {
      "type": "summary",
      "content": "...",
      "confidence": 0.8
    }
  ]
}
```

## Models

- OCR: Tesseract (or DeepSeek-OCR)
- Layout: LayoutLMv3-base
- LLM: Phi-2 or Qwen-2-1.5B

Adjust model names in `app.py` based on availability/preference.


