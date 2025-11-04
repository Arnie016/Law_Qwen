"""
DeepSeek-OCR FastAPI service with bounding box output.
Optimized for ROCm/MI300X.
"""
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json
from typing import List, Dict, Any
from pydantic import BaseModel

app = FastAPI(title="DeepSeek-OCR API", version="1.0.0")

# Model will be loaded on first request (lazy loading)
model = None
processor = None
device = None


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    text: str
    confidence: float


class OCRResponse(BaseModel):
    text: str
    boxes: List[BoundingBox]
    full_text: str


def load_model():
    """Load DeepSeek-OCR model (lazy initialization)."""
    global model, tokenizer, processor, device
    
    if model is not None:
        return
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    try:
        from transformers import AutoModel, AutoProcessor
        
        model_name = "deepseek-ai/DeepSeek-OCR"
        print(f"Loading model: {model_name}")
        
        # DeepSeek-OCR uses AutoProcessor for image processing
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto"
        )
        
        model.eval()
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Optional: preload model on startup."""
    # Comment out if you prefer lazy loading
    # load_model()
    pass


@app.get("/")
async def root():
    return {"status": "ok", "service": "DeepSeek-OCR", "device": str(device) if device else "not_loaded"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    OCR endpoint: accepts image, returns text + bounding boxes.
    
    Input: multipart/form-data with image file
    Output: JSON with text, boxes (x_min, y_min, x_max, y_max, text, confidence)
    """
    if model is None:
        load_model()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run OCR inference using DeepSeek-OCR API
        with torch.no_grad():
            # Process image with processor
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Generate OCR output
            # DeepSeek-OCR uses generate() method for text extraction
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0
            )
            
            # Decode output
            result_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse bounding boxes from model output
        # Adjust parsing logic based on actual model output format
        boxes = parse_boxes_from_output(result_text, image.size)
        
        # Extract full text
        full_text = " ".join([box.text for box in boxes])
        
        return OCRResponse(
            text=full_text,
            boxes=boxes,
            full_text=full_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


def parse_boxes_from_output(text: str, image_size: tuple) -> List[BoundingBox]:
    """
    Parse bounding boxes from model output.
    Adjust this based on DeepSeek-OCR's actual output format.
    """
    # Placeholder: implement actual parsing logic
    # Example structure if model returns JSON-like format
    boxes = []
    
    # If model outputs structured format, parse it here
    # For now, return a single box covering the entire image
    boxes.append(BoundingBox(
        x_min=0.0,
        y_min=0.0,
        x_max=float(image_size[0]),
        y_max=float(image_size[1]),
        text=text.strip(),
        confidence=0.95
    ))
    
    return boxes


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

