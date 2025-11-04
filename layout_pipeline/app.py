"""
Layout Analysis Pipeline: OCR → LayoutLMv3 → LLM reasoning
Single /analyze endpoint returning text, layout, and insights.
"""
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

app = FastAPI(title="Layout Analysis Pipeline", version="1.0.0")

# Models (lazy loaded)
ocr_model = None
layout_model = None
llm_model = None
device = None


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    text: str
    confidence: float
    label: Optional[str] = None  # paragraph, table, title, etc.


class LayoutInsight(BaseModel):
    type: str  # summary, entity, total, etc.
    content: str
    confidence: float


class AnalyzeResponse(BaseModel):
    text: str
    boxes: List[BoundingBox]
    layout_structure: Dict[str, Any]  # reading order, paragraphs, tables
    insights: List[LayoutInsight]


def load_models():
    """Load OCR, LayoutLMv3, and LLM models."""
    global ocr_model, layout_model, llm_model, device
    
    if device is not None:
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load OCR model (e.g., Tesseract wrapper or DeepSeek-OCR)
    # Placeholder: adjust based on actual OCR choice
    try:
        import pytesseract
        print("OCR ready (Tesseract)")
    except:
        print("Warning: Tesseract not available")
    
    # Load LayoutLMv3
    try:
        from transformers import AutoModelForTokenClassification, AutoProcessor
        
        layout_model_name = "microsoft/layoutlmv3-base"
        print(f"Loading LayoutLMv3: {layout_model_name}")
        
        processor = AutoProcessor.from_pretrained(layout_model_name)
        layout_model = AutoModelForTokenClassification.from_pretrained(
            layout_model_name,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
        )
        layout_model = layout_model.to(device)
        layout_model.eval()
        print("LayoutLMv3 loaded")
    except Exception as e:
        print(f"LayoutLMv3 loading error: {e}")
    
    # Load LLM for reasoning (lightweight, e.g., Phi-2 or Qwen)
    # Placeholder: adjust based on choice
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        llm_name = "microsoft/phi-2"  # or "Qwen/Qwen-2-1.5B"
        print(f"Loading LLM: {llm_name}")
        
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto"
        )
        llm_model.eval()
        print("LLM loaded")
    except Exception as e:
        print(f"LLM loading error: {e}")


@app.on_event("startup")
async def startup_event():
    """Preload models on startup."""
    # load_models()  # Uncomment to preload
    pass


@app.get("/")
async def root():
    return {"status": "ok", "service": "Layout Analysis Pipeline"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "ocr": ocr_model is not None,
            "layout": layout_model is not None,
            "llm": llm_model is not None
        }
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    Full pipeline: OCR → Layout → LLM reasoning.
    
    Returns: text, bounding boxes, layout structure, and insights.
    """
    if device is None:
        load_models()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # 1. Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. OCR step
        boxes, full_text = run_ocr(image)
        
        # 3. Layout analysis
        layout_structure = run_layout_analysis(image, boxes)
        
        # 4. LLM reasoning
        insights = run_llm_reasoning(full_text, layout_structure)
        
        return AnalyzeResponse(
            text=full_text,
            boxes=boxes,
            layout_structure=layout_structure,
            insights=insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def run_ocr(image: Image.Image) -> tuple[List[BoundingBox], str]:
    """Run OCR on image, return boxes and text."""
    try:
        import pytesseract
        
        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        boxes = []
        words = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text:
                boxes.append(BoundingBox(
                    x_min=float(ocr_data['left'][i]),
                    y_min=float(ocr_data['top'][i]),
                    x_max=float(ocr_data['left'][i] + ocr_data['width'][i]),
                    y_max=float(ocr_data['top'][i] + ocr_data['height'][i]),
                    text=text,
                    confidence=float(ocr_data['conf'][i]) / 100.0
                ))
                words.append(text)
        
        full_text = " ".join(words)
        return boxes, full_text
        
    except Exception as e:
        # Fallback: return empty
        return [], ""


def run_layout_analysis(image: Image.Image, boxes: List[BoundingBox]) -> Dict[str, Any]:
    """Run LayoutLMv3 to detect structure (paragraphs, tables, titles)."""
    if layout_model is None:
        return {"reading_order": [], "paragraphs": [], "tables": []}
    
    try:
        from transformers import AutoProcessor
        
        processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
        
        # Prepare inputs
        words = [box.text for box in boxes]
        boxes_coords = [[box.x_min, box.y_min, box.x_max, box.y_max] for box in boxes]
        
        encoding = processor(image, words, boxes=boxes_coords, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = layout_model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Map predictions to labels (adjust based on model's label set)
        label_map = {
            0: "other",
            1: "title",
            2: "paragraph",
            3: "table",
            4: "list"
        }
        
        # Group by label
        paragraphs = []
        tables = []
        titles = []
        
        for i, pred in enumerate(predictions):
            label = label_map.get(pred, "other")
            if label == "paragraph":
                paragraphs.append(boxes[i].text)
            elif label == "table":
                tables.append(boxes[i].text)
            elif label == "title":
                titles.append(boxes[i].text)
        
        return {
            "reading_order": [box.text for box in boxes],
            "paragraphs": paragraphs,
            "tables": tables,
            "titles": titles
        }
        
    except Exception as e:
        print(f"Layout analysis error: {e}")
        return {"error": str(e)}


def run_llm_reasoning(text: str, layout: Dict[str, Any]) -> List[LayoutInsight]:
    """Use LLM to extract insights (totals, entities, summaries)."""
    if llm_model is None:
        return []
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        
        # Build prompt
        prompt = f"""Analyze this document text and extract key insights:
- Any monetary totals or sums
- Important entities (names, dates, locations)
- Document summary (2-3 sentences)

Text: {text[:1000]}  # Truncate for context

Provide insights in JSON format with type, content, and confidence.
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse result (simplified; adjust based on actual LLM output)
        insights = [
            LayoutInsight(
                type="summary",
                content=result[:200],
                confidence=0.8
            )
        ]
        
        return insights
        
    except Exception as e:
        print(f"LLM reasoning error: {e}")
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


