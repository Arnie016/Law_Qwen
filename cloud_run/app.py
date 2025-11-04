"""
Example Cloud Run service template.
Customize based on your hackathon category (AI Studio, AI Agents, GPU).
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os

app = FastAPI(title="Cloud Run Hackathon App")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Cloud Run Hackathon App",
        "category": os.getenv("CATEGORY", "AI Studio")
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Add your endpoints here based on hackathon category
# Example: AI Studio endpoint
@app.post("/generate")
async def generate():
    # Your logic here
    return {"result": "generated"}


