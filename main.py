import sys
import os

# Добавляем корень проекта в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import uuid
from core.pipeline import AgroInferencePipeline

app = FastAPI(title="AgroScan AI API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pipeline
# Note: In production, paths to .pt weights would be provided
pipeline = AgroInferencePipeline()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), culture: str = Form("auto")):
    """
    Standard analysis endpoint using the Hybrid Pipeline with Culture Filtering.
    """
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        results = pipeline.run_inference(file_path, culture=culture)
        os.remove(file_path) # Clean up
        
        disease_name = results["disease_name"]
        
        return {
            "status": "success",
            "data": {
                "disease": disease_name,
                "confidence": round(results["confidence"] * 100, 2),
                "affected_area": round(results["affected_area_pct"], 2),
                "recommendation": "Consult with an agronomist for specific treatment." if "Healthy" not in disease_name else "Keep up the good work!",
                "probabilities": results["probs"]
            }
        }
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fast-check")
async def fast_check(file: UploadFile = File(...)):
    """
    Lightweight endpoint for Real-Time Camera streams.
    """
    file_path = os.path.join(UPLOAD_DIR, f"rt_{uuid.uuid4()}.jpg")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        results = pipeline.run_inference(file_path, fast_mode=True)
        os.remove(file_path)
        
        disease_name = pipeline.class_names[results["disease_index"]] if results["disease_index"] < len(pipeline.class_names) else "Unknown"
        
        return {
            "disease": disease_name,
            "confidence": round(results["confidence"] * 100, 2)
        }
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
    return {"status": "AgroScan AI API is Online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
