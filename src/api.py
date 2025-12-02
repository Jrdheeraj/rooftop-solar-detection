# src/api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import uvicorn

app = FastAPI(
    title="Solar Panel Detection API",
    description="Real-time solar panel detection using YOLOv8",
    version="1.0.0"
)

# Load model once at startup
MODEL_PATH = "models/solar_model_best.pt"
model = YOLO(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Solar Panel Detection API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch": "/batch (POST)"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: float = 0.5):
    """
    Predict solar panels on uploaded image
    
    Args:
        file: image file (jpg/png)
        confidence: detection confidence threshold (0-1)
    
    Returns:
        JSON with:
        - detections: number of panels found
        - confidence_scores: list of scores per detection
        - boxes: [x, y, width, height] for each box (normalized)
    """
    
    if confidence < 0 or confidence > 1:
        raise HTTPException(status_code=400, detail="Confidence must be 0-1")
    
    try:
        # Read uploaded image
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Save to temp location
        temp_image_path = "temp_upload.jpg"
        cv2.imwrite(temp_image_path, image)
        
        # Run inference
        results = model.predict(
            source=temp_image_path,
            conf=confidence,
            verbose=False
        )
        result = results[0]
        
        # Extract detections
        detections = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": round(conf, 4)
                })
        
        # Clean up
        Path(temp_image_path).unlink(missing_ok=True)
        
        return {
            "filename": file.filename,
            "status": "success",
            "detections": len(detections),
            "detection_list": detections,
            "confidence_threshold": confidence
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/batch")
async def batch_predict(files: list[UploadFile] = File(...), confidence: float = 0.5):
    """
    Predict on multiple images
    """
    results = []
    
    for file in files:
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is not None:
            temp_path = "temp_batch.jpg"
            cv2.imwrite(temp_path, image)
            
            preds = model.predict(source=temp_path, conf=confidence, verbose=False)
            pred = preds[0]
            
            results.append({
                "filename": file.filename,
                "detections": len(pred.boxes),
                "confidence_threshold": confidence
            })
            
            Path(temp_path).unlink(missing_ok=True)
    
    return {
        "total_images": len(files),
        "results": results
    }


if __name__ == "__main__":
    print("\n🚀 Starting FastAPI server...")
    print("API docs: http://localhost:8000/docs")
    print("API redoc: http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
