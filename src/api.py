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
import tempfile
import os
from typing import Dict, Any

# Import our custom modules
from inference import SolarPanelInference
from download_google_staticmaps import GoogleStaticMapsClient, StaticMapConfig
from batch_inference import create_overlay_image

app = FastAPI(
    title="Solar Panel Detection API",
    description="Real-time solar panel detection using YOLOv8",
    version="1.0.0"
)

# Initialize model and inference engine
MODEL_PATH = "models/solar_model_best.pt"
model = YOLO(MODEL_PATH)
inference_engine = SolarPanelInference()
print(f"✅ Model loaded: {MODEL_PATH}")

# Initialize Google Static Maps client (will raise error if API key not set)
try:
    maps_client = GoogleStaticMapsClient()
    print("✅ Google Static Maps client initialized")
except RuntimeError as e:
    print(f"⚠️  Google Static Maps client not initialized: {e}")
    maps_client = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Solar Panel Detection API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "coords": "/coords (POST)",
            "batch": "/batch (POST)"
        },
        "features": {
            "image_upload": True,
            "coordinate_analysis": maps_client is not None,
            "google_static_maps": maps_client is not None
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
        JSON with detection results
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
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image)
        
        try:
            # Run inference using our custom inference engine
            # Create a mock sample_id and coordinates for uploaded image
            sample_id = 9999  # Use a special ID for uploaded images
            lat, lon = 0.0, 0.0  # Default coordinates for uploaded images
            
            # Temporarily save the image as if it were a Google Static Map
            google_img_dir = Path("data/processed/google_images_all")
            google_img_dir.mkdir(parents=True, exist_ok=True)
            temp_google_path = google_img_dir / f"{sample_id}.jpg"
            
            # Copy temp file to Google images directory
            import shutil
            shutil.copy2(temp_path, temp_google_path)
            
            # Run inference with "UPLOAD" type to use standard panel size heuristic
            result = inference_engine.predict(sample_id=sample_id, lat=lat, lon=lon, image_type="UPLOAD")
            panels = result.get("panels_in_buffer", [])

            # Load the image for overlay generation
            img_bgr = cv2.imread(str(temp_google_path))
            
            # Create overlay image with bounding boxes
            overlays_dir = Path("outputs/overlays")
            overlays_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = overlays_dir / f"{sample_id}_overlay.jpg"
            
            # Generate the overlay with bounding boxes
            create_overlay_image(img_bgr, result, overlay_path)
            
            # Transform result to match expected API format
            detections = []
            for panel in panels:
                bbox = panel["bbox_center"]
                detections.append({
                    "x1": bbox[0] - bbox[2]/2,
                    "y1": bbox[1] - bbox[3]/2,
                    "x2": bbox[0] + bbox[2]/2,
                    "y2": bbox[1] + bbox[3]/2,
                    "confidence": panel["conf"]
                })
            
            # Convert overlay image to base64 for frontend display
            import base64
            with open(overlay_path, "rb") as img_file:
                overlay_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                overlay_data_url = f"data:image/jpeg;base64,{overlay_base64}"
            
            api_result = {
                "filename": file.filename,
                "status": "success",
                "detections": len(detections),
                "detection_list": detections,
                "confidence_threshold": confidence,
                "has_solar": result["has_solar"],
                "confidence": result["confidence"],
                "pv_area_sqm_est": result["pv_area_sqm_est"],
                "estimated_capacity_kw": round(float(result["pv_area_sqm_est"]) * 0.2, 2),
                "estimated_annual_production_kwh": round(float(result["pv_area_sqm_est"]) * 0.2 * 1460, 2),
                "panels_in_buffer": result["panels_in_buffer"],
                "qc_status": result["qc_status"],
                "buffer_radius_sqft": result["buffer_radius_sqft"],
                "overlay_image": overlay_data_url,  # Add overlay image as base64
                "sample_id": sample_id,
                "latitude": lat,
                "longitude": lon,
                "best_panel_id": result["best_panel_id"],
                "bbox_or_mask": result["bbox_or_mask"],
                "image_metadata": result["image_metadata"],
                "financial_insights": result["financial_insights"],
                "environmental_impact": result["environmental_impact"],
                "technical_specs": result["technical_specs"]
            }
            
        finally:
            # Clean up temp files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if 'temp_google_path' in locals() and os.path.exists(temp_google_path):
                os.unlink(temp_google_path)
            if 'overlay_path' in locals() and os.path.exists(overlay_path):
                os.unlink(overlay_path)
        
        return api_result
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/coords")
async def predict_by_coords(lat: float, lng: float, confidence: float = 0.5):
    """
    Predict solar panels using coordinates and Google Static Maps
    
    Args:
        lat: latitude
        lng: longitude
        confidence: detection confidence threshold (0-1)
    
    Returns:
        JSON with detection results
    """
    if maps_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Google Static Maps API key not configured. Please set GOOGLE_STATIC_MAPS_API_KEY in .env file"
        )
    
    if confidence < 0 or confidence > 1:
        raise HTTPException(status_code=400, detail="Confidence must be 0-1")
    
    try:
        # Generate a sample_id for this coordinate request
        sample_id = hash(f"{lat}_{lng}") % 10000  # Create a unique ID
        
        # Fetch satellite image from Google Static Maps
        print(f"Fetching satellite image for coordinates ({lat}, {lng})")
        satellite_image = maps_client.fetch_image(lat, lng)
        
        # Save image to expected location
        google_img_dir = Path("data/processed/google_images_all")
        google_img_dir.mkdir(parents=True, exist_ok=True)
        img_path = google_img_dir / f"{sample_id}.jpg"
        satellite_image.save(img_path, format="JPEG")
        
        # Run inference with "SATELLITE" type to use pixel-to-meter conversion
        result = inference_engine.predict(sample_id=sample_id, lat=lat, lon=lng, image_type="SATELLITE")
        
        # Load the image for overlay generation
        img_bgr = cv2.imread(str(img_path))
        
        # Create overlay image with bounding boxes
        overlays_dir = Path("outputs/overlays")
        overlays_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlays_dir / f"{sample_id}_overlay.jpg"
        
        # Generate the overlay with bounding boxes
        create_overlay_image(img_bgr, result, overlay_path)
        
        # Transform result to match expected API format
        detections = []
        for panel in result.get("panels_in_buffer", []):
            bbox = panel["bbox_center"]
            detections.append({
                "x1": bbox[0] - bbox[2]/2,
                "y1": bbox[1] - bbox[3]/2,
                "x2": bbox[0] + bbox[2]/2,
                "y2": bbox[1] + bbox[3]/2,
                "confidence": panel["conf"]
            })
        
        # Convert overlay image to base64 for frontend display
        import base64
        with open(overlay_path, "rb") as img_file:
            overlay_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            overlay_data_url = f"data:image/jpeg;base64,{overlay_base64}"
        
        api_result = {
            "status": "success",
            "coordinates": {"lat": lat, "lng": lng},
            "detections": len(detections),
            "detection_list": detections,
            "confidence_threshold": confidence,
            "has_solar": result["has_solar"],
            "confidence": result["confidence"],
            "pv_area_sqm_est": result["pv_area_sqm_est"],
            "estimated_capacity_kw": round(float(result["pv_area_sqm_est"]) * 0.2, 2),
            "estimated_annual_production_kwh": round(float(result["pv_area_sqm_est"]) * 0.2 * 1460, 2),
            "panels_in_buffer": result["panels_in_buffer"],
            "qc_status": result["qc_status"],
            "buffer_radius_sqft": result["buffer_radius_sqft"],
            "image_metadata": result["image_metadata"],
            "overlay_image": overlay_data_url,  # Add overlay image as base64
            "sample_id": sample_id,
            "latitude": lat,
            "longitude": lng,
            "best_panel_id": result["best_panel_id"],
            "bbox_or_mask": result["bbox_or_mask"],
            "financial_insights": result["financial_insights"],
            "environmental_impact": result["environmental_impact"],
            "technical_specs": result["technical_specs"]
        }
        
        # Clean up the temporary image
        if img_path.exists():
            img_path.unlink()
        # Clean up the overlay image
        if overlay_path.exists():
            overlay_path.unlink()
        
        return api_result


        
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
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image)
            
            try:
                # Use our inference engine for better results
                sample_id = hash(file.filename) % 10000
                lat, lon = 0.0, 0.0
                
                # Temporarily save to Google images directory
                google_img_dir = Path("data/processed/google_images_all")
                google_img_dir.mkdir(parents=True, exist_ok=True)
                temp_google_path = google_img_dir / f"{sample_id}.jpg"
                
                import shutil
                shutil.copy2(temp_path, temp_google_path)
                
                result = inference_engine.predict(sample_id=sample_id, lat=lat, lon=lon)
                
                results.append({
                    "filename": file.filename,
                    "detections": len(result.get("panels_in_buffer", [])),
                    "confidence_threshold": confidence,
                    "has_solar": result["has_solar"],
                    "pv_area_sqm_est": result["pv_area_sqm_est"]
                })
                
                # Clean up
                if os.path.exists(temp_google_path):
                    os.unlink(temp_google_path)
                    
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    return {
        "total_images": len(files),
        "results": results
    }


if __name__ == "__main__":
    print("\n🚀 Starting FastAPI server...")
    print("API docs: http://localhost:8002/docs")
    print("API redoc: http://localhost:8002/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
