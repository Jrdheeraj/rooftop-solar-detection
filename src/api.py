# src/api.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn
import time
from datetime import datetime
from pydantic import BaseModel

class CoordinateRequest(BaseModel):
    latitude: float
    longitude: float
    buffer_radius_sqft: int | None = 1200

# Import our custom modules
from inference import SolarPanelInference
from download_google_staticmaps import GoogleStaticMapsClient
from batch_inference import render_panel_detections
from request_utils import build_request_sample_id, validate_coordinates

app = FastAPI(
    title="Solar Panel Detection API",
    description="Real-time solar panel detection using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware immediately after app initialization
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rooftop-solar-detection.vercel.app",
        "https://rooftop-solar-detection.vercel.app/",
        "https://rooftop-solar-detection.onrender.com",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and inference engine
MODEL_PATH = "models/solar_model_best.pt"
model = YOLO(MODEL_PATH)
inference_engine = SolarPanelInference()
print(f"Model loaded: {MODEL_PATH}")

# Initialize Google Static Maps client (will raise error if API key not set)
try:
    maps_client = GoogleStaticMapsClient()
    print("Google Static Maps client initialized")
except RuntimeError as e:
    print(f"ERROR: Google Static Maps client not initialized: {e}")
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
            "satellite_upload": True,
            "coordinate_analysis": maps_client is not None,
            "google_static_maps": maps_client is not None
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    image_type: str = Form("PHOTO"),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    buffer_radius_sqft: int | None = Form(None),
):
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
        t_start = time.time()
        # Read uploaded image
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        t_read = time.time()
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        requested_image_type = (image_type or "PHOTO").upper()
        if requested_image_type not in {"PHOTO", "SATELLITE", "UPLOAD"}:
            raise HTTPException(status_code=400, detail="image_type must be PHOTO or SATELLITE")

        if requested_image_type == "SATELLITE":
            if latitude is None or longitude is None:
                raise HTTPException(
                    status_code=400,
                    detail="Satellite image uploads require latitude and longitude for accurate scaling",
                )
            lat, lon = validate_coordinates(latitude, longitude)
            model_image_type = "SATELLITE"
        else:
            lat = float(latitude) if latitude is not None else 0.0
            lon = float(longitude) if longitude is not None else 0.0
            model_image_type = "UPLOAD"

        # Run inference
        result = inference_engine.predict(
            sample_id=build_request_sample_id("upload", file.filename or "upload", len(contents)),
            lat=lat,
            lon=lon,
            image_type=model_image_type,
            img_bgr=image_bgr,
            conf_threshold=confidence,
            buffer_sqft=buffer_radius_sqft,
        )
        t_infer = time.time()
        
        # Generate overlay
        overlay_data_url = render_panel_detections(image_bgr, result, return_base64=True)
        t_overlay = time.time()
        
        # Log timings
        print(f"⏱️  UPLOAD ANALYSIS: Read={t_read-t_start:.3f}s, Inference={t_infer-t_read:.3f}s, Overlay={t_overlay-t_infer:.3f}s, Total={t_overlay-t_start:.3f}s")
        
        # Transform result to match expected API format
        panels = result.get("panels_in_buffer", [])
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
        
        api_result = {
            "filename": file.filename,
            "status": "success",
            "overlay_path": overlay_data_url,
            "requested_image_type": requested_image_type,
            **result,
            "processed_at": datetime.now().isoformat()
        }
        
        return api_result
    
    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/coords")
async def predict_by_coords(req: CoordinateRequest, confidence: float = 0.5):
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
        lat, lng = validate_coordinates(req.latitude, req.longitude)
        buffer_radius_sqft = int(req.buffer_radius_sqft or 1200)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    try:
        t_start = time.time()
        sample_id = build_request_sample_id("coords", lat, lng)
        
        # Fetch satellite image from Google Static Maps
        satellite_image = maps_client.fetch_image(lat, lng)
        t_fetch = time.time()
        
        # Convert PIL to BGR numpy array
        image_np = np.array(satellite_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run inference using in-memory image
        result = inference_engine.predict(
            sample_id=sample_id, 
            lat=lat, 
            lon=lng, 
            image_type="SATELLITE",
            img_bgr=image_bgr,
            conf_threshold=confidence,
            buffer_sqft=buffer_radius_sqft,
        )
        t_infer = time.time()
        
        # Generate overlay in-memory as base64
        overlay_data_url = render_panel_detections(image_bgr, result, return_base64=True)
        t_overlay = time.time()
        
        # Log timings
        print(f"⏱️  COORDS ANALYSIS: Fetch={t_fetch-t_start:.3f}s, Inference={t_infer-t_fetch:.3f}s, Overlay={t_overlay-t_infer:.3f}s, Total={t_overlay-t_start:.3f}s")
        
        # Combine result with overlay
        api_result = {
            "status": "success",
            "overlay_path": overlay_data_url,
            **result,
            "processed_at": datetime.now().isoformat()
        }
        
        return api_result


        
    except HTTPException:
        raise
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
    
    for idx, file in enumerate(files):
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is not None:
            sample_id = build_request_sample_id("batch_upload", file.filename or "upload", len(contents), idx)
            lat, lon = 0.0, 0.0
            result = inference_engine.predict(
                sample_id=sample_id,
                lat=lat,
                lon=lon,
                image_type="UPLOAD",
                img_bgr=image,
                conf_threshold=confidence,
                buffer_sqft=0,
            )
            
            results.append({
                "filename": file.filename,
                "detections": len(result.get("panels_in_buffer", [])),
                "confidence_threshold": confidence,
                "has_solar": result["has_solar"],
                "pv_area_sqm_est": result["pv_area_sqm_est"]
            })
    
    return {
        "total_images": len(files),
        "results": results
    }


if __name__ == "__main__":
    print("\nStarting FastAPI server...")
    print("API docs: http://localhost:8002/docs")
    print("API redoc: http://localhost:8002/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
