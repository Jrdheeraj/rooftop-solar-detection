# Frontend-Backend Integration Guide

## Overview
This project now has a fully integrated frontend and backend system for rooftop solar panel detection using YOLOv8 and Google Static Maps API.

## Architecture

### Backend (FastAPI)
- **Location**: `src/api.py`
- **Port**: 8002
- **Features**:
  - Image upload and analysis
  - Coordinate-based satellite imagery fetching
  - Advanced YOLO inference with buffer logic
  - Quality control and area calculations

### Frontend (React + Vite)
- **Location**: `frontend/`
- **Port**: 5174
- **Features**:
  - Modern UI with TailwindCSS
  - Image upload interface
  - Coordinate input with validation
  - Real-time results display

## API Endpoints

### 1. Health Check
```
GET /health
```
Returns server status.

### 2. Image Upload Analysis
```
POST /predict
Content-Type: multipart/form-data
Parameters:
- file: Image file (jpg/png)
- confidence: float (0-1, default 0.5)
```

### 3. Coordinate-based Analysis
```
POST /coords?lat={lat}&lng={lng}&confidence={confidence}
```
Requires Google Static Maps API key in `.env` file.

### 4. Batch Processing
```
POST /batch
Content-Type: multipart/form-data
Parameters:
- files: Multiple image files
- confidence: float (0-1, default 0.5)
```

## Setup Instructions

### 1. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up Google Static Maps API key (optional but recommended)
cp .env.example .env
# Edit .env and add your API key

# Start the backend server
python src/api.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Access the Application
- Frontend: http://localhost:5174
- Backend API: http://localhost:8002
- API Documentation: http://localhost:8002/docs

## Key Features

### Advanced Inference Engine
- **Buffer Logic**: Uses 1200 sqft and 2400 sqft buffers for comprehensive detection
- **Area Calculation**: Precise area measurements in square meters and square feet
- **Quality Control**: Sharpness and lighting assessment
- **Multi-panel Detection**: Identifies and analyzes all solar panels in the buffer area

### Google Static Maps Integration
- **Automatic Image Fetching**: Retrieves satellite imagery for any coordinates
- **High Resolution**: Uses zoom level 19 for detailed rooftop analysis
- **API Key Management**: Secure handling of Google Maps API credentials

### Frontend Features
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Analysis**: Immediate feedback and results
- **Interactive Results**: Detailed panel information and confidence scores
- **Error Handling**: User-friendly error messages and validation

## Configuration

### Environment Variables (.env)
```
GOOGLE_STATIC_MAPS_API_KEY=your_api_key_here
```

### Model Configuration
- **Model Path**: `models/solar_model_best.pt`
- **Confidence Threshold**: 0.25 (default for inference)
- **Buffer Sizes**: 1200 sqft (residential), 2400 sqft (commercial)

## Troubleshooting

### Common Issues

1. **Port Conflicts**: If ports 8002 or 5174 are in use, the system will automatically try alternative ports.

2. **Google Maps API Error**: 
   - Ensure the API key is valid and has the "Static Maps API" enabled
   - Check billing is enabled for your Google Cloud project

3. **Model Loading Error**: 
   - Verify `models/solar_model_best.pt` exists
   - Check PyTorch and Ultralytics dependencies

4. **Frontend 502 Error**: 
   - Ensure the backend server is running on port 8002
   - Check the Vite proxy configuration in `vite.config.js`

### Development Tips

1. **Testing**: Use the API documentation at http://localhost:8002/docs to test endpoints directly

2. **Debugging**: Check browser console for frontend errors and backend terminal for API logs

3. **Performance**: For batch processing, consider using smaller images or implementing queuing

## Future Enhancements

1. **Real-time Coordinates**: Add map interface for coordinate selection
2. **Historical Data**: Store and analyze detection results over time
3. **Export Features**: Add PDF/CSV export for results
4. **Authentication**: User accounts and usage tracking
5. **Cloud Deployment**: Docker containers for cloud deployment
