# Rooftop Solar Panel Detection from Overhead Imagery

**End-to-end computer vision pipeline** for detecting and measuring solar panels on rooftops from Google Static Maps satellite imagery using YOLOv8 and geospatial buffer logic.

## Overview

This project:
- **Trains** a YOLOv8 object detection model on an open rooftop-solar dataset (Roboflow export) in Google Colab.
- **Infers** on Google Static Maps tiles, applying a two-stage buffer strategy (1,200 and 2,400 sq ft).
- **Calculates** precise panel area intersection with buffers, handling partial overlaps and multiple detections.
- **Outputs** a standardized JSON with solar panel predictions and quality assurance metadata.

> **Data Compliance**: The model trains on open, publicly available imagery only (not Google Static Maps). Inference runs on Google tiles, which is permitted under their terms for non-commercial mapping use.

---

## Project Structure

```
rooftop-solar-detection/
│
├── config/                             # Configuration files
│
├── data/
│   ├── raw/
│   │   └── EI_train_data.csv           # Input: (sample_id, latitude, longitude)
│   └── processed/
│       ├── dataset/                    # YOLOv8 training dataset
│       │   ├── images/
│       │   │   ├── train/              # 500+ training tiles
│       │   │   └── val/                # 100+ validation tiles
│       │   ├── labels/
│       │   │   ├── train/              # YOLO format annotations (.txt)
│       │   │   └── val/
│       │   └── data.yaml               # Dataset config (Roboflow export)
│       └── google_images_all/          # Downloaded Google tiles: {sample_id}.jpg
│
├── models/
│   └── solar_model_best.pt             # YOLOv8n weights (trained in Colab, ~50 epochs)
│
├── outputs/
│   ├── solar_rooftops_google.json      # Intermediate: batch inference results
│   └── predictions_final.json          # Final output: standardized predictions
│
├── src/
│   ├── __init__.py                     # Python package marker
│   │
│   ├── inference.py                    # Core inference engine
│   │   ├── AreaCalculator              # Bounding box ↔ area conversions (m², sq ft)
│   │   ├── QCChecker                   # Quality assurance (sharpness, darkness, conf)
│   │   └── SolarPanelInference         # Main: 1200→2400 buffer logic, panel selection
│   │
│   ├── export_rooftop_json.py          # Batch processor: CSV → intermediate JSON
│   ├── build_final_predictions_json.py # Final processor: type casting + schema validation
│   │
│   ├── image_fetcher.py                # Google Static Maps API downloader
│   ├── staticmaps_fetcher.py           # Legacy / alternative fetcher
│   │
│   ├── model_trainer.py                # (Reference) Local YOLOv8 training script
│   ├── yolo_dataset_creator.py         # YOLO-format dataset utilities
│   ├── augmentation.py                 # (Optional) Custom image augmentations
│   ├── data_explorer.py                # (Optional) Exploratory data analysis
│   └── api.py                          # (Optional) REST API / demo wrapper
│
├── notebooks/                          # (Optional) Jupyter notebooks for exploration
│
├── tests/                              # (Optional) Unit tests
│
├── .env.example                        # Environment variables template (API keys, paths)
├── .gitignore
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container for inference
├── docker-compose.yml                  # (Optional) Multi-service composition
├── LICENSE
├── Model-Card.md                       # Model documentation
└── README.md                           # This file
```

---

## Training Pipeline (Google Colab)

**Training is performed in a dedicated Google Colab notebook:**

- **Notebook URL**: https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing
- **Runtime**: GPU (A100 or T4)
- **Time**: ~30–60 minutes for 50 epochs on ~640 images

### Training Steps

1. **Mount Google Drive**  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Install YOLOv8**  
   ```python
   !pip install ultralytics
   ```

3. **Prepare Dataset**  
   - Upload `dataset.zip` (Roboflow export) to Google Drive.
   - Unzip into Colab workspace:

   ```python
   !unzip -q "/content/drive/My Drive/solar panel detection/dataset.zip" -d "/content/datasets"
   ```

4. **Train YOLOv8**  
   ```python
   !yolo detect train \
     model=yolov8n.pt \
     data=/content/datasets/dataset/data.yaml \
     epochs=50 \
     imgsz=640 \
     batch=16
   ```

5. **Export Weights**  
   ```python
   from google.colab import files
   files.download('/content/runs/detect/train/weights/best.pt')
   ```

6. **Deploy Locally**  
   - Save downloaded `best.pt` to: `models/solar_model_best.pt`

---

## Inference Pipeline (Local / VS Code)

### Prerequisites

```bash
# Clone or navigate to project
cd rooftop-solar-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install ultralytics opencv-python numpy pandas pyyaml
```

### Files Required

Before running inference:

- ✅ `models/solar_model_best.pt` (trained model)
- ✅ `data/raw/EI_train_data.csv` (sample list)
- ✅ `data/processed/google_images_all/{sample_id}.jpg` (Google tiles)

### Running Inference

**Option 1: Single-Sample Test**

```bash
python src/inference.py
```

Expected output (for sample_id=1234):

```json
{
  "sample_id": 1234,
  "lat": 12.9716,
  "lon": 77.5946,
  "has_solar": true,
  "confidence": 0.5277,
  "pv_area_sqm_est": 2.83,
  "buffer_radius_sqft": 2400,
  "qc_status": "VERIFIABLE",
  "bbox_or_mask": "0.4118,0.5786,0.0564,0.0539",
  "image_metadata": {
    "source": "XYZ",
    "capture_date": "YYYY-MM-DD"
  }
}
```

**Option 2: Batch Processing (All Samples)**

```bash
# Batch inference
python -m src.export_rooftop_json

# Final JSON construction
python -m src.build_final_predictions_json
```

Outputs:
- `outputs/solar_rooftops_google.json` (intermediate, untyped)
- `outputs/predictions_final.json` (final, schema-validated)

---

## Core Algorithm: Buffer-Based Panel Detection

### Logic Flow

1. **Load tile**: `data/processed/google_images_all/{sample_id}.jpg`

2. **Run YOLOv8**: Detect all panels in the image.

3. **Build buffer**: Create circular buffer (1,200 or 2,400 sq ft) centered on the tile.

4. **Compute intersections**:
   - For each detected panel bbox:
     - Convert YOLO format (center-based `x_c, y_c, w, h`) to normalized top-left (`x1, y1, w, h`).
     - Compute intersection ratio with buffer:
       ```
       inter_ratio = area(panel ∩ buffer) / area(panel)
       ```
     - If `inter_ratio > 0`:
       - Compute full panel area: `full_area_m² = bbox_area_m²(panel)`
       - Compute inside area: `inside_area_m² = full_area_m² × inter_ratio`
     - Append to valid panels list.

5. **Select best panel**:
   - Among all valid panels, pick the one with **max `inside_area_m²`**.
   - Set `has_solar = True`, `pv_area_sqm_est = inside_area_m²`.

6. **Two-pass rule**:
   - Run with 1,200 sq ft buffer first.
   - If `has_solar = True`, return 1,200-sq-ft record.
   - If `has_solar = False`, run with 2,400 sq ft buffer.
   - If 2,400 finds solar, return 2,400-sq-ft record.
   - Otherwise, return 1,200-sq-ft "no solar" record.

7. **Quality control**:
   - Compute image sharpness (Laplacian variance) and darkness ratio.
   - Set `qc_status = "VERIFIABLE"` or `"NOT_VERIFIABLE"`.

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | YOLOv8n | Small, fast, sufficient for rooftop detection |
| Image size | 640×640 px | Google Static Maps default; ~19 zoom level |
| Buffer 1 | 1,200 sq ft | ~110 m radius; typical rooftop extent |
| Buffer 2 | 2,400 sq ft | ~155 m radius; fallback for edge cases |
| Confidence threshold | > 0 (all detections considered) | Filtering by intersection ratio |
| Intersection ratio | > 0 (any overlap) | Panels partially inside buffer accepted |

---

## Output Schema

Final JSON (`predictions_final.json`):

```json
[
  {
    "sample_id": 1234,
    "lat": 12.9716,
    "lon": 77.5946,
    "has_solar": true,
    "confidence": 0.5277,
    "pv_area_sqm_est": 2.83,
    "buffer_radius_sqft": 2400,
    "qc_status": "VERIFIABLE",
    "bbox_or_mask": "0.4118,0.5786,0.0564,0.0539",
    "image_metadata": {
      "source": "XYZ",
      "capture_date": "YYYY-MM-DD"
    }
  },
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | int | Unique tile identifier |
| `lat`, `lon` | float | Coordinates (WGS84) |
| `has_solar` | bool | Panel detected and inside buffer |
| `confidence` | float | YOLO model confidence [0–1] |
| `pv_area_sqm_est` | float | Panel area inside buffer (m²) |
| `buffer_radius_sqft` | int | Buffer used: 1200 or 2400 |
| `qc_status` | str | "VERIFIABLE" or "NOT_VERIFIABLE" |
| `bbox_or_mask` | str | Normalized bbox: "x_c,y_c,w,h" or empty |
| `image_metadata.source` | str | "XYZ" or data source name |
| `image_metadata.capture_date` | str | "YYYY-MM-DD" or actual date |

---

## Dependencies

See `requirements.txt`:

```
ultralytics>=8.0.0
opencv-python>=4.6.0
numpy>=1.23.0
pandas>=1.5.0
pyyaml>=5.3.1
torch>=1.8.0
torchvision>=0.9.0
```

---

## Docker (Optional)

Build and run inference inside a container:

```bash
docker build -t rooftop-solar-detection .
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/outputs:/app/outputs \
           rooftop-solar-detection \
           python -m src.export_rooftop_json
```

---

## References

- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Roboflow Universe**: https://universe.roboflow.com/
- **Web Mercator Projection**: https://en.wikipedia.org/wiki/Web_Mercator_projection

---

## License

See LICENSE file.

---

## Contact & Support

For issues or questions, open a GitHub issue or reach out to the maintainer.