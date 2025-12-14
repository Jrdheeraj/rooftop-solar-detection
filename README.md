# Rooftop Solar Panel Detection from Overhead Imagery

**End-to-end computer vision pipeline** for detecting and measuring solar panels on rooftops from Google Static Maps satellite imagery using YOLOv8, parallel batch processing, and geospatial buffer logic with **multi-panel detection and visualization**.

## 🐳 Docker Hub Repository

**Docker Image**: `dheerajjk/rooftop-solar-cpu:latest`

- **Repository (Docker Hub URL)**: https://hub.docker.com/r/dheerajjk/rooftop-solar-cpu
- **Image name**: `dheerajjk/rooftop-solar-cpu`
- **Tag**: `latest`
- **Full image reference**: `dheerajjk/rooftop-solar-cpu:latest`
- **Pull command**: 
  ```bash
  docker pull dheerajjk/rooftop-solar-cpu:latest
  ```

---

## 🎓 Quick Links

### **🔥 TRAIN MODEL IN COLAB (RECOMMENDED)**
**⭐ [Open Training Notebook in Google Colab](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)**

**Direct Link**: https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing

- **Runtime**: GPU (A100 or T4) - Free tier available
- **Time**: ~30–60 minutes for 50 epochs
- **No setup needed**: Runs entirely in browser
- **Auto-downloads**: Model weights available after training

---

## Overview

This project:
- **Trains** a YOLOv8 object detection model in Google Colab (see link above)
- **Infers** on Google Static Maps tiles using **multiprocessing for parallel batch processing**, applying a two-stage buffer strategy (1,200 and 2,400 sq ft)
- **Detects multiple panels** per image and highlights the largest panel in lime green
- **Calculates** precise panel area intersection with buffers, handling partial overlaps and multiple detections
- **Generates** overlay visualizations with multi-panel detection, audit-friendly metadata for each sample
- **Exports** standardized JSON with solar panel predictions and quality assurance metadata
- **Containerized** with Docker for easy deployment and reproducibility

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
│   │   └── EI_train_data.csv           # Input: (sampleid, latitude, longitude, hassolar)
│   └── processed/
│       ├── dataset/                    # YOLOv8 training dataset
│       │   ├── images/
│       │   │   ├── train/              # 500+ training tiles
│       │   │   └── val/                # 100+ validation tiles
│       │   ├── labels/
│       │   │   ├── train/              # YOLO format annotations (.txt)
│       │   │   └── val/
│       │   └── data.yaml               # Dataset config (Roboflow export)
│       └── google_images_all/          # Downloaded Google tiles{sampleid}.jpg
│
├── models/
│   └── solar_model_best.pt             # YOLOv8n weights (trained in Colab, ~30 epochs)
│
├── outputs/
│   ├── logs/
│       ├──results.csv
│       └──results.png
│   ├── predictions.json                # Final output: standardized predictions (type-cast)
│   ├── predictions.csv                 # CSV export of predictions
│   ├── predictions_final.json          # Final submission format
│   └── overlays/                       # Visualization artifacts
│       └── {sample_id}_overlay.jpg     # Annotated detection overlays (multi-panel)
│
├── src/
│   ├── __init__.py                     # Python package marker
│   ├── inference.py                    # Core inference engine
│   ├── batch_inference.py              # ⭐ Unified batch processor with multi-panel detection
│   ├── download_google_staticmaps.py   # Google Static Maps API downloader
│   ├── api.py                          # (Optional) REST API / demo wrapper
│   ├── model_trainer.py                # (Reference) Local YOLOv8 training script
│   ├── yolo_dataset_creator.py         # YOLO-format dataset utilities
│   ├── augmentation.py                 # (Optional) Custom image augmentations
│   └── data_explorer.py                # (Optional) Exploratory data analysis
│
├── notebooks/                          # (Optional) Jupyter notebooks for exploration
├── tests/                              # (Optional) Unit tests
├── venv/                               # virtual environment 
├── .gitignore
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container for batch inference
├── docker-compose.yml                  # (Optional) Multi-service composition
├── LICENSE
├── Model-Card.md                       # Model documentation
└── README.md                           # This file
```

---

## 🐳 Docker Deployment

### Quick Start with Docker

**Pull the image**:
```bash
docker pull dheerajjk/rooftop-solar-cpu:latest
```

**Run inference inside container**:
```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/outputs:/app/outputs \
           dheerajjk/rooftop-solar-cpu:latest \
           python src/batch_inference.py
```

### Build Locally (Optional)

```bash
docker build -t rooftop-solar-detection .
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/outputs:/app/outputs \
           rooftop-solar-detection \
           python src/batch_inference.py
```

### Using Docker Compose

```bash
docker-compose up --build
```

---

## Training Pipeline (Google Colab) ⭐

### **[👉 CLICK HERE TO OPEN COLAB NOTEBOOK 👈](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)**

**Direct Link**: https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing

### Training Configuration

- **Runtime**: GPU (A100 or T4)
- **Time**: ~30–60 minutes for 50 epochs on ~640 images
- **Environment**: Python 3.10, CUDA 12.x, PyTorch 2.x
- **Cost**: Free tier (15 GPU hours/month)

### Training Steps (in Colab Notebook)

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
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Files Required

Before running inference:

- ✅ `models/solar_model_best.pt` (trained in Colab, see link above)
- ✅ `data/raw/EI_train_data.csv` (sample list with columns: sampleid, latitude, longitude, hassolar)
- ✅ `data/processed/google_images_all/{sample_id}.jpg` (Google tiles, one per sample)

### Running Inference

**Option 1: Single-Sample Test**

```bash
python src/inference.py
```

Expected output (for sample_id=1234):
```json
{
  "sample_id": 1234,
  "latitude": 12.9716,
  "longitude": 77.5946,
  "has_solar": true,
  "confidence": 0.5277,
  "pv_area_sqm_est": 2.83,
  "buffer_radius_sqft": 2400,
  "qc_status": "VERIFIABLE",
  "panels_in_buffer": [
    {
      "panel_id": 0,
      "conf": 0.5277,
      "full_area_sqm": 3.12,
      "inside_area_sqm": 2.83,
      "overlap_ratio": 0.907,
      "bbox_center": [0.4118, 0.5786, 0.0564, 0.0539]
    }
  ],
  "best_panel_id": 0,
  "bbox_or_mask": "0.4118,0.5786,0.0564,0.0539",
  "image_metadata": {
    "source": "GOOGLE_STATIC_MAPS",
    "zoom": 19,
    "conf_threshold": 0.25,
    "img_shape": [400, 400]
  }
}
```

**Option 2: Batch Processing (All Samples) - RECOMMENDED**

```bash
# Run batch pipeline with multi-panel detection
python src/batch_inference.py

# OR with custom worker count
python -m src.batch_inference 4
```

Outputs:
- `outputs/solar_rooftops_google.json` (intermediate, untyped)
- `outputs/predictions.json` (final, schema-validated, type-cast)
- `outputs/predictions.csv` (CSV export)
- `outputs/predictions_final.json` (final submission format)
- `outputs/overlays/{sample_id}_overlay.jpg` (per-sample visualizations with multi-panel detection)

---

## NEW: Multi-Panel Detection & Visualization

### Key Features

1. **Multiple Panel Detection**: Detects and displays all solar panels found in each image
2. **Largest Panel Highlighting**: The panel with the largest area inside the buffer is highlighted in **lime green**
3. **Other Panels**: Additional panels are displayed in **cyan**
4. **Comprehensive Labels**: Each panel shows:
   - Panel ID
   - Confidence score
   - Area in m² (intersection with buffer)
5. **Summary Statistics**: Overlay includes:
   - Total area of all panels
   - Average confidence across all panels
   - Total number of panels found

### Visualization Format

Each overlay image (`outputs/overlays/{sample_id}_overlay.jpg`) includes:

- **Title**: "Solar Panel Detection - PANELS DETECTED" (when panels found)
- **Legend**: 
  - Lime = Best Panel (Highest Overlap)
  - Cyan = Other Panels
  - Area = Intersection with Buffer
  - Confidence = Model Detection Score
- **Panel Bounding Boxes**:
  - Lime green box (thicker line) = Largest panel
  - Cyan boxes = Other detected panels
  - Labels: "Panel X | Conf: Y | Area: Z m²"
- **Information Box** (bottom-left):
  - Sample ID
  - Latitude/Longitude (formatted as "21.7609°N | 70.6191°E")
  - has_solar: YES/NO
  - Total Area: X.X m²
  - Confidence: X.XXXX
  - Panels Found: N
  - QC Status
  - Buffer: XXXX sqft

---

## Batch Processing with Parallel Workers

### Quick Start

```bash
# Navigate to project
cd rooftop-solar-detection
venv\Scripts\activate  # Windows

# Run batch inference with auto-detected worker count
python src/batch_inference.py

# OR specify custom worker count
python -m src.batch_inference 4
```

### Output Files Generated

After running batch pipeline:

```
outputs/
├── solar_rooftops_google.json          # Intermediate results (untyped)
├── predictions.json                   # Final predictions (type-cast + validated)
├── predictions.csv                    # CSV export
├── predictions_final.json             # Final submission format
└── overlays/
    ├── 1001_overlay.jpg               # Per-sample visualization (multi-panel)
    ├── 1002_overlay.jpg
    └── ...
```

### Performance

| Metric | Value |
|--------|-------|
| **Workers** | Auto: `cpu_count() - 1` (e.g., 7 on 8-core system) |
| **Speed per sample** | ~0.5–2 seconds (model load + inference + overlay) |
| **For 3,000 samples** | ~25–50 minutes total (parallel) vs 1.5–3 hours (serial) |
| **Memory** | ~2GB per worker + base (model shared) |

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
     - Append to valid panels list (`panels_in_buffer`).

5. **Select best panel**:
   - Among all valid panels, pick the one with **max `inside_area_m²`**.
   - Set `best_panel_id` to this panel's ID.
   - Set `hassolar = True`, `pv_area_sqm_est = inside_area_m²`.

6. **Two-pass rule**:
   - Run with 1,200 sq ft buffer first.
   - If `hassolar = True`, return 1,200-sq-ft record.
   - If `hassolar = False`, run with 2,400 sq ft buffer.
   - If 2,400 finds solar, return 2,400-sq-ft record.
   - Otherwise, return 1,200-sq-ft "no solar" record.

7. **Quality control**:
   - Compute image sharpness (Laplacian variance) and darkness ratio.
   - Set `qc_status = "VERIFIABLE"` or `"NOT_VERIFIABLE"`.

8. **Visualization**:
   - Draw all panels from `panels_in_buffer`.
   - Highlight `best_panel_id` in lime green (thicker line).
   - Draw other panels in cyan.
   - Add labels with confidence and area.

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | YOLOv8n | Small, fast, sufficient for rooftop detection |
| Image size | 640×640 px | Google Static Maps default; ~19 zoom level |
| Buffer 1 | 1,200 sq ft | ~110 m radius; typical rooftop extent |
| Buffer 2 | 2,400 sq ft | ~155 m radius; fallback for edge cases |
| Confidence threshold | 0.25 | All detections above this considered |
| Intersection ratio | > 0 (any overlap) | Panels partially inside buffer accepted |

---

## Output Schema

Final JSON (`predictions.json`):

```json
[
  {
    "sample_id": 1234,
    "latitude": 12.9716,
    "longitude": 77.5946,
    "has_solar": true,
    "confidence": 0.5277,
    "pv_area_sqm_est": 2.83,
    "buffer_radius_sqft": 2400,
    "panels_in_buffer": [
      {
        "panel_id": 0,
        "conf": 0.5277,
        "full_area_sqm": 3.12,
        "inside_area_sqm": 2.83,
        "overlap_ratio": 0.907,
        "bbox_center": [0.4118, 0.5786, 0.0564, 0.0539]
      },
      {
        "panel_id": 1,
        "conf": 0.415,
        "full_area_sqm": 1.85,
        "inside_area_sqm": 1.45,
        "overlap_ratio": 0.784,
        "bbox_center": [0.5234, 0.6123, 0.0345, 0.0289]
      }
    ],
    "best_panel_id": 0,
    "qc_status": "VERIFIABLE",
    "bbox_or_mask": "0.4118,0.5786,0.0564,0.0539",
    "image_metadata": {
      "source": "GOOGLE_STATIC_MAPS",
      "zoom": 19,
      "conf_threshold": 0.25,
      "img_shape": [400, 400],
      "overlay_path": "overlays/1234_overlay.jpg"
    }
  },
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | int | Unique tile identifier |
| `latitude`, `longitude` | float | Coordinates (WGS84) |
| `has_solar` | bool | Panel detected and inside buffer |
| `confidence` | float | YOLO model confidence [0–1] (best panel) |
| `pv_area_sqm_est` | float | Panel area inside buffer (m²) (best panel) |
| `buffer_radius_sqft` | int | Buffer used: 1200 or 2400 |
| `panels_in_buffer` | list | All detected panels with area calculations |
| `best_panel_id` | int | ID of panel with largest area inside buffer |
| `qc_status` | str | "VERIFIABLE" or "NOT_VERIFIABLE" |
| `bbox_or_mask` | str | Normalized bbox: "x_c,y_c,w,h" (best panel) |
| `image_metadata.source` | str | "GOOGLE_STATIC_MAPS" |
| `image_metadata.zoom` | int | Tile zoom level (typically 19) |
| `image_metadata.conf_threshold` | float | Confidence filter used (0.25) |
| `image_metadata.img_shape` | list | [height, width] of image |
| `image_metadata.overlay_path` | str | Relative path to overlay image |

---

## Overlay Visualizations

Each sample generates an annotated JPG overlay (`outputs/overlays/{sample_id}_overlay.jpg`) showing:

- **Lime green box** (thicker): Largest solar panel detected (best panel)
- **Cyan boxes**: Other detected panels
- **Confidence scores**: Model confidence displayed on each panel
- **Panel areas**: Estimated area in m² for each panel
- **Total statistics**: Sum of all panel areas, average confidence, panel count
- **Metadata panel**: Sample ID, coordinates, QC status, source
- **Legend**: Interpretation guide for colors and metrics

### Example Overlays

- **VERIFIABLE + Multiple Panels**: Multiple cyan boxes, one lime green box (largest), clear detections, high sharpness
- **VERIFIABLE + Single Panel**: One lime green box, clear detection
- **NOT_VERIFIABLE**: Orange box, possible shadow/cloud cover, note in QC status

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
matplotlib>=3.5.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Model not found
```
Error: models/solar_model_best.pt not found
```
→ [Train in Colab](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing) and download weights

### CSV not found
```
Error: data/raw/EI_train_data.csv not found
```
→ Ensure CSV has columns: `sampleid, latitude, longitude, hassolar`

### Images not found
```
Warning: data/processed/google_images_all/1234.jpg not found
```
→ Run `python src/download_google_staticmaps.py` to download Google tiles first

### Out of memory (multiprocessing)
```
MemoryError: Process pool exceeded limits
```
→ Reduce worker count: `python src/batch_inference.py` (uses fewer workers)

### Slow batch processing
→ Check CPU usage; increase workers if available

---

## References

- **🎓 [Training Notebook (Google Colab)](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)** ⭐
- **🐳 [Docker Hub Repository](https://hub.docker.com/r/dheerajjk/rooftop-solar-cpu)** 🐳
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Roboflow Universe**: https://universe.roboflow.com/
- **Web Mercator Projection**: https://en.wikipedia.org/wiki/Web_Mercator_projection

---

## License

See LICENSE file.

---

## Contact & Support

For issues or questions, open a GitHub issue or reach out to the maintainer.

---

## Version History

| Version | Date | Changes |
|---------|------|---------| 
| 2.3 | 2025-12-14 | Added Docker Hub integration and deployment instructions |
| 2.2 | 2025-12-07 | Added multi-panel detection, lime green highlighting for largest panel, merged batch_inference.py |
| 2.1 | 2025-12-07 | Added prominent Colab notebook links |
| 2.0 | 2025-12-07 | Added batch_pipeline.py with multiprocessing, overlay generation, CSV export |
| 1.0 | 2025-12-05 | Initial single-sample inference |

---

**Last Updated**: 2025-12-14  
**Docker Image**: `dheerajjk/rooftop-solar-cpu:latest`  
**Docker Hub**: https://hub.docker.com/r/dheerajjk/rooftop-solar-cpu  
**Training Colab**: https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing
