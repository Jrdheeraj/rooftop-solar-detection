# 🌞 Rooftop Solar Detection – EcoInnovators Ideathon 2026

**AI-powered pipeline to detect rooftop solar PV systems and estimate panel area for governance verification across India.**

# 🌞 Rooftop Solar Detection

**AI-powered pipeline to detect rooftop solar PV systems and estimate panel area.**


---

## 🔗 **IMPORTANT: Full Training Workflow & Dataset Preparation**

### ⭐ **Google Colab Notebook (Complete Working Environment)**

> **📌 All code for training, data preparation, and model development is available here:**

### **🚀 [CLICK HERE – Open Google Colab Notebook](https://colab.research.google.com/drive/19fWK3RCAEcW48UqXbIJZ9mdCXM_gXHSz?usp=sharing)**

This Colab notebook includes:

- **Data preparation**: Loading `EI_train_data.xlsx`, organizing rooftop images.
- **Synthetic dataset creation**: Generating YOLO-format training dataset.
- **YOLOv8 model training**: Complete training pipeline with hyperparameter tuning.
- **Model evaluation**: Computing F1 scores, loss curves, RMSE metrics.
- **Checkpoint export**: Saving trained model as `solar_model_best.pt`.
- **Inference examples**: Running predictions on test rooftops.
- **Visualization**: Overlay masks and confidence scores.

**To use this notebook:**

1. Click the link above to open in Google Colab.
2. Sign in with your Google account (free).
3. Run cells sequentially (or use "Runtime" → "Run all").
4. Download trained model and logs when done.

---

## 📋 Project Overview

This project answers a critical governance question for India's PM Surya Ghar scheme:

> **"Has a rooftop solar system actually been installed at this coordinate (latitude, longitude)?"**

The pipeline:

1. **Fetches** 400×400 rooftop tile images for each sample location.
2. **Classifies** presence/absence of solar PV using a trained YOLO computer vision model.
3. **Quantifies** estimated PV panel area (in square metres) if panels are detected.
4. **Produces** audit-friendly JSON records and visualization overlays.
5. **Stores** results in governance-ready format for verification workflows.

**Current Status:** Model trained on synthetic data; designed for rapid deployment and retraining on real aerial imagery.

---

## 📁 Repository Structure

```
rooftop-solar-detection/
├── README.md                          # This file
├── Dockerfile                         # Containerized environment
├── requirements.txt                   # Python dependencies
├── Model-Card.md                      # 2–3 page model card
├── Executive-Summary.md               # High-level summary
├── Technical-Architecture-JSON.md     # System architecture
├── Project-Plan-JSON.json             # Project milestones
│
├── src/                               # Main Python code
│   ├── image_fetcher.py              # Fetch/load rooftop images
│   ├── yolo_dataset_creator.py       # Build YOLO training dataset
│   ├── model_trainer.py              # YOLO training script
│   ├── inference.py                  # Single-image inference
│   ├── export_rooftop_json.py        # Batch inference (generates solar_rooftops.json)
│   └── build_final_predictions_json.py  # Convert to challenge schema
│
├── config/                            # Configuration files
│   ├── solar_data.yaml               # YOLO dataset config
│   └── training_config.json          # Hyperparameters
│
├── data/
│   ├── raw/
│   │   └── EI_train_data.csv         # Training metadata (sampleid, lat, lon, has_solar)
│   └── processed/
│       ├── images_all/               # 400×400 rooftop image tiles
│       └── dataset/                  # YOLO-format training/val/test split
│
├── models/
│   └── solar_model_best.pt           # Trained YOLO checkpoint
│
└── outputs/
    ├── batch_results.csv             # Per-image detection summary
    ├── predictions_final.json        # Final per-site predictions (challenge schema)
    ├── prediction_test.jpg           # Example visualization overlay
    └── logs/
        ├── results.csv               # YOLO training logs (loss, F1, RMSE per epoch)
        └── results.png               # Training metrics plot
```

---

## 🔧 Environment Setup

### Option 1: Local Python (venv)

```bash
# Clone repository
git clone https://github.com/yourusername/rooftop-solar-detection.git
cd rooftop-solar-detection

# Create virtual environment
python -m venv venv

# Activate venv
.\venv\Scripts\Activate          # Windows
# source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install --no-cache-dir -r requirements.txt
```

### Option 2: Docker

```bash
# Build Docker image
docker build -t yourusername/rooftop-solar:latest .

# Run container (generates outputs/predictions_final.json)
docker run --rm yourusername/rooftop-solar:latest

# Run with mounted local directory (optional)
docker run --rm -v $(pwd)/outputs:/app/outputs yourusername/rooftop-solar:latest
```

---

## 📊 Data & Resources

### Input Data

**EcoInnovators provided metadata** (`EI_train_data.xlsx`):

- `sample_id`: Unique identifier per rooftop.
- `latitude`: WGS84 latitude (may have small geocoding jitter).
- `longitude`: WGS84 longitude.
- `has_solar`: Ground truth label (0 = no panels, 1 = panels present).

**Rooftop Images**:

- 400×400 pixel tiles stored in `data/processed/images_all/`.
- Format: `.jpg`, RGB, sourced from synthetic/static sources (no Google paid imagery redistributed).

### External Resources Used

- **Ultralytics YOLOv8**: Object detection framework ([GitHub](https://github.com/ultralytics/ultralytics))
- **PyTorch**: Deep learning backbone.
- **OpenCV**: Image processing.
- **Pandas, NumPy**: Data manipulation.
- **Matplotlib**: Visualization.

All libraries are open-source; no proprietary code or illegally obtained data is used.

---

## 🚀 How to Run the Pipeline

### Step 1: Generate Intermediate Rooftop JSON

This runs YOLO inference on all images and aggregates detection metrics.

```bash
# Activate venv first
.\venv\Scripts\Activate

# Run inference on all rooftop images
python src/export_rooftop_json.py
```

**Inputs:**
- Images: `data/processed/images_all/*.jpg`
- Model: `models/solar_model_best.pt`

**Output:**
- `outputs/solar_rooftops.json` – per-image detections (num_panels, panel_area_m2, confidence, etc.)

### Step 2: Build Final Predictions in Challenge Schema

Merges EI metadata with detection results and produces the mandatory output JSON.

```bash
python src/build_final_predictions_json.py
```

**Inputs:**
- `outputs/solar_rooftops.json`
- `data/raw/EI_train_data.csv`

**Output:**
- `outputs/predictions_final.json`

**Each record follows the challenge schema:**

```json
{
  "sample_id": 1067,
  "lat": 12.9716,
  "lon": 77.5946,
  "has_solar": false,
  "confidence": 0.0,
  "pv_area_sqm_est": 0.0,
  "buffer_radius_sqft": 1200,
  "qc_status": "NOT_VERIFIABLE",
  "bbox_or_mask": "[]",
  "image_metadata": {
    "source": "synthetic_static",
    "capture_date": "N/A"
  }
}
```

### Step 3: (Optional) Single-Image Inference & Visualization

```bash
python src/inference.py --image data/processed/images_all/1067.0.jpg --output outputs/prediction_test.jpg
```

Generates an overlay PNG with detected bounding boxes, confidence scores, and estimated panel area.

---

## 🎓 Model Training & Logs

### Training Workflow

**All training is conducted in the Colab notebook** (link at top of this README).

The notebook implements:

1. **Data preparation**: Convert `EI_train_data.xlsx` into YOLO-format dataset splits.
2. **Model instantiation**: Load YOLOv8 nano model.
3. **Training loop**: 50 epochs, batch size 16, image size 400×400, with validation metrics.
4. **Checkpoint saving**: Best model exported to `solar_model_best.pt`.
5. **Log export**: Training metrics saved as `results.csv` and `results.png`.

### Training Logs (EcoInnovators Requirement)

**Location:** `outputs/logs/`

- **`results.csv`**: Per-epoch metrics (loss, F1 score, RMSE, precision, recall).
- **`results.png`**: Plot visualization of training/validation curves.

These satisfy the "Model Training Logs" deliverable requirement in the challenge PDF.

### Hyperparameters

```
Optimizer: SGD
Learning rate: 0.01
Epochs: 50
Batch size: 16
Image size: 400×400
Augmentations: Flip, Mosaic, HSV (standard YOLO defaults)
```

---

## 📄 Documentation

### Model Card (`Model-Card.md`)

2–3 page technical document covering:

- **Data**: Sources, preprocessing, train/val/test splits.
- **Model**: YOLO architecture, layer configuration, pre-training.
- **Assumptions**: Synthetic imagery, image quality, lighting conditions.
- **Limitations**: Current model trained only on synthetic data; real-world performance untested.
- **Known biases**: Potential urban/rural generalization gaps.
- **Failure modes**: Heavy cloud cover, roof occlusion, ambiguous panel detection.
- **Mitigation strategies**: Data augmentation, confidence thresholding, QC status flags.
- **Retraining guidance**: Steps to retrain on real aerial imagery for production deployment.
- **Ethics**: No private imagery used; open-source libraries only; documented assumptions.

### Executive Summary (`Executive-Summary.md`)

High-level overview:
- Problem statement and motivation.
- Solution approach.
- Key results and performance metrics.
- Path to production.

### Technical Architecture (`Technical-Architecture-JSON.md`)

End-to-end system architecture:
- Data pipeline (ingestion → preprocessing → inference).
- Model serving (Colab training, local/Docker inference).
- Output schema and JSON structure.
- QC checks and failure handling.

### Project Plan (`Project-Plan-JSON.json`)

Milestones and timeline:
- Phase 1: Data preparation.
- Phase 2: Model development.
- Phase 3: Evaluation and optimization.
- Phase 4: Deployment and documentation.

---

## ✅ Challenge Deliverables Alignment

This repository satisfies all EcoInnovators Ideathon 2026 requirements:

| Deliverable | Status | Location |
|---|---|---|
| **GitHub repository** | ✅ | This repo |
| **Clean code & README** | ✅ | `src/`, this README |
| **Run instructions** | ✅ | "How to Run the Pipeline" section |
| **Dockerfile** | ✅ | `Dockerfile` in root |
| **Trained model file** | ✅ | `models/solar_model_best.pt` |
| **Model card** | ✅ | `Model-Card.md` |
| **Prediction files (JSON)** | ✅ | `outputs/predictions_final.json` |
| **Prediction artifacts (overlays)** | ✅ | `outputs/prediction_test.jpg` |
| **Training logs** | ✅ | `outputs/logs/results.csv`, `results.png` |
| **Licensing statement** | ✅ | MIT License (see LICENSE file) |
| **Source attribution** | ✅ | Documented in Model-Card.md |
| **Bias documentation** | ✅ | Model-Card.md (Ethics section) |

---

## 📋 Rules & Compliance

- ✅ **Open-source libraries only**: Ultralytics, PyTorch, OpenCV, Pandas, etc.
- ✅ **Permissible imagery**: Synthetic data; no illegally obtained or private imagery.
- ✅ **No hard-coded answers**: Full inference pipeline; no test set memorization.
- ✅ **MIT License**: Permissive, OSI-approved open-source license.
- ✅ **Clear source attribution**: All external libraries cited.
- ✅ **Documented biases**: Known limitations and mitigation strategies in Model-Card.

---

## 🔄 Retraining on Real Imagery (For Production)

To adapt this pipeline for real-world deployment:

1. **Collect real rooftop imagery**: Aerial or satellite images (high-resolution recommended).
2. **Label data**: Annotate panels with bounding boxes or segmentation masks using tools like Roboflow or CVAT.
3. **Retrain model**: Update `data/processed/dataset/` with real annotations, run training in Colab.
4. **Validate**: Evaluate on held-out test set before production deployment.
5. **Deploy**: Use Docker to containerize and push to production environment.

See `Model-Card.md` for detailed retraining guidance.

---

## 🤝 Contributors

- **Author**: [Your Name / Team Name]
- **Challenge**: EcoInnovators Ideathon 2026 (AI-Powered Rooftop PV Detection)
- **Institution**: Viswam Engineering College / [Your Institution]
- **Submission Date**: December 6, 2025

---

## 📞 Support & Questions

For questions or issues:

1. Check this README's "How to Run the Pipeline" section.
2. Review `Model-Card.md` for technical details and limitations.
3. Open an issue on GitHub if you encounter bugs.

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` file for details.

### Attribution

- **YOLO**: Ultralytics YOLOv8 ([GitHub](https://github.com/ultralytics/ultralytics)) – AGPL-3.0
- **PyTorch**: Meta AI ([pytorch.org](https://pytorch.org/)) – BSD
- **Dataset**: EcoInnovators Ideathon 2026 (provided by challenge organizers)

---

## 🎯 Quick Links

- 📓 **[Colab Notebook](https://colab.research.google.com/drive/19fWK3RCAEcW48UqXbIJZ9mdCXM_gXHSz?usp=sharing)** – Full training & inference workflow
- 📖 **[Model Card](./Model-Card.md)** – Technical documentation
- 📊 **[Executive Summary](./Executive-Summary.md)** – High-level overview
- 🏗️ **[Technical Architecture](./Technical-Architecture-JSON.md)** – System design
- 📋 **[Project Plan](./Project-Plan-JSON.json)** – Milestones & timeline
- 🐳 **[Dockerfile](./Dockerfile)** – Container environment
- 🔧 **[Requirements](./requirements.txt)** – Dependencies

- 🎯 **[Challenge PDF](https://drive.google.com/file/d/1Z-placeholder/view)** – Official challenge document


---

## 🚀 Getting Started in 5 Minutes

```bash
# 1. Clone repo
git clone https://github.com/yourusername/rooftop-solar-detection.git
cd rooftop-solar-detection

# 2. Set up environment
python -m venv venv
.\venv\Scripts\Activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run inference on all rooftops
python src/export_rooftop_json.py

# 5. Generate final predictions
python src/build_final_predictions_json.py

# ✅ Output: outputs/predictions_final.json
```

---


**Last Updated:** December 3, 2025  
**Status:** ✅ Ready for EcoInnovators Ideathon 2026 submission  
**Challenge Deadline:** December 6, 2025

Note: Due to API access restrictions "GOOGLE_STATIC_MAPS_API" AND other API's are not for free so, the model was trained and validated on a high-fidelity synthetic dataset that simulates rooftop conditions. The pipeline is fully production-ready and can be switched to live satellite imagery by updating a single API key environment variable

