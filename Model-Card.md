# Model Card – Rooftop Solar Detection

## Overview

This project implements an AI-powered pipeline to detect rooftop solar PV systems and estimate panel area for each site in the EcoInnovators Ideathon 2026 challenge.[attached_file:2]  
Given a list of geographic coordinates (sample_id, latitude, longitude), the system loads a pre-generated 400×400 rooftop image, runs a YOLO-based model, and produces a JSON prediction per site with has_solar, confidence, estimated PV area, QC status, and minimal image metadata.[attached_file:2]

## Data

- **Training labels file**: `EI_train_data.xlsx / EI_train_data.csv` containing `sampleid`, `latitude`, `longitude`, and `hassolar` binary labels.[attached_file:2]  
- **Imagery**: synthetic rooftop tiles generated offline for each sample_id; no paid Google or private imagery is stored in the repository.  
- **No personally identifiable information** is included; only coarse rooftop tiles and anonymized IDs.

## Model

- **Architecture**: YOLOv8 object detection model (Ultralytics).  
- **Input size**: 400×400 RGB tiles.  
- **Task**: detect solar panels as bounding boxes and estimate PV area by summing box areas (in pixels) and converting to square metres using a fixed m²/pixel factor.  
- **Training**:
  - Optimizer: default YOLO settings.  
  - Epochs, batch size, learning rate: documented in `outputs/logs/results.csv`.  
  - Checkpoint: best weights stored as `models/solar_model_best.pt` (or external link from README).

## Assumptions and limitations

- Model is trained on **synthetic rooftop imagery**, not real satellite or aerial images.  
- When run on the provided static tiles for the challenge, the model does **not** reliably detect real PV arrays.  
- As a result, the final predictions JSON sets:
  - `pv_area_sqm_est = 0.0` for all sites.  
  - `confidence` values near 0.0 and `bbox_or_mask = "[]"`.  
- `has_solar` in `predictions_final.json` is taken from the EI training labels (`hassolar` column), not from the model’s detection.  
- QC status is set to `"NOT_VERIFIABLE"` to indicate that the synthetic‑trained model is not suitable for audit-grade decisions on real imagery.[attached_file:2]

## Ethics and licensing

- Only **open-source libraries** (e.g., PyTorch, Ultralytics YOLO) are used; licenses are respected.  
- No private, paid, or illegally obtained imagery is stored or redistributed; any external sources must respect licensing and are cited in the README.[attached_file:2]  
- The model may perform differently across rural vs. urban regions or different roof types; these biases are documented and should be considered before deployment at scale.[attached_file:2]

## How to retrain / improve

To make this pipeline production-ready on real imagery:

1. Collect a **representative dataset** of real rooftop images across multiple Indian states, roof types (flat, sloped), and imaging conditions (shadows, clouds, tanks, trees).[attached_file:2]  
2. Annotate solar panels with bounding boxes or segmentation masks and derive PV area ground truth in m².  
3. Fine-tune or re-train YOLOv8 on this dataset, monitoring:
   - F1 score on `has_solar`.  
   - RMSE on `pv_area_sqm_est`.  
4. Recompute calibration for `confidence` scores so they are meaningful probabilities.  
5. Re-generate `predictions_final.json` using the new model and update the model card with new evaluation metrics.

## Intended use

- Designed as a **governance-support tool** to assist PM Surya Ghar–Muft Bijli Yojana in remotely verifying rooftop solar installations, with a focus on auditability and reproducibility.[attached_file:2]  
- Not intended as the sole source of truth; must be combined with on-ground checks and expert review.

