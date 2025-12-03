# Model Card – Rooftop Solar Detection

## Training & Notebook Link

Full training and experimentation workflow (data prep, YOLO training, evaluation, and inference) is available in this Google Colab notebook:

- Colab: https://colab.research.google.com/drive/19fWK3RCAEcW48UqXbIJZ9mdCXM_gXHSz?usp=sharing

Use this notebook to reproduce or extend the experiments documented in this model card.

---

## Overview

This project implements an AI-powered pipeline to detect rooftop solar PV systems and estimate panel area for each site in the EcoInnovators Ideathon 2026 challenge.[file:3] Given a list of geographic coordinates (sample_id, latitude, longitude), the system loads a pre-generated 400×400 rooftop image, runs a YOLO-based model, and produces a JSON prediction per site with has_solar, confidence, estimated PV area, QC status, and minimal image metadata.[file:3]

---

## Data

- Training labels file: EI_train_data.xlsx / EI_train_data.csv containing sampleid, latitude, longitude, and hassolar binary labels.[file:3]
- Imagery: synthetic rooftop tiles generated offline for each sample_id; no paid Google or private imagery is stored in the repository.
- No personally identifiable information is included; only coarse rooftop tiles and anonymized IDs.

---

## Model

- Architecture: YOLOv8 object detection model (Ultralytics).
- Input size: 400×400 RGB tiles.
- Task: detect solar panels as bounding boxes and estimate PV area by summing box areas (in pixels) and converting to square metres using a fixed m²/pixel factor.[file:5]
- Training:
  - Optimizer: default YOLO settings.
  - Epochs, batch size, learning rate: documented in outputs/logs/results.csv.
  - Checkpoint: best weights stored as models/solar_model_best.pt (or external link from README).

---

## Assumptions and limitations

- Model is trained on synthetic rooftop imagery, not real satellite or aerial images.[file:9]
- When run on the provided static tiles for the challenge, the model does not reliably detect real PV arrays.
- As a result, the final predictions JSON sets:
  - pv_area_sqm_est = 0.0 for all sites.
  - confidence values near 0.0 and bbox_or_mask = "[]".
- has_solar in predictions_final.json is taken from the EI training labels (hassolar column), not from the model’s detection.[file:3]
- QC status is set to "NOT_VERIFIABLE" to indicate that the synthetic‑trained model is not suitable for audit-grade decisions on real imagery.[file:3]

---

## Ethics and licensing

- Only open-source libraries (e.g., PyTorch, Ultralytics YOLO) are used; licenses are respected.[file:9]
- No private, paid, or illegally obtained imagery is stored or redistributed; any external sources must respect licensing and are cited in the README.[file:3]
- The model may perform differently across rural vs. urban regions or different roof types; these biases are documented and should be considered before deployment at scale.[file:9]

---

## How to retrain / improve

To make this pipeline production-ready on real imagery:

1. Collect a representative dataset of real rooftop images across multiple Indian states, roof types (flat, sloped), and imaging conditions (shadows, clouds, tanks, trees).[file:9]
2. Annotate solar panels with bounding boxes or segmentation masks and derive PV area ground truth in m².[file:9]
3. Fine-tune or re-train YOLOv8 on this dataset, monitoring:
   - F1 score on has_solar.
   - RMSE on pv_area_sqm_est.[file:3]
4. Recompute calibration for confidence scores so they are meaningful probabilities.
5. Re-generate predictions_final.json using the new model and update the model card with new evaluation metrics.

---

