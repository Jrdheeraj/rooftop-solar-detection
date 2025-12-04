# Model Card: Rooftop Solar Panel Detector (YOLOv8)

## 1. Model Details

### Model Overview

- **Name**: YOLOv8 Rooftop Solar Panel Detector
- **Architecture**: YOLOv8 Nano (YOLOv8n)
- **Task**: Object Detection (Solar Panels)
- **Framework**: Ultralytics YOLOv8
- **Input Size**: 640×640 px (3-channel RGB)
- **Training Environment**: Google Colab with GPU (A100/T4)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model Base | yolov8n.pt |
| Epochs | 50 |
| Batch Size | 16 |
| Learning Rate | 0.01 (default) |
| Image Size | 640 |
| Optimizer | SGD |
| Augmentation | YOLOv8 default (mosaic, color jitter, flip, rotation) |

### Dataset

- **Source**: Roboflow (open rooftop-solar dataset)
- **Format**: YOLOv8 format (images + YOLO .txt annotations)
- **Train Split**: ~550 images
- **Val Split**: ~90 images
- **Classes**: 1 (solar_panel)
- **Total Labeled Panels**: ~1,200+ annotations

---

## 2. Intended Use

### Primary Use Case

Detect and measure solar panels on residential and commercial rooftops from overhead satellite/aerial imagery. Supports:
- Solar energy potential assessment.
- Rooftop coverage analysis.
- Urban solar adoption mapping.

### Suitable Applications

- Solar installation planning.
- Energy assessment studies.
- Urban climate/sustainability research.
- Automated rooftop surveying.

### Out-of-Scope Uses

- Not intended for real-time video streams.
- Not validated for very-high-resolution (>50 cm/px) or very-low-resolution (<20 cm/px) imagery.
- Not suitable for indoor imagery or non-aerial views.

---

## 3. Model Performance

### Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **mAP@0.5** | ~0.68–0.72 | Moderate; 640 images is modest dataset |
| **Precision** | ~0.70 | Low false positives |
| **Recall** | ~0.65 | Some missed detections on edge cases |
| **Inference Speed** | ~20–30 ms/img | On GPU; sufficient for batch processing |

### Limitations

- **Small dataset**: 640 images is below ideal (typically 2,000+). Model may generalize poorly to unseen geographies or panel types.
- **Single class**: Only detects solar panels, not other rooftop objects (HVAC, skylights, etc.).
- **Partial panels**: Panels at tile edges may be truncated and harder to detect.
- **Weather artifacts**: Strong shadows or cloud cover can reduce accuracy (handled by QC module).

---

## 4. Inference Pipeline

### Core Algorithm: Two-Stage Buffer Detection

```
For each tile (sample_id):
  1. Load Google Static Maps image (640×640 px, ~19 zoom)
  2. Run YOLOv8 inference
  3. For each detected panel:
     - Convert bbox to normalized coordinates
     - Compute intersection with 1200 sq ft buffer
     - If overlap > 0:
       * Calculate panel area inside buffer
       * Add to valid panels list
  4. Select panel with max inside_area
  5. If no solar found at 1200 sq ft:
     - Retry with 2400 sq ft buffer
     - Use result only if solar detected
  6. Output record with buffer_radius_sqft, pv_area_sqm_est
  7. Compute QC status (sharpness, darkness, confidence)
```

### Key Design Decisions

1. **Two buffers**: Improves recall for edge cases without increasing false positives.
2. **Area intersection**: Counts only panel area inside buffer; accurate for boundary cases.
3. **Max-area selection**: Among multiple overlapping panels, chooses the largest inside buffer.
4. **Partial panels**: Fully supported; area calculated proportionally.

---

## 5. Training Data & Bias

### Data Source

- **Roboflow Universe** (open collection)
- **Geographic diversity**: Panels from multiple countries/regions
- **Diversity**: Various roof types, orientations, colors, installation angles

### Known Biases

- **Geography bias**: Dataset skewed toward developed regions; may underperform in understudied areas.
- **Panel type bias**: Mostly rectangular black/blue panels; rare/exotic designs less tested.
- **Scale bias**: Small and very large panels may have detection rate variations.
- **Lighting bias**: Daytime imagery only; no night/cloudy performance validation.

### Bias Mitigation

- Geographic stratification recommended in future retraining.
- Augmentation (mosaic, color jitter) reduces reliance on dataset-specific patterns.
- Regular validation on holdout regional datasets advised.

---

## 6. Ethical Considerations

### Privacy

- Model operates on publicly available Google Static Maps satellite imagery.
- No personally identifiable information extracted or stored.
- Applicable privacy laws (GDPR, etc.) should be observed when collecting metadata.

### Fairness

- Solar adoption mapping could inadvertently expose socioeconomic disparities.
- Use responsibly in vulnerable communities; ensure stakeholder engagement.

### Environmental Impact

- Training in Google Colab; GPU energy minimal (one-time, ~2 kWh).
- Inference is lightweight and distributed; modest compute footprint.

---

## 7. Model Versions & Updates

| Version | Date | Changes | Notes |
|---------|------|---------|-------|
| 1.0 | 2025-12-05 | Initial YOLOv8n model | 50 epochs, Roboflow dataset |
| — | Future | Larger dataset (1000+ images) | Planned |
| — | Future | YOLOv8s / YOLOv8m | Better accuracy trade-off |

---

## 8. How to Use This Model

### Deployment

1. Download `best.pt` from training (see README Training Pipeline).
2. Place in `models/solar_model_best.pt`.
3. Run batch inference:
   ```bash
   python -m src.export_rooftop_json
   ```

### Custom Images

```python
from ultralytics import YOLO
model = YOLO('models/solar_model_best.pt')
results = model('path/to/image.jpg', conf=0.25)
```

### API / Docker

See README for containerized inference.

---

## 9. Evaluation & Monitoring

### Recommended Tests

- **Holdout validation**: Evaluate on geographic regions not in training set.
- **Seasonal variation**: Test on different seasons to assess shadow/lighting robustness.
- **Panel diversity**: Test on rare panel orientations, colors, or technologies.

### Monitoring in Production

- Track `qc_status` distribution (should be mostly "VERIFIABLE").
- Monitor `confidence` scores; sudden drops may indicate data drift.
- Periodically sample predictions for manual review.

---

## 10. Maintenance & Retraining

### When to Retrain

- Dataset grows to 1,000+ labeled images.
- Performance drops on validation set > 5%.
- New panel technologies or orientations emerge.

### Retraining Steps

1. Expand dataset using open sources (Roboflow, research datasets).
2. Run Colab training notebook with new dataset.
3. Validate on holdout set.
4. Replace `models/solar_model_best.pt` and tag version.

---

## 11. Colab Training Notebook

**Training is performed in Google Colab:**

- **Notebook URL**: https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing
- **Runtime Required**: GPU (A100 or T4 recommended)
- **Training Time**: ~30–60 minutes for 50 epochs

---

## 12. Contact & Attribution

- **Model Author**: AI/ML Research Team
- **Training Framework**: Ultralytics YOLOv8
- **Dataset Source**: Roboflow Universe
- **For questions**: Open GitHub issues or contact maintainer.