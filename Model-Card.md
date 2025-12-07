# Model Card: Rooftop Solar Panel Detector (YOLOv8)

## 🎓 Quick Access

### **⭐ [TRAIN IN GOOGLE COLAB (CLICK HERE)](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)**

**Direct Link**: https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing

- Free GPU access (A100/T4)
- 30–60 minutes training time
- Auto-download weights after completion

---

## 1. Model Details

### Model Overview

- **Name**: YOLOv8 Rooftop Solar Panel Detector
- **Architecture**: YOLOv8 Nano (YOLOv8n)
- **Task**: Object Detection (Solar Panels)
- **Framework**: Ultralytics YOLOv8
- **Input Size**: 640×640 px (3-channel RGB)
- **Training Environment**: Google Colab with GPU (A100/T4)
- **Inference**: Batch processing via Python multiprocessing with multi-panel detection
- **Training Notebook**: [https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)

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
- Batch processing of satellite tiles for large-scale analysis.
- **Multi-panel detection** per image with area calculations.

### Suitable Applications

- Solar installation planning.
- Energy assessment studies.
- Urban climate/sustainability research.
- Automated rooftop surveying.
- Regional solar adoption studies (500–5,000+ samples).
- Multi-panel rooftop analysis.

### Out-of-Scope Uses

- Not intended for real-time video streams.
- Not validated for very-high-resolution (>50 cm/px) or very-low-resolution (<20 cm/px) imagery.
- Not suitable for indoor imagery or non-aerial views.
- Not for medical/security/surveillance purposes.

---

## 3. Model Performance

### Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **mAP@0.5** | ~0.68–0.72 | Moderate; 640 images is modest dataset |
| **Precision** | ~0.70 | Low false positives |
| **Recall** | ~0.65 | Some missed detections on edge cases |
| **Inference Speed** | ~20–30 ms/img | On GPU; ~50 ms on CPU |
| **Batch Throughput** | 4–8 samples/sec | On 8-core system with 7 workers + GPU |
| **Multi-Panel Detection** | ✅ Supported | All panels detected, largest highlighted |

### Limitations

- **Small dataset**: 640 images is below ideal (typically 2,000+). Model may generalize poorly to unseen geographies or panel types.
- **Single class**: Only detects solar panels, not other rooftop objects (HVAC, skylights, etc.).
- **Partial panels**: Panels at tile edges may be truncated and harder to detect.
- **Weather artifacts**: Strong shadows or cloud cover can reduce accuracy (handled by QC module).
- **Geographic bias**: Dataset skewed toward developed regions; may underperform in understudied areas.

---

## 4. Inference Pipeline

### Core Algorithm: Two-Stage Buffer Detection with Multi-Panel Support

```
For each tile (sample_id):
  1. Load Google Static Maps image (640×640 px, ~19 zoom)
  2. Run YOLOv8 inference
  3. For each detected panel:
     - Convert bbox to normalized coordinates
     - Compute intersection with 1200 sq ft buffer
     - If overlap > 0:
       * Calculate panel area inside buffer
       * Add to valid panels list (panels_in_buffer)
  4. Select panel with max inside_area (best_panel_id)
  5. If no solar found at 1200 sq ft:
     - Retry with 2400 sq ft buffer
     - Use result only if solar detected
  6. Output record with:
     - buffer_radius_sqft, pv_area_sqm_est (best panel)
     - panels_in_buffer (all panels with areas)
     - best_panel_id (largest panel ID)
  7. Compute QC status (sharpness, darkness, confidence)
  8. Generate overlay visualization:
     - Lime green box for best_panel_id (largest area)
     - Cyan boxes for other panels
     - Labels with confidence and area for each panel
```

### Key Design Decisions

1. **Two buffers**: Improves recall for edge cases without increasing false positives.
2. **Area intersection**: Counts only panel area inside buffer; accurate for boundary cases.
3. **Max-area selection**: Among multiple overlapping panels, chooses the largest inside buffer.
4. **Multi-panel tracking**: All panels stored in `panels_in_buffer` for visualization and analysis.
5. **Partial panels**: Fully supported; area calculated proportionally.
6. **Parallel batch processing**: Uses multiprocessing for 3–5x speedup on multi-core systems.
7. **Visual highlighting**: Largest panel highlighted in lime green for easy identification.

### Batch Processing Pipeline

```python
# Parallel inference with multiprocessing
workers = cpu_count() - 1  # Auto-detect

with Pool(processes=workers) as pool:
    results = pool.imap_unordered(process_single_sample, samples, chunksize=2)
    for result in results:
        predictions.append(result)

# Outputs
- solar_rooftops_google.json (intermediate, untyped)
- predictions.json (final, type-cast + validated)
- predictions.csv (CSV export)
- overlays/{sample_id}_overlay.jpg (visualizations with multi-panel detection)
```

### Multi-Panel Visualization

Each overlay image includes:
- **Title**: "Solar Panel Detection - PANELS DETECTED" (when panels found)
- **Legend**: 
  - Lime = Best Panel (Highest Overlap)
  - Cyan = Other Panels
  - Area = Intersection with Buffer
  - Confidence = Model Detection Score
- **Panel Boxes**:
  - **Lime green** (thicker line, 3px) = Largest panel (best_panel_id)
  - **Cyan** (2px) = Other detected panels
  - Labels: "Panel X | Conf: Y | Area: Z m²"
- **Information Box**:
  - Sample ID, Lat/Lon (formatted)
  - has_solar: YES/NO
  - Total Area (sum of all panels)
  - Confidence (average)
  - Panels Found (count)
  - QC Status, Buffer size

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
- **Climate bias**: Temperate regions over-represented; tropical/arctic regions under-represented.

### Bias Mitigation

- Geographic stratification recommended in future retraining.
- Augmentation (mosaic, color jitter) reduces reliance on dataset-specific patterns.
- Regular validation on holdout regional datasets advised.
- QC module flags low-confidence/poor-quality images for manual review.

---

## 6. Ethical Considerations

### Privacy

- Model operates on publicly available Google Static Maps satellite imagery.
- No personally identifiable information extracted or stored.
- Applicable privacy laws (GDPR, CCPA, etc.) should be observed when collecting/sharing metadata.
- Consider local regulations on satellite imagery and property mapping.

### Fairness

- Solar adoption mapping could inadvertently expose socioeconomic disparities.
- Use responsibly in vulnerable communities; ensure stakeholder engagement.
- Avoid using results to discriminate in energy pricing or policy.

### Environmental Impact

- Training in Google Colab; GPU energy minimal (one-time, ~2 kWh).
- Inference is lightweight and distributed; modest compute footprint.
- Parallel batch processing improves efficiency over serial processing.
- Consider carbon footprint of cloud infrastructure for large-scale deployments.

### Transparency

- Model is open-source; weights, architecture, and training data publicly available.
- Model Card documents intended use, limitations, and known biases.
- Users encouraged to report findings transparently.

---

## 7. Model Versions & Updates

| Version | Date | Changes | Status | Colab |
|---------|------|---------|--------|-------|
| 2.2 | 2025-12-07 | Added multi-panel detection, lime green highlighting, merged batch_inference.py | ✅ Current | [Link](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing) |
| 2.1 | 2025-12-07 | Added prominent Colab links | ✅ | [Link](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing) |
| 2.0 | 2025-12-07 | Added batch_pipeline.py, multiprocessing, overlay generation, CSV export | ✅ | [Link](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing) |
| 1.0 | 2025-12-05 | Initial YOLOv8n model | Deprecated | N/A |
| — | Future | Larger dataset (1000+ images) | Planned | TBD |
| — | Future | YOLOv8s / YOLOv8m variants | Planned | TBD |

---

## 8. How to Use This Model

### Step 1: Train in Colab (FIRST TIME ONLY)

**[👉 CLICK TO OPEN COLAB NOTEBOOK](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)**

- No installation needed
- Runs in browser
- Auto-downloads model weights after training

### Step 2: Deploy (Batch Mode - RECOMMENDED)

1. Download `best.pt` from Colab training.
2. Place in `models/solar_model_best.pt`.
3. Ensure `data/raw/EI_train_data.csv` and image tiles exist.
4. Run batch pipeline:
   ```bash
   python src/batch_inference.py
   ```

### Step 3: Deploy (Single Sample)

```python
from src.inference import SolarPanelInference

inf = SolarPanelInference()
record = inf.predict(sample_id=1234, lat=12.9716, lon=77.5946)
print(record)
# Output includes: panels_in_buffer, best_panel_id, etc.
```

### Custom Images

```python
from ultralytics import YOLO

model = YOLO('models/solar_model_best.pt')
results = model('path/to/image.jpg', conf=0.25)

for result in results:
    print(f"Detections: {len(result.boxes)}")
    for box in result.boxes:
        print(f"  Confidence: {box.conf[0]:.2f}")
```

### Docker / Containerized Inference

```bash
docker build -t rooftop-solar .
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/outputs:/app/outputs \
           rooftop-solar \
           python src/batch_inference.py
```

---

## 9. Evaluation & Monitoring

### Recommended Tests

- **Holdout validation**: Evaluate on geographic regions not in training set.
- **Seasonal variation**: Test on different seasons to assess shadow/lighting robustness.
- **Panel diversity**: Test on rare panel orientations, colors, or technologies.
- **Scale testing**: Validate on 100, 1000, 5000+ sample batches.
- **Edge cases**: Verify behavior on partially visible, occluded, or heavily shadowed panels.
- **Multi-panel scenarios**: Test on rooftops with multiple panel installations.

### Monitoring in Production

- Track `qc_status` distribution (should be mostly "VERIFIABLE").
- Monitor `confidence` scores; sudden drops may indicate data drift.
- Log batch processing time; watch for slowdowns or crashes.
- Periodically sample predictions + overlays for manual review.
- Track false positive / false negative rates on test set.
- Monitor multi-panel detection accuracy (count vs ground truth).

### Performance Metrics to Log

```json
{
  "batch_size": 3000,
  "processing_time_minutes": 45,
  "avg_confidence": 0.54,
  "verifiable_rate": 0.82,
  "solar_detected_rate": 0.31,
  "avg_panels_per_detection": 1.8,
  "multi_panel_rate": 0.25,
  "errors": 0
}
```

---

## 10. Maintenance & Retraining

### When to Retrain

- Dataset grows to 1,000+ labeled images.
- Performance drops on validation set > 5%.
- New panel technologies or orientations emerge.
- Geographic coverage expands to new regions.
- QC status < 75% on recent batches (indicates data drift).
- Multi-panel detection accuracy degrades.

### Retraining Steps (in Colab)

1. [Open Colab Notebook](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)
2. Expand dataset using open sources (Roboflow, research datasets).
3. Run training with new dataset.
4. Validate on holdout set.
5. Compare metrics to v2.2 baseline.
6. Replace `models/solar_model_best.pt` and tag version (e.g., v2.3).

### Version Control

- Tag releases: `git tag -a v2.2 -m "Multi-panel detection"`
- Document performance metrics for each version.
- Maintain backward compatibility for inference API.

---

## 11. Output Artifacts

### Generated Files

| File | Format | Description |
|------|--------|-------------|
| `predictions.json` | JSON | Final predictions, type-cast, schema-validated, includes panels_in_buffer |
| `predictions.csv` | CSV | Flat table of predictions (for spreadsheet analysis) |
| `predictions_final.json` | JSON | Final submission format |
| `solar_rooftops_google.json` | JSON | Intermediate results (untyped) |
| `overlays/{sample_id}_overlay.jpg` | JPG | Annotated detection visualization with multi-panel highlighting |

### Overlay Format

Each overlay JPG shows:
- **Original image** from Google Static Maps
- **Bounding boxes**:
  - **Lime green** (thicker) = Largest panel (best_panel_id)
  - **Cyan** = Other detected panels
- **Confidence scores** overlaid on each box
- **Panel areas** (m²) displayed for each panel
- **Total statistics**: Sum of areas, average confidence, panel count
- **Metadata panel** (bottom): sample ID, lat/lon, QC status, image source
- **Legend** (top): interpretation guide

### Output Schema

```json
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
}
```

---

## 12. Google Colab Training Notebook

### **[👉 OPEN COLAB NOTEBOOK HERE 👈](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)**

**Direct Link**: https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing

### Setup

- **Runtime Required**: GPU (A100 or T4 recommended)
- **Training Time**: ~30–60 minutes for 50 epochs on ~640 images
- **Cost**: Free tier (15 GPU hours/month)
- **Steps**: Mount Drive → Install YOLO → Prepare dataset → Train → Export weights

### What You'll Get

✅ Trained YOLOv8n model weights  
✅ Training metrics and plots  
✅ Model ready for download  
✅ Step-by-step documentation  

---

## 13. FAQ

**Q: Why use YOLOv8n instead of larger models?**  
A: YOLOv8n balances speed and accuracy. Inference ~20–30 ms/image allows parallel batch processing. Larger models (YOLOv8m, YOLOv8l) are slower but slightly more accurate; available for future versions.

**Q: How are multiple panels handled in one image?**  
A: Algorithm detects **ALL panels** and stores them in `panels_in_buffer`. The panel with **maximum area inside buffer** is selected as `best_panel_id` and highlighted in lime green. Other panels are displayed in cyan. Total area is the sum of all panel areas.

**Q: What does "NOT_VERIFIABLE" mean?**  
A: Image has low sharpness, high darkness, or low model confidence. Results marked NOT_VERIFIABLE should be manually reviewed.

**Q: Can I use this on my own imagery?**  
A: Yes, if images are 640×640 RGB tiles with ~19 zoom level (similar to Google Static Maps). Adjust buffer radius and area calculations for different zoom levels.

**Q: How long does batch processing take?**  
A: ~0.5–2 sec/sample on GPU with 7 workers. 3,000 samples = ~25–50 minutes. CPU-only is 2–3x slower.

**Q: Is the model commercial-ready?**  
A: Model is trained on open data and is research-grade. Validate thoroughly on your specific use case before commercial deployment.

**Q: Do I need to run Colab training locally?**  
A: No! Colab runs in your browser with GPU. Download trained weights and use locally.

**Q: How does multi-panel visualization work?**  
A: All detected panels are drawn on the overlay. The largest panel (by area inside buffer) is highlighted in **lime green** with a thicker line (3px). Other panels are shown in **cyan** (2px). Each panel has a label showing Panel ID, Confidence, and Area.

---

## 14. Contact & Attribution

- **Model Author**: AI/ML Research Team
- **Training Framework**: Ultralytics YOLOv8
- **Dataset Source**: Roboflow Universe
- **Batch Pipeline**: Custom Python multiprocessing implementation with multi-panel detection
- **Training Platform**: Google Colab
- **For questions**: Open GitHub issues or contact maintainer
- **Citation**: Please cite this Model Card and Roboflow dataset if using in research

---

## 15. License & Disclaimer

- Model and code released under [LICENSE](LICENSE)
- No warranty; use at your own risk
- Validate predictions on your data before production use
- Respect Google Static Maps ToS and local privacy regulations

---

**Last Updated**: 2025-12-07  
**Version**: 2.2  
**Colab Notebook**: [https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing](https://colab.research.google.com/drive/1Cl9KowI1deMolhE3wjfg165TRbDsuX4-?usp=sharing)
