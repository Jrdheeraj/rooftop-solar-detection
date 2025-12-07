from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json
import logging

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.inference import SolarPanelInference, load_image_for_sample

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

# Paths
CSV_PATH = Path("data/raw/EI_train_data.csv")
OUTPUT_DIR = Path("outputs")
OVERLAYS_DIR = OUTPUT_DIR / "overlays"
SOLAR_JSON_PATH = "outputs/solar_rooftops.json"
EI_CSV_PATH = "data/raw/EI_train_data.csv"
OUTPUT_JSON_PATH = "outputs/predictions_final.json"
INPUT_JSON = Path("outputs/solar_rooftops_google.json")

# Default values
DEFAULT_BUFFER_RADIUS_SQFT = 1200
QC_STATUS = "NOT_VERIFIABLE"
IMAGE_SOURCE = "synthetic_static"
CAPTURE_DATE = "N/A"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# FUNCTIONS FROM build_final_predictions_json.py
# ════════════════════════════════════════════════════════════════════════════════

def load_solar_rooftops(path: str):
    """Load solar_rooftops.json as dict keyed by image_id."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist. Run export_rooftop_json.py first.")
    with open(p, "r") as f:
        data = json.load(f)
    # index by image_id string, e.g. "1067.0"
    idx = {str(rec["image_id"]): rec for rec in data}
    return idx


def load_ei_csv(path: str):
    """
    Load EI_train_data.csv as DataFrame indexed by sampleid.
    
    CSV file has exact columns: sampleid, latitude, longitude, hassolar
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist. Put EI_train_data.csv in data/raw/.")
    df = pd.read_csv(p)
    
    # Verify exact column names from CSV file
    expected_columns = {"sampleid", "latitude", "longitude", "hassolar"}
    if not expected_columns.issubset(df.columns):
        missing = expected_columns - set(df.columns)
        raise ValueError(
            f"Expected columns {expected_columns} in EI_train_data.csv. "
            f"Missing: {missing}. Found: {set(df.columns)}"
        )
    
    df = df.set_index("sampleid")
    return df


def build_final_predictions_json():
    """Build final predictions JSON from solar rooftops and EI CSV."""
    # 1) Load inputs
    solar_idx = load_solar_rooftops(SOLAR_JSON_PATH)
    df = load_ei_csv(EI_CSV_PATH)

    final_records = []

    # 2) For each sample_id in EI data, build a prediction record
    for sample_id, row in df.iterrows():
        # image_id in solar_rooftops.json is like "1067.0"
        image_id_str = f"{float(sample_id)}"

        solar_rec = solar_idx.get(image_id_str, None)

        if solar_rec is None:
            # no entry => treat as 0 detections
            pv_area_m2 = 0.0
            confidence = 0.0
        else:
            pv_area_m2 = float(solar_rec.get("panel_area_m2", 0.0))
            confidence = float(solar_rec.get("max_confidence", 0.0))

        # ground-truth label from EI: 0/1
        has_solar_label = int(row.get("hassolar", 0))
        has_solar_bool = bool(has_solar_label)

        # coordinates
        lat = float(row.get("latitude"))
        lon = float(row.get("longitude"))

        # QC status (fixed for now)
        qc_status = QC_STATUS

        # no detections -> empty bbox list encoded as string
        bbox_or_mask = "[]"

        # 3) Build record in required schema
        rec = {
            "sample_id": int(sample_id),
            "lat": lat,
            "lon": lon,
            "has_solar": has_solar_bool,
            "confidence": round(confidence, 4),      # your model's confidence, currently 0.0
            "pv_area_sqm_est": round(pv_area_m2, 2), # estimated panel area in m², currently 0.0
            "buffer_radius_sqft": DEFAULT_BUFFER_RADIUS_SQFT,
            "qc_status": qc_status,
            "bbox_or_mask": bbox_or_mask,
            "image_metadata": {
                "source": IMAGE_SOURCE,
                "capture_date": CAPTURE_DATE
            }
        }

        final_records.append(rec)

    # 4) Save to final JSON file
    out_path = Path(OUTPUT_JSON_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(final_records, f, indent=2)

    print(f"✅ Final predictions JSON saved to: {out_path}")
    print(f"   Total sites: {len(final_records)}")

# ════════════════════════════════════════════════════════════════════════════════
# FUNCTIONS FROM export_rooftop_json.py
# ════════════════════════════════════════════════════════════════════════════════

def export_rooftop_json():
    """Export rooftop JSON to final format."""
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    final_recs: List[Dict[str, Any]] = []

    for rec in records:
        # ✅ FIXED: Use correct keys from inference.py output
        out: Dict[str, Any] = {
            "sample_id": int(rec["sample_id"]),
            "latitude": float(rec["latitude"]),      # ✅ Matches inference.py
            "longitude": float(rec["longitude"]),   # ✅ Matches inference.py
            "hassolar": bool(rec["hassolar"]),       # ✅ Matches inference.py
            "confidence": float(rec["confidence"]),
            "pv_area_sqm_est": float(rec["pv_area_sqm_est"]),
            "buffer_radius_sqft": int(rec["buffer_radius_sqft"]),
            "qc_status": str(rec["qc_status"]),
            "bbox_or_mask": rec.get("bbox_or_mask", ""),
            "image_metadata": {
                "source": rec.get("image_metadata", {}).get("source", "GOOGLE_STATIC_MAPS"),
                "zoom": rec.get("image_metadata", {}).get("zoom", 19),
                "capture_date": rec.get("image_metadata", {}).get("capture_date", "2025-12-05"),
            },
        }
        final_recs.append(out)

    out_path = Path(OUTPUT_JSON_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_recs, f, indent=2)

    print(f"🎯 FINAL SUBMISSION: {len(final_recs)} records saved to {OUTPUT_JSON_PATH}")

# ════════════════════════════════════════════════════════════════════════════════
# FUNCTIONS FROM gen_artifacts.py (UPDATED FOR MULTI-PANEL DETECTION)
# ════════════════════════════════════════════════════════════════════════════════

def load_input_csv() -> pd.DataFrame:
    """
    Load EI_train_data.csv and normalize column names.
    
    CSV file has exact columns: sampleid, latitude, longitude, hassolar
    (all lowercase, no underscores except in column names)
    
    Internal names after normalization: sample_id, lat, lon, hassolar
    """
    df = pd.read_csv(CSV_PATH)
    
    # Verify exact column names from CSV file
    expected_columns = {"sampleid", "latitude", "longitude", "hassolar"}
    actual_columns = set(df.columns)
    
    if not expected_columns.issubset(actual_columns):
        missing = expected_columns - actual_columns
        raise ValueError(
            f"CSV must contain columns {expected_columns}. "
            f"Missing: {missing}. Found: {actual_columns}"
        )
    
    # Rename columns to internal names for consistency
    df = df.rename(
        columns={
            "sampleid": "sample_id",      # sampleid -> sample_id
            "latitude": "lat",            # latitude -> lat
            "longitude": "lon",           # longitude -> lon
            # hassolar stays as hassolar (no rename needed)
        }
    )
    
    # Convert sample_id to int (handles string IDs like "0001", "2447", etc.)
    # Strip whitespace and convert to int
    df["sample_id"] = df["sample_id"].astype(str).str.strip().astype(int)
    
    # Ensure numeric columns are float
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    
    # Remove any rows with invalid data
    df = df.dropna(subset=["sample_id", "lat", "lon"])
    
    return df


def create_overlay_image(
    img_bgr: np.ndarray,
    record: Dict[str, Any],
    overlay_path: Path,
) -> None:
    """
    Draw overlay for a single prediction record with multiple panels.
    Highlights the panel with the largest area in lime green, others in cyan.

    record keys (from SolarPanelInference.predict):
      sample_id, latitude, longitude, hassolar, confidence,
      pv_area_sqm_est, buffer_radius_sqft, qc_status, bbox_or_mask, 
      panels_in_buffer, best_panel_id, image_metadata
    """

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    sample_id = record["sample_id"]
    lat = record["latitude"]
    lon = record["longitude"]
    hassolar = record["has_solar"]
    buffer_radius_sqft = record["buffer_radius_sqft"]
    qc_status = record["qc_status"]
    panels_in_buffer = record.get("panels_in_buffer", [])
    best_panel_id = record.get("best_panel_id", -1)

    # Calculate total area and average confidence from all panels
    total_area = sum(panel.get("inside_area_sqm", 0.0) for panel in panels_in_buffer)
    if panels_in_buffer:
        avg_confidence = sum(panel.get("conf", 0.0) for panel in panels_in_buffer) / len(panels_in_buffer)
    else:
        avg_confidence = record.get("confidence", 0.0)
        total_area = record.get("pv_area_sqm_est", 0.0)

    num_panels = len(panels_in_buffer)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(img_rgb)

    # Title: "Solar Panel Detection - PANELS DETECTED" when panels found
    if num_panels > 0:
        title_text = "Solar Panel Detection - PANELS DETECTED"
        title_color = "darkblue"
    else:
        title_text = "Solar Panel Detection - NOT_VERIFIABLE"
        title_color = "orange"
    
    ax.set_title(
        title_text,
        fontsize=14,
        fontweight="bold",
        color=title_color,
        pad=10,
    )

    # Legend (matching reference image format)
    legend_text = (
        "Lime = Best Panel (Highest Overlap)\n"
        "Cyan = Other Panels\n"
        "Area = Intersection with Buffer\n"
        "Confidence = Model Detection Score"
    )
    ax.text(
        10,
        30,
        legend_text,
        fontsize=9,
        color="white",
        fontweight="bold",
        bbox=dict(
            boxstyle="round",
            facecolor="darkblue",
            alpha=0.85,
            edgecolor="cyan",
            linewidth=2,
        ),
    )

    # Draw all panels
    for panel in panels_in_buffer:
        panel_id = panel["panel_id"]
        conf = panel["conf"]
        inside_area = panel["inside_area_sqm"]
        x_c, y_c, w_n, h_n = panel["bbox_center"]

        # Convert normalized center format to pixel coordinates
        x1 = (x_c - w_n / 2) * w
        y1 = (y_c - h_n / 2) * h
        bw_px = w_n * w
        bh_px = h_n * h

        # Highlight best panel in lime green, others in cyan
        if panel_id == best_panel_id:
            edge_color = "lime"
            linewidth = 3
        else:
            edge_color = "cyan"
            linewidth = 2

        rect = Rectangle(
            (x1, y1),
            bw_px,
            bh_px,
            linewidth=linewidth,
            edgecolor=edge_color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Label with Panel ID, Confidence, and Area
        label_text = f"Panel {panel_id} | Conf: {conf:.3f} | Area: {inside_area:.1f} m²"
        ax.text(
            x1 + 4,
            y1 - 8,
            label_text,
            fontsize=9,
            color=edge_color,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    # Information box (matching reference image format)
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    has_solar_text = "YES" if hassolar else "NO"
    
    meta = (
        f"Sample ID: {sample_id}\n"
        f"Lat: {abs(lat):.4f}°{lat_dir} | Lon: {abs(lon):.4f}°{lon_dir}\n"
        f"has_solar: {has_solar_text}\n"
        f"Total Area: {total_area:.1f} m²\n"
        f"Confidence: {avg_confidence:.4f}\n"
        f"Panels Found: {num_panels}\n"
        f"QC Status: {qc_status}\n"
        f"Buffer: {buffer_radius_sqft} sqft"
    )
    
    box_color = "darkgreen" if hassolar else "darkorange"
    edge_color = "lime" if hassolar else "yellow"
    
    ax.text(
        10,
        h - 10,
        meta,
        fontsize=9,
        color="white",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor=box_color,
            alpha=0.85,
            edgecolor=edge_color,
            linewidth=2,
        ),
    )

    ax.axis("off")
    plt.tight_layout()
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(overlay_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_batch() -> None:
    """
    Batch inference with buffer logic rules:
    
    BUFFER LOGIC RULES:
    ===================
    1. Run inference with 1200 sqft buffer first
    2. Check all panels inside the 1200 sqft buffer area
    3. Calculate panel areas (including partial overlaps)
    4. Select panel with maximum area inside buffer
    5. If has_solar=False after 1200 sqft check:
       - Run again with 2400 sqft buffer
       - Apply same rules: check panels, calculate areas, select max
    6. If 2400 sqft finds solar, use that record
    7. Otherwise, keep the 1200 sqft "no solar" record
    
    Processing steps:
    1. Load data/raw/EI_train_data.csv.
    2. For each row, load matching Google tile from
       data/processed/google_images_all/{sample_id}.jpg.
    3. Run SolarPanelInference.predict() to apply YOLO + buffer logic.
    4. Save overlay image and append record.
    5. Write outputs/predictions.json and outputs/predictions.csv.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input_csv()
    logger.info(f"Loaded {len(df)} rows from {CSV_PATH}")

    inf = SolarPanelInference()  # uses models/solar_model_best.pt and GOOGLE_IMG_DIR internally
    predictions: List[Dict[str, Any]] = []
    errors = []
    processed = 0
    failed = 0

    for idx, row in df.iterrows():
        try:
            # Access columns after renaming (sample_id, lat, lon from original sampleid, latitude, longitude)
            sample_id = int(row["sample_id"])  # Originally "sampleid" in CSV
            lat = float(row["lat"])             # Originally "latitude" in CSV
            lon = float(row["lon"])              # Originally "longitude" in CSV

            logger.info(
                f"[{idx+1}/{len(df)}] Processing sample {sample_id} "
                f"({lat:.4f}, {lon:.4f})"
            )

            # Check if image exists before processing
            from pathlib import Path
            img_path = Path("data/processed/google_images_all") / f"{sample_id}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            # Run YOLO inference + buffer logic
            # This follows the buffer rules:
            # 1. Run with 1200 sqft buffer first
            # 2. Check panels inside 1200 sqft buffer area
            # 3. If has_solar=False, run again with 2400 sqft buffer
            # 4. Check panels inside 2400 sqft buffer area
            # 5. If 2400 sqft finds solar, use that record; otherwise keep 1200 sqft "no solar" record
            rec = inf.predict(sample_id=sample_id, lat=lat, lon=lon)

            # Load the same Google image tile your inference uses
            img_bgr = load_image_for_sample(sample_id)
            overlay_path = OVERLAYS_DIR / f"{sample_id}_overlay.jpg"
            create_overlay_image(img_bgr, rec, overlay_path)

            # Store overlay relative path for auditing
            rec["image_metadata"]["overlay_path"] = str(
                overlay_path.relative_to(OUTPUT_DIR)
            )

            predictions.append(rec)
            processed += 1
            
            # Periodic save every 100 samples to prevent data loss
            if processed % 100 == 0:
                json_path_temp = OUTPUT_DIR / "predictions_temp.json"
                with open(json_path_temp, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=2)
                logger.info(f"💾 Periodic save: {processed} samples processed so far")
            
        except FileNotFoundError as e:
            logger.warning(f"⚠️  Image not found for sample {sample_id}: {e}")
            failed += 1
            errors.append({"sample_id": sample_id, "error": str(e), "type": "FileNotFoundError"})
            # Create default record for missing images
            default_rec = {
                "sample_id": sample_id,
                "latitude": lat,
                "longitude": lon,
                "has_solar": False,
                "confidence": 0.0,
                "pv_area_sqm_est": 0.0,
                "buffer_radius_sqft": 1200,
                "panels_in_buffer": [],
                "best_panel_id": -1,
                "qc_status": "NOT_VERIFIABLE",
                "bbox_or_mask": "",
                "image_metadata": {
                    "source": "GOOGLE_STATIC_MAPS",
                    "error": str(e)
                }
            }
            predictions.append(default_rec)
            
        except KeyboardInterrupt:
            logger.error(f"⚠️  Processing interrupted by user at sample {sample_id}")
            break
            
        except Exception as e:
            logger.error(f"❌ Error processing sample {sample_id}: {e}", exc_info=True)
            failed += 1
            errors.append({"sample_id": sample_id, "error": str(e), "type": type(e).__name__})
            # Create default record for errors
            try:
                default_rec = {
                    "sample_id": sample_id,
                    "latitude": lat,
                    "longitude": lon,
                    "has_solar": False,
                    "confidence": 0.0,
                    "pv_area_sqm_est": 0.0,
                    "buffer_radius_sqft": 1200,
                    "panels_in_buffer": [],
                    "best_panel_id": -1,
                    "qc_status": "NOT_VERIFIABLE",
                    "bbox_or_mask": "",
                    "image_metadata": {
                        "source": "GOOGLE_STATIC_MAPS",
                        "error": str(e)
                    }
                }
                predictions.append(default_rec)
            except:
                logger.error(f"❌ Failed to create default record for sample {sample_id}")
                pass

    # Save results
    json_path = OUTPUT_DIR / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    csv_path = OUTPUT_DIR / "predictions.csv"
    pd.DataFrame(predictions).to_csv(csv_path, index=False)
    
    # Save error log if any
    if errors:
        error_path = OUTPUT_DIR / "errors.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        logger.warning(f"⚠️  {len(errors)} errors encountered. See {error_path}")

    logger.info(f"✅ Processed: {processed} samples successfully")
    logger.info(f"⚠️  Failed: {failed} samples")
    logger.info(f"📊 Saved predictions JSON to {json_path}")
    logger.info(f"📊 Saved predictions CSV to {csv_path}")
    logger.info(f"🖼️  Overlays written to {OVERLAYS_DIR}")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_batch()

