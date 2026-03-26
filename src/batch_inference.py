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

from inference import SolarPanelInference, load_image_for_sample


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
            "estimated_capacity_kw": round(pv_area_m2 * 0.2, 2),
            "estimated_annual_production_kwh": round(pv_area_m2 * 0.2 * 1460.0, 2),
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
            "estimated_capacity_kw": round(float(rec["pv_area_sqm_est"]) * 0.2, 2),
            "estimated_annual_production_kwh": round(float(rec["pv_area_sqm_est"]) * 0.2 * 1460.0, 2),
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

    # Get overall area and source
    total_area = record.get("pv_area_sqm_est", 0.0)
    source = record.get("image_metadata", {}).get("source", "GOOGLE_STATIC_MAPS")
    is_upload = (source == "USER_UPLOAD")
    
    if panels_in_buffer:
        avg_confidence = sum(panel.get("conf", 0.0) for panel in panels_in_buffer) / len(panels_in_buffer)
    else:
        avg_confidence = record.get("confidence", 0.0)
        # total_area already set from record above
    
    # Use standard panel area for labels if it's an upload
    STANDARD_PANEL_AREA = 1.8 # Sync with inference.py

    num_panels = len(panels_in_buffer)

    # Set modern font style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial', 'DejaVu Sans']

    # Create a panoramic 3-panel figure: [Sidebar | Hero Image | Sidebar]
    # Re-balancing to give side panels more horizontal space (3:6.5:3 ratio)
    fig = plt.figure(figsize=(32, 18), dpi=100, facecolor='#000000')
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 6.5, 3], wspace=0.1)
    
    ax_legend = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    ax_info = fig.add_subplot(gs[2])

    for ax_item in [ax_legend, ax_main, ax_info]:
        ax_item.set_facecolor('#000000')
        ax_item.axis("off")

    # --- CENTER PANEL: IMAGE & DETECTION ---
    ax_main.imshow(img_rgb)
    
    if num_panels > 0:
        title_text = "Solar Panel Detection - PANELS DETECTED"
        title_color = "#10b981" # Emerald 500
    else:
        title_text = "Solar Panel Detection - NOT_VERIFIABLE"
        title_color = "#f59e0b" # Amber 500
    
    ax_main.set_title(
        title_text,
        fontsize=24,
        fontweight="black",
        color=title_color,
        pad=40,
    )

    # Draw all panels on main axis
    for panel in panels_in_buffer:
        panel_id = panel["panel_id"]
        conf = panel["conf"]
        inside_area = panel.get("full_area_sqm", panel.get("inside_area_sqm", 0.0))
        x_c, y_c, w_n, h_n = panel["bbox_center"]

        x1 = (x_c - w_n / 2) * w
        y1 = (y_c - h_n / 2) * h
        bw_px = w_n * w
        bh_px = h_n * h

        if panel_id == best_panel_id:
            edge_color = "lime"
            linewidth = 2.5
        else:
            edge_color = "cyan"
            linewidth = 1.5

        rect = Rectangle(
            (x1, y1),
            bw_px,
            bh_px,
            linewidth=linewidth,
            edgecolor=edge_color,
            facecolor="none",
        )
        ax_main.add_patch(rect)

        display_area = STANDARD_PANEL_AREA if is_upload else inside_area
        label_text = f"P{panel_id} | {conf:.2f} | {display_area:.1f}m²"
        ax_main.text(
            x1,
            y1 - 15,
            label_text,
            fontsize=16,
            color="white",
            fontweight="black",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=edge_color, alpha=1.0, edgecolor="none"),
        )

    # --- LEFT PANEL: LEGEND & SYSTEM ---
    legend_text = (
        "VISUAL LEGEND\n"
        "───────────────\n"
        "● Lime: Prime Target\n"
        "● Cyan: Verified Panel\n\n"
        "DETECTION SPECS\n"
        "───────────────\n"
        "Engine: YOLOv8 Solar\n"
        "Type: " + ("PHOTO" if is_upload else "SATELLITE") + "\n"
        "Zoom: " + str(record.get("image_metadata", {}).get("zoom", 19)) + "x\n"
        "Buffer: " + str(buffer_radius_sqft) + " sqft"
    )
    ax_legend.text(
        0.5, 0.5, 
        legend_text,
        fontsize=18,
        color="white",
        fontweight="bold",
        linespacing=2.8,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=3.5", facecolor="#1e293b", alpha=0.6, edgecolor="#334155", linewidth=3.5)
    )

    # --- RIGHT PANEL: PERFORMANCE METRICS ---
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    has_solar_text = "VERIFIED" if hassolar else "NONE"
    
    meta_text = (
        "CORE INTELLIGENCE\n"
        "──────────────────\n"
        f"STATUS: {has_solar_text}\n"
        f"COUNT: {num_panels} Units\n"
        f"CONF: {avg_confidence*100:.1f}%\n"
        f"QC: {qc_status}\n\n"
        "ENERGY ESTIMATES\n"
        "──────────────────\n"
        f"AREA: {total_area:.1f} m²\n"
        f"CAPACITY: {total_area * 0.2:.2f} kW\n"
        f"YIELD: {total_area * 0.2 * 1460:.0f} kWh/y\n\n"
        "LOCATION DATA\n"
        "──────────────────\n"
        f"LAT: {abs(lat):.4f}°{lat_dir}\n"
        f"LON: {abs(lon):.4f}°{lon_dir}"
    )
    
    ax_info.text(
        0.5, 0.5, 
        meta_text,
        fontsize=18,
        color="white",
        fontweight="bold",
        linespacing=2.8,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=3.5", facecolor="#1e293b", alpha=0.6, edgecolor="#334155", linewidth=3.5)
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(overlay_path, dpi=100, bbox_inches="tight", facecolor='#000000')
    plt.close(fig)


def run_batch() -> None:
    """
    Batch inference (no buffer constraints):
    
    Processing steps:
    1. Load data/raw/EI_train_data.csv.
    2. For each row, load matching Google tile from
       data/processed/google_images_all/{sample_id}.jpg.
    3. Run SolarPanelInference.predict() to detect all panels.
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

            # Run YOLO inference (no buffer constraints - all panels counted)
            rec = inf.predict(sample_id=sample_id, lat=lat, lon=lon, image_type="SATELLITE")

            # Add capacity and energy estimates based on area (Assume 20% efficiency, 1460 sun hours)
            area = float(rec.get("pv_area_sqm_est", 0.0))
            rec["estimated_capacity_kw"] = round(area * 0.2, 2)
            rec["estimated_annual_production_kwh"] = round(area * 0.2 * 1460.0, 2)

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
                "estimated_capacity_kw": 0.0,
                "estimated_annual_production_kwh": 0.0,
                "buffer_radius_sqft": 0,
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
                    "estimated_capacity_kw": 0.0,
                    "estimated_annual_production_kwh": 0.0,
                    "buffer_radius_sqft": 0,
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

if __name__ == "__main__":
    run_batch()

