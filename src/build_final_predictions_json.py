from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.inference import SolarPanelInference  # ✅ Uses your fixed inference.py

CSV_PATH = Path("data/raw/EI_train_data.csv")
OUTPUT_JSON = Path("outputs/solar_rooftops_google.json")

def main() -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # ✅ NO RENAMING NEEDED - CSV already has correct columns
    # CSV format: sampleid,latitude,longitude,hassolar
    infer = SolarPanelInference()
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        sid = int(row["sampleid"])  # ✅ CSV has "sampleid"
        lat = float(row["latitude"])  # ✅ CSV has "latitude"
        lon = float(row["longitude"])  # ✅ CSV has "longitude"

        try:
            rec = infer.predict(
                sample_id=sid,
                lat=lat,
                lon=lon,
            )
            
            records.append(rec)
            print(
                f"[✅] sample_id={sid} "
                f"buffer={rec['buffer_radius_sqft']}sqft "
                f"hassolar={rec['hassolar']} "
                f"area={rec['pv_area_sqm_est']}m² "
                f"conf={rec['confidence']:.3f} "
                f"qc={rec['qc_status']}"
            )
        except Exception as e:
            print(f"[❌] sample_id={sid}: {e}")
            # Add error record with defaults
            records.append({
                "sample_id": sid,
                "latitude": lat,
                "longitude": lon,
                "hassolar": False,
                "confidence": 0.0,
                "pv_area_sqm_est": 0.0,
                "buffer_radius_sqft": 1200,
                "qc_status": "ERROR",
                "bbox_or_mask": "",
            })

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"🎉 Saved {len(records)} predictions to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
