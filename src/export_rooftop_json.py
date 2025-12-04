from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.inference import SolarPanelInference

CSV_PATH = Path("data/raw/EI_train_data.csv")
OUTPUT_JSON = Path("outputs/solar_rooftops_google.json")


def main() -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Normalize column names
    if "sampleid" in df.columns:
        df = df.rename(columns={"sampleid": "sample_id"})
    if "latitude" in df.columns:
        df = df.rename(columns={"latitude": "lat"})
    if "longitude" in df.columns:
        df = df.rename(columns={"longitude": "lon"})

    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    infer = SolarPanelInference()
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        sid = int(row["sample_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])

        try:
            # Single call: internally does
            # 1) 1200 sqft buffer
            # 2) If no solar, 2400 sqft
            # 3) Panel selection + partial area inside buffer
            rec = infer.predict(
                sample_id=sid,
                lat=lat,
                lon=lon,
            )

            # rec already looks like:
            # {
            #   "sample_id": 1234,
            #   "lat": 12.9716,
            #   "lon": 77.5946,
            #   "has_solar": True/False,
            #   "confidence": 0.xx,
            #   "pv_area_sqm_est": 23.5,
            #   "buffer_radius_sqft": 1200 or 2400,
            #   "qc_status": "VERIFIABLE" / "NOT_VERIFIABLE",
            #   "bbox_or_mask": "x_c,y_c,w,h" or "",
            #   "image_metadata": {"source": "XYZ", "capture_date": "YYYY-MM-DD"},
            # }

            records.append(rec)
            print(
                f"[ok] sample_id={sid} "
                f"buffer={rec['buffer_radius_sqft']} "
                f"has_solar={rec['has_solar']} "
                f"conf={rec['confidence']}"
            )

        except Exception as e:
            print(f"[error] sample_id={sid}: {e}")

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
