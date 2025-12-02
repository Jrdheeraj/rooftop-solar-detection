# src/build_final_predictions_json.py

import json
from pathlib import Path
import pandas as pd

SOLAR_JSON_PATH = "outputs/solar_rooftops.json"
EI_CSV_PATH = "data/raw/EI_train_data.csv"
OUTPUT_JSON_PATH = "outputs/predictions_final.json"

DEFAULT_BUFFER_RADIUS_SQFT = 1200
QC_STATUS = "NOT_VERIFIABLE"
IMAGE_SOURCE = "synthetic_static"
CAPTURE_DATE = "N/A"


def load_solar_rooftops():
    p = Path(SOLAR_JSON_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run export_rooftop_json.py first.")
    with open(p, "r") as f:
        data = json.load(f)
    return {str(rec["image_id"]): rec for rec in data}


def load_ei_csv():
    p = Path(EI_CSV_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    df = pd.read_csv(p)
    if "sampleid" not in df.columns:
        raise ValueError("Column 'sampleid' missing in EI_train_data.csv")
    return df.set_index("sampleid")


def main():
    solar_idx = load_solar_rooftops()
    df = load_ei_csv()

    final_records = []

    for sample_id, row in df.iterrows():
        image_id_str = f"{float(sample_id)}"
        solar_rec = solar_idx.get(image_id_str)

        if solar_rec is None:
            pv_area_m2 = 0.0
            confidence = 0.0
        else:
            pv_area_m2 = float(solar_rec.get("panel_area_m2", 0.0))
            confidence = float(solar_rec.get("max_confidence", 0.0))

        has_solar = bool(int(row.get("hassolar", 0)))
        lat = float(row.get("latitude"))
        lon = float(row.get("longitude"))

        rec_out = {
            "sample_id": int(sample_id),
            "lat": lat,
            "lon": lon,
            "has_solar": has_solar,
            "confidence": round(confidence, 4),
            "pv_area_sqm_est": round(pv_area_m2, 2),
            "buffer_radius_sqft": DEFAULT_BUFFER_RADIUS_SQFT,
            "qc_status": QC_STATUS,
            "bbox_or_mask": "[]",
            "image_metadata": {
                "source": IMAGE_SOURCE,
                "capture_date": CAPTURE_DATE,
            },
        }

        final_records.append(rec_out)

    out_path = Path(OUTPUT_JSON_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_records, f, indent=2)

    print(f"✅ Final predictions JSON saved to: {out_path}")
    print(f"   Total sites: {len(final_records)}")


if __name__ == "__main__":
    main()
