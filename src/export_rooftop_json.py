from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

INPUT_JSON = Path("outputs/solar_rooftops_google.json")
OUTPUT_JSON = Path("outputs/predictions_final.json")

def main() -> None:
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

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(final_recs, f, indent=2)

    print(f"🎯 FINAL SUBMISSION: {len(final_recs)} records saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
