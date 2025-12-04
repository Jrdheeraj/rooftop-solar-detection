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
        out: Dict[str, Any] = {
            "sample_id": int(rec["sample_id"]),
            "lat": float(rec["lat"]),
            "lon": float(rec["lon"]),
            "has_solar": bool(rec["has_solar"]),
            "confidence": float(rec["confidence"]),
            "pv_area_sqm_est": float(rec["pv_area_sqm_est"]),
            "buffer_radius_sqft": int(rec["buffer_radius_sqft"]),
            "qc_status": str(rec["qc_status"]),
            "bbox_or_mask": rec.get("bbox_or_mask", ""),
            "image_metadata": {
                "source": rec.get("image_metadata", {}).get("source", "XYZ"),
                "capture_date": rec.get("image_metadata", {}).get("capture_date", "YYYY-MM-DD"),
            },
        }
        final_recs.append(out)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(final_recs, f, indent=2)

    print(f"Saved {len(final_recs)} final records to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
