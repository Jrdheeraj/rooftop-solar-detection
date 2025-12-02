# src/export_rooftop_json.py
# Run from project root:
#   .\venv\Scripts\Activate
#   python src/export_rooftop_json.py

import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ---------- CONFIG ----------
MODEL_PATH = "models/solar_model_best.pt"
IMAGE_FOLDER = "data/processed/images_all"
OUTPUT_JSON = "outputs/solar_rooftops.json"

IMG_W = 400
IMG_H = 400

M_PER_PX = 0.1
AREA_PER_PX = M_PER_PX * M_PER_PX
# ---------------------------


def yolo_box_area_px(box, img_w, img_h):
    w_norm = box.xywhn[0][2].item()
    h_norm = box.xywhn[0][3].item()
    w_px = w_norm * img_w
    h_px = h_norm * img_h
    return w_px * h_px


def main():
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")

    image_dir = Path(IMAGE_FOLDER)
    image_paths = sorted(image_dir.glob("*.jpg"))
    print(f"📂 Found {len(image_paths)} images in {image_dir}")

    if not image_paths:
        print("⚠️ No .jpg files found. Check IMAGE_FOLDER.")
        return

    roof_area_px = IMG_W * IMG_H
    roof_area_m2 = roof_area_px * AREA_PER_PX

    all_records = []

    for img_path in tqdm(image_paths, desc="Processing"):
        results = model.predict(source=str(img_path), conf=0.5, verbose=False)
        r = results[0]

        num_boxes = len(r.boxes)
        panel_area_px = 0.0
        confidences = []

        if num_boxes > 0:
            for box in r.boxes:
                a_px = yolo_box_area_px(box, IMG_W, IMG_H)
                panel_area_px += a_px
                confidences.append(box.conf[0].item())

        max_conf = max(confidences) if confidences else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        panel_area_m2 = panel_area_px * AREA_PER_PX
        panel_fraction = panel_area_px / roof_area_px if roof_area_px > 0 else 0.0

        rec = {
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "num_panels": int(num_boxes),
            "panel_area_px": round(panel_area_px, 2),
            "roof_area_px": int(roof_area_px),
            "panel_area_m2": round(panel_area_m2, 2),
            "roof_area_m2": round(roof_area_m2, 2),
            "panel_fraction": round(panel_fraction, 4),
            "max_confidence": round(max_conf, 4),
            "avg_confidence": round(avg_conf, 4),
        }

        all_records.append(rec)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / Path(OUTPUT_JSON).name

    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\n✅ JSON saved to: {out_path}")
    print(f"   Total rooftops: {len(all_records)}")


if __name__ == "__main__":
    main()