# src/batch_processor.py

import csv
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


class BatchProcessor:
    def __init__(self, model_path="models/solar_model_best.pt"):
        self.model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")

    def process_folder(self, image_folder, output_csv="outputs/batch_results.csv", confidence=0.5):
        """
        Run detection on all .jpg images in image_folder and save results to CSV.
        """
        image_folder = Path(image_folder)
        image_paths = sorted(image_folder.glob("*.jpg"))
        print(f"📂 Found {len(image_paths)} images in {image_folder}")

        if not image_paths:
            print("⚠️ No .jpg files found. Check your folder path.")
            return []

        results = []

        for img_path in tqdm(image_paths, desc="Processing"):
            preds = self.model.predict(
                source=str(img_path),
                conf=confidence,
                verbose=False
            )
            pred = preds[0]
            num_det = len(pred.boxes)
            confs = pred.boxes.conf.tolist() if num_det > 0 else []
            max_conf = max(confs) if confs else 0.0
            avg_conf = sum(confs) / len(confs) if confs else 0.0

            results.append({
                "image_id": img_path.stem,
                "image_path": str(img_path),
                "detections": num_det,
                "max_confidence": round(max_conf, 4),
                "avg_confidence": round(avg_conf, 4)
            })

        # Make sure outputs dir exists
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        output_csv = out_dir / Path(output_csv).name

        # Write CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"✅ Results saved to: {output_csv}")

        total_det = sum(r["detections"] for r in results)
        imgs_with = sum(1 for r in results if r["detections"] > 0)

        print("\n📊 Summary:")
        print(f"   Total images: {len(results)}")
        print(f"   Images with detections: {imgs_with}")
        print(f"   Total detections: {total_det}")
        print(f"   Avg detections per image: {total_det / len(results):.2f}")

        return results


if __name__ == "__main__":
    processor = BatchProcessor(model_path="models/solar_model_best.pt")
    processor.process_folder(
        image_folder="data/processed/images_all",
        output_csv="outputs/batch_results.csv",
        confidence=0.5
    )
