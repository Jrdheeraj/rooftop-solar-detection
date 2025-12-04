from __future__ import annotations
from pathlib import Path

import pandas as pd

from src.staticmaps_fetcher import GoogleStaticMapsClient


CSV_PATH = Path("data/raw/EI_train_data.csv")
OUT_DIR = Path("data/processed/google_images_all")



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    if "sampleid" in df.columns:
        df = df.rename(columns={"sampleid": "sample_id"})
    if "latitude" in df.columns:
        df = df.rename(columns={"latitude": "lat"})
    if "longitude" in df.columns:
        df = df.rename(columns={"longitude": "lon"})

    client = GoogleStaticMapsClient()

    print(f"Downloading {len(df)} tiles to {OUT_DIR} ...")

    for _, row in df.iterrows():
        sid = int(row["sample_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])

        out_path = OUT_DIR / f"{sid}.jpg"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        try:
            img = client.fetch_image(lat, lon)
            img.save(out_path, format="JPEG")
            print(f"[ok] sample_id={sid} -> {out_path}")
        except Exception as e:
            print(f"[error] sample_id={sid}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
