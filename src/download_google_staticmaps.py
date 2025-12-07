# src/download_google_staticmaps.py
from __future__ import annotations

import os
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv
import requests
from PIL import Image

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

CSV_PATH = Path("data/raw/EI_train_data.csv")
OUT_DIR = Path("data/processed/google_images_all")

GOOGLE_STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"

load_dotenv()


# ----------------------------------------------------------------------
# STATIC MAP CLIENT
# ----------------------------------------------------------------------
@dataclass
class StaticMapConfig:
    zoom: int = 19                  # ~0.3 m/pixel
    size: Tuple[int, int] = (400, 400)  # width x height
    maptype: str = "satellite"
    scale: int = 1                  # 1 or 2 (2 = retina, 2x pixels)
    api_key: str = os.getenv("GOOGLE_STATIC_MAPS_API_KEY", "")


class GoogleStaticMapsClient:
    def __init__(self, config: StaticMapConfig | None = None) -> None:
        self.config = config or StaticMapConfig()
        if not self.config.api_key:
            raise RuntimeError(
                "GOOGLE_STATIC_MAPS_API_KEY not set in .env or environment."
            )

    def fetch_image(self, lat: float, lon: float) -> Image.Image:
        """Fetch a rooftop satellite tile centered at (lat, lon)."""
        width, height = self.config.size
        params = {
            "center": f"{lat},{lon}",
            "zoom": self.config.zoom,
            "size": f"{width}x{height}",
            "maptype": self.config.maptype,
            "scale": self.config.scale,
            "key": self.config.api_key,
        }

        resp = requests.get(GOOGLE_STATIC_MAPS_URL, params=params, timeout=20)
        resp.raise_for_status()
        img_bytes = io.BytesIO(resp.content)
        return Image.open(img_bytes).convert("RGB")


# ----------------------------------------------------------------------
# MAIN DOWNLOADER
# ----------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Normalize column names
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
