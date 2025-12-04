# src/staticmaps_fetcher.py
import os
import io
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv
import requests
from PIL import Image

load_dotenv()

GOOGLE_STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"


@dataclass
class StaticMapConfig:
    zoom: int = 19                         # ~0.3 m/pixel
    size: Tuple[int, int] = (400, 400)     # width x height
    maptype: str = "satellite"
    scale: int = 1                         # 1 or 2 (2 = retina, 2x pixels)
    api_key: str = os.getenv("GOOGLE_STATIC_MAPS_API_KEY", "")


class GoogleStaticMapsClient:
    def __init__(self, config: StaticMapConfig | None = None):
        self.config = config or StaticMapConfig()
        if not self.config.api_key:
            raise RuntimeError("GOOGLE_STATIC_MAPS_API_KEY not set in .env or environment.")

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
