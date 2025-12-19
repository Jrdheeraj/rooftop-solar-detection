from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List
import json
import logging
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO



MODEL_PATH = Path("models/solar_model_best.pt")
GOOGLE_IMG_DIR = Path("data/processed/google_images_all")

# 🎯 CRITICAL THRESHOLDS FOR PANEL DETECTION
# Lower confidence threshold = higher recall (catch more panels)
CONFIDENCE_THRESHOLD = 0.25   # 25%   (typical YOLO default)
MIN_OVERLAP_THRESHOLD = 0.10  # 10%   (panel must overlap buffer by ≥10%)

# Buffer sizes (square feet)
BUFFER_1200 = 1200  # Typical residential
BUFFER_2400 = 2400  # Larger residential / commercial fallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_image_for_sample(sample_id: int) -> np.ndarray:
    """Load Google Static Maps tile for a given sample_id."""
    img_path = GOOGLE_IMG_DIR / f"{sample_id}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found for sample_id={sample_id}: {img_path}")
    return img


class AreaCalculator:
    """BBox → area in m² at given zoom (zoom=19 default for Google Static Maps)."""

    SQFT_TO_M2 = 0.092903

    @staticmethod
    def meters_per_pixel(zoom: int = 19) -> float:
        """Meters per pixel at given zoom level."""
        return 156543.03392 / (2 ** zoom)

    def bbox_area_m2(
        self,
        bbox_norm: Tuple[float, float, float, float],  # (x1, y1, w, h) normalized
        img_w: int,
        img_h: int,
        zoom: int = 19,
    ) -> float:
        """Area of full panel bbox in m²."""
        _, _, w_n, h_n = bbox_norm
        w_px = w_n * img_w
        h_px = h_n * img_h
        mpp = self.meters_per_pixel(zoom)
        return float((w_px * mpp) * (h_px * mpp))

    def bbox_intersection_ratio(
        self,
        bbox1_norm: Tuple[float, float, float, float],  # panel (x1, y1, w, h)
        bbox2_norm: Tuple[float, float, float, float],  # buffer (x1, y1, w, h)
    ) -> float:
        """Fraction of panel area that lies inside buffer (0–1)."""
        x1_1, y1_1, w1, h1 = bbox1_norm
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = bbox2_norm
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Intersection bounds
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # No intersection
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0

        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        panel_area = w1 * h1
        return float(inter_area / panel_area) if panel_area > 0 else 0.0


class QCChecker:
    """Quality control based on sharpness, lighting, and confidence."""

    def sharpness_score(self, img_rgb: np.ndarray) -> float:
        """Laplacian variance (higher = sharper). Normalized 0–1."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(var / 1000.0, 1.0))

    def darkness_ratio(self, img_rgb: np.ndarray) -> float:
        """Fraction of dark pixels (<80 gray value)."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return float((gray < 80).sum() / gray.size)

    def qc_status(self, img_rgb: np.ndarray, conf: float) -> Tuple[str, List[str]]:
        """Returns QC status and reasons."""
        reasons: List[str] = []
        sharp = self.sharpness_score(img_rgb)
        dark = self.darkness_ratio(img_rgb)

        if sharp < 0.25:
            reasons.append("LOW_RESOLUTION_OR_BLUR")
        if dark > 0.4:
            reasons.append("HEAVY_SHADOW_OR_CLOUD")
        if conf < 0.5:
            reasons.append("LOW_CONFIDENCE")

        return ("NOT_VERIFIABLE", reasons) if reasons else ("VERIFIABLE", ["Clear evidence"])


class SolarPanelInference:
    """Production-ready solar panel detection for EcoInnovators Ideathon.
    
    Implements tiered buffer strategy:
    1. Run with 1200 sqft buffer (residential)
    2. If has_solar=False, run with 2400 sqft buffer (larger roofs)
    3. Return best result
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        zoom: int = 19,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.model = YOLO(str(model_path))
        self.area_calc = AreaCalculator()
        self.qc_checker = QCChecker()
        self.zoom = zoom
        self.conf_threshold = conf_threshold
        logger.info(
            f" Model loaded: {model_path} | Config: zoom={zoom}, conf_threshold={conf_threshold}"
        )

    def _buffer_bbox_normalized(
        self, img_w: int, img_h: int, buffer_sqft: int
    ) -> Tuple[float, float, float, float]:
        """Circular buffer bbox (normalized x1, y1, w, h)."""
        buffer_m2 = buffer_sqft * AreaCalculator.SQFT_TO_M2
        buffer_radius_m = np.sqrt(buffer_m2 / np.pi)
        mpp = self.area_calc.meters_per_pixel(self.zoom)
        buffer_radius_px = buffer_radius_m / mpp

        x_center, y_center = img_w / 2, img_h / 2
        x1 = max(0, x_center - buffer_radius_px)
        y1 = max(0, y_center - buffer_radius_px)
        x2 = min(img_w, x_center + buffer_radius_px)
        y2 = min(img_h, y_center + buffer_radius_px)

        return (x1 / img_w, y1 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h)

    def _predict_with_buffer(
        self, sample_id: int, lat: float, lon: float, buffer_sqft: int
    ) -> Dict[str, Any]:
        """Single-buffer prediction:
        - keeps ALL detected panels in panels_in_buffer
        - best_panel_id = panel with max area inside buffer
        - pv_area_sqm_est = that best panel's inside-buffer area
        """
        img = load_image_for_sample(sample_id)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        buffer_bbox = self._buffer_bbox_normalized(img_w, img_h, buffer_sqft)

        # Run YOLO inference
        results = self.model(img, verbose=False, conf=self.conf_threshold)[0]
        boxes = results.boxes

        panels: List[Dict[str, Any]] = []

        # Process all detected boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf.item())
                xywhn = box.xywhn[0].cpu().numpy()
                x_c, y_c, w_n, h_n = map(float, xywhn)

                # Convert to top-left format for intersection calculation
                x1, y1 = x_c - w_n / 2, y_c - h_n / 2
                bbox_norm = (x1, y1, w_n, h_n)

                # Calculate intersection ratio with buffer
                inter_ratio = self.area_calc.bbox_intersection_ratio(bbox_norm, buffer_bbox)
                full_area = self.area_calc.bbox_area_m2(bbox_norm, img_w, img_h, self.zoom)
                inside_area = full_area * inter_ratio

                panels.append(
                    {
                        "panel_id": len(panels),
                        "conf": conf,
                        "full_area_sqm": round(full_area, 2),
                        "inside_area_sqm": round(inside_area, 2),
                        "overlap_ratio": round(inter_ratio, 4),
                        "bbox_center": (x_c, y_c, w_n, h_n),
                    }
                )

        # Determine best panel and solar status
        if panels:
            best_panel = max(panels, key=lambda x: x["inside_area_sqm"])
            best_panel_id = best_panel["panel_id"]
            has_solar = best_panel["inside_area_sqm"] > 0.0
            confidence = best_panel["conf"]
            pv_area_sqm_est = best_panel["inside_area_sqm"]
            x_c, y_c, w_n, h_n = best_panel["bbox_center"]
            bbox_or_mask = f"{x_c:.4f},{y_c:.4f},{w_n:.4f},{h_n:.4f}"
        else:
            has_solar = False
            confidence = 0.0
            pv_area_sqm_est = 0.0
            bbox_or_mask = ""
            best_panel_id = -1
            panels = []

        qc_status, qc_reasons = self.qc_checker.qc_status(img_rgb, confidence)

        return {
            "sample_id": int(sample_id),
            "latitude": float(lat),
            "longitude": float(lon),
            "has_solar": has_solar,
            "confidence": round(confidence, 4),
            "pv_area_sqm_est": round(pv_area_sqm_est, 2),
            "buffer_radius_sqft": buffer_sqft,
            "panels_in_buffer": panels,
            "best_panel_id": best_panel_id,
            "qc_status": qc_status,
            "bbox_or_mask": bbox_or_mask,
            "image_metadata": {
                "source": "GOOGLE_STATIC_MAPS",
                "capture_date": datetime.now().strftime("%Y-%m-%d"),
                "zoom": self.zoom,
                "conf_threshold": self.conf_threshold,
                "overlap_threshold": MIN_OVERLAP_THRESHOLD,
                "img_shape": (img_h, img_w),
                "qc_reasons": qc_reasons,
            },
        }

    def predict(self, sample_id: int, lat: float, lon: float) -> Dict[str, Any]:
        """Tiered prediction: 1200 sqft → 2400 sqft fallback."""
        # Primary: 1200 sqft residential buffer
        rec_1200 = self._predict_with_buffer(sample_id, lat, lon, BUFFER_1200)
        if rec_1200["has_solar"]:
            return rec_1200

        # Fallback: 2400 sqft (larger roofs/commercial)
        rec_2400 = self._predict_with_buffer(sample_id, lat, lon, BUFFER_2400)
        if rec_2400["has_solar"]:
            return rec_2400

        # No solar found in either buffer → return 1200 by default
        return rec_1200


if __name__ == "__main__":
    try:
        inf = SolarPanelInference()
        result = inf.predict(sample_id=1, lat=17.3850, lon=78.4867)
        print(json.dumps(result, indent=2))
    except FileNotFoundError as e:
        logger.error(f"Test failed: {e}")
        print("Make sure data/processed/google_images_all/1.jpg exists")
