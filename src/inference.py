from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, List
import json

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = Path("models/solar_model_best.pt")
GOOGLE_IMG_DIR = Path("data/processed/google_images_all")
BUFFER_1200 = 1200
BUFFER_2400 = 2400
MIN_OVERLAP_THRESHOLD = 0.01  # 1% minimum overlap


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
        bbox_norm: Tuple[float, float, float, float],  # (x1,y1,w,h) normalized
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
        bbox1_norm: Tuple[float, float, float, float],   # panel (x1,y1,w,h)
        bbox2_norm: Tuple[float, float, float, float],  # buffer (x1,y1,w,h)
    ) -> float:
        """Fraction of panel area that lies inside buffer (0–1). ✅ FIXED"""
        # Convert to absolute coordinates [x1,y1,x2,y2]
        x1_1, y1_1, w1, h1 = bbox1_norm
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1  # ✅ FIXED: Proper y2_1 calculation

        x1_2, y1_2, w2, h2 = bbox2_norm
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Intersection rectangle
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0

        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        panel_area = w1 * h1
        return float(inter_area / panel_area) if panel_area > 0 else 0.0


class QCChecker:
    """Quality control based on sharpness, lighting, and confidence."""

    def sharpness_score(self, img_rgb: np.ndarray) -> float:
        """Laplacian variance (higher = sharper). Normalized 0-1."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(var / 1000.0, 1.0))

    def darkness_ratio(self, img_rgb: np.ndarray) -> float:
        """Fraction of dark pixels (<80 gray value)."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return float((gray < 80).sum() / gray.size)

    def qc_status(
        self, img_rgb: np.ndarray, conf: float
    ) -> Tuple[str, List[str]]:
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
    """🚀 Production-ready solar panel detection - EcoInnovators Ideathon 2026."""

    def __init__(self, model_path: Path = MODEL_PATH, zoom: int = 19, conf_threshold: float = 0.25):
        """Initialize with production settings. 🔧 Tunable confidence threshold."""
        self.model = YOLO(str(model_path))
        self.area_calc = AreaCalculator()
        self.qc_checker = QCChecker()
        self.zoom = zoom
        self.conf_threshold = conf_threshold
        print(f"✅ Model loaded: {model_path} | Config: zoom={zoom}, conf={conf_threshold}")

    def _buffer_bbox_normalized(
        self,
        img_w: int,
        img_h: int,
        buffer_sqft: int,
    ) -> Tuple[float, float, float, float]:
        """⚡ Optimized circular buffer bbox (normalized x1,y1,w,h)."""
        buffer_m2 = buffer_sqft * AreaCalculator.SQFT_TO_M2
        buffer_radius_m = np.sqrt(buffer_m2 / np.pi)
        mpp = self.area_calc.meters_per_pixel(self.zoom)

        buffer_radius_px = buffer_radius_m / mpp
        x_center, y_center = img_w / 2, img_h / 2

        x1 = max(0, x_center - buffer_radius_px)
        y1 = max(0, y_center - buffer_radius_px)
        x2 = min(img_w, x_center + buffer_radius_px)
        y2 = min(img_h, y_center + buffer_radius_px)

        return (x1/img_w, y1/img_h, (x2-x1)/img_w, (y2-y1)/img_h)

    def _predict_with_buffer(
        self,
        sample_id: int,
        lat: float,
        lon: float,
        buffer_sqft: int,
    ) -> Dict[str, Any]:
        """🔍 Single buffer prediction - Production optimized."""
        img = load_image_for_sample(sample_id)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        buffer_bbox = self._buffer_bbox_normalized(img_w, img_h, buffer_sqft)
        results = self.model(img, verbose=False, conf=self.conf_threshold)[0]  # 🔧 Tunable conf
        boxes = results.boxes

        panels = []
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                conf = float(box.conf.item())
                xywhn = box.xywhn[0].cpu().numpy()  # YOLO center format (x_c, y_c, w, h)
                x_c, y_c, w_n, h_n = map(float, xywhn)

                # Convert to top-left for buffer intersection check
                x1, y1 = x_c - w_n/2, y_c - h_n/2
                bbox_norm = (x1, y1, w_n, h_n)

                inter_ratio = self.area_calc.bbox_intersection_ratio(bbox_norm, buffer_bbox)
                
                # 📏 Tunable overlap threshold
                if inter_ratio > MIN_OVERLAP_THRESHOLD:
                    full_area = self.area_calc.bbox_area_m2(bbox_norm, img_w, img_h, self.zoom)
                    inside_area = full_area * inter_ratio
                    panels.append({
                        "conf": conf,
                        "inside_area": inside_area,
                        "bbox_center": (x_c, y_c, w_n, h_n),  # ✅ Store YOLO center format directly
                    })

        if panels:
            # Select panel with maximum area inside buffer
            best = max(panels, key=lambda x: x["inside_area"])
            hassolar = True                           # ✅ FIXED: camelCase
            confidence = best["conf"]
            pv_area_sqm_est = round(best["inside_area"], 2)
            
            # ✅ Use stored center format directly (no conversion needed)
            x_c, y_c, w_n, h_n = best["bbox_center"]
            bbox_or_mask = f"{x_c:.4f},{y_c:.4f},{w_n:.4f},{h_n:.4f}"
        else:
            hassolar = False                          # ✅ FIXED: camelCase
            confidence = 1.0
            pv_area_sqm_est = 0.0
            bbox_or_mask = ""

        qc_status, reasons = self.qc_checker.qc_status(img_rgb, confidence)

        return {
            "sample_id": int(sample_id),
            "latitude": float(lat),                   # ✅ FIXED: Matches CSV
            "longitude": float(lon),                 # ✅ FIXED: Matches CSV
            "hassolar": hassolar,                    # ✅ FIXED: Matches CSV exactly
            "confidence": round(confidence, 4),
            "pv_area_sqm_est": pv_area_sqm_est,
            "buffer_radius_sqft": buffer_sqft,
            "qc_status": qc_status,
            "bbox_or_mask": bbox_or_mask,
            "image_metadata": {
                "source": "GOOGLE_STATIC_MAPS",
                "zoom": self.zoom,
                "conf_threshold": self.conf_threshold,
                "img_shape": (img_h, img_w),
            },
        }

    def predict(
        self,
        sample_id: int,
        lat: float,
        lon: float,
    ) -> Dict[str, Any]:
        """🎯 Tiered prediction: 1200sqft → 2400sqft fallback (Production logic)."""
        # Primary: 1200 sqft residential buffer
        rec_1200 = self._predict_with_buffer(sample_id, lat, lon, BUFFER_1200)
        if rec_1200["hassolar"]:                  # ✅ FIXED: Correct key
            return rec_1200

        # Fallback: 2400 sqft (larger roofs/commercial)
        rec_2400 = self._predict_with_buffer(sample_id, lat, lon, BUFFER_2400)
        if rec_2400["hassolar"]:                  # ✅ FIXED: Correct key
            return rec_2400

        return rec_1200  # Return 1200sqft no-solar result


if __name__ == "__main__":
    # 🚀 Initialize production inference engine
    inf = SolarPanelInference(
        model_path=MODEL_PATH,
        zoom=19,
        conf_threshold=0.25  # 🔧 Optimal confidence threshold
    )
    
    # Test prediction
    result = inf.predict(sample_id=1234, lat=12.9716, lon=77.5946)
    
    # ✅ PERFECT CSV-COMPATIBLE OUTPUT
    print(json.dumps(result, indent=2))