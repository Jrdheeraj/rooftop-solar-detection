from __future__ import annotations

import hashlib


MAX_SAFE_JS_INTEGER = 9_007_199_254_740_991


def validate_coordinates(lat: float, lon: float) -> tuple[float, float]:
    """Validate and normalize latitude/longitude inputs."""
    lat = float(lat)
    lon = float(lon)

    if not -90.0 <= lat <= 90.0:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("Longitude must be between -180 and 180.")

    return lat, lon


def build_request_sample_id(*parts: object) -> int:
    """Create a stable, collision-resistant sample id for request-scoped images."""
    payload = "|".join(_normalize_part(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
    sample_id = int(digest[:13], 16)
    return min(sample_id, MAX_SAFE_JS_INTEGER)


def _normalize_part(part: object) -> str:
    if isinstance(part, float):
        return f"{part:.8f}"
    return str(part).strip()
