from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from request_utils import MAX_SAFE_JS_INTEGER, build_request_sample_id, validate_coordinates


def test_build_request_sample_id_is_stable_and_safe_for_frontend():
    sample_id_a = build_request_sample_id("coords", 22.7337, 72.4351)
    sample_id_b = build_request_sample_id("coords", 22.7337, 72.4351)

    assert sample_id_a == sample_id_b
    assert 0 < sample_id_a <= MAX_SAFE_JS_INTEGER


def test_build_request_sample_id_changes_with_coordinates():
    sample_id_a = build_request_sample_id("coords", 22.7337, 72.4351)
    sample_id_b = build_request_sample_id("coords", 22.7338, 72.4351)

    assert sample_id_a != sample_id_b


def test_validate_coordinates_rejects_invalid_values():
    with pytest.raises(ValueError, match="Latitude"):
        validate_coordinates(91, 72.4351)

    with pytest.raises(ValueError, match="Longitude"):
        validate_coordinates(22.7337, 181)
