from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import batch_inference


def test_load_input_csv_returns_normalized_dataframe(monkeypatch):
    source_df = pd.DataFrame(
        {
            "sampleid": ["0008", "0009"],
            "latitude": ["22.7337", "22.7359"],
            "longitude": ["72.4351", "72.4350"],
            "hassolar": [1, 0],
        }
    )

    monkeypatch.setattr(batch_inference.pd, "read_csv", lambda *_args, **_kwargs: source_df.copy())

    result = batch_inference.load_input_csv()

    assert list(result.columns) == ["sample_id", "lat", "lon", "hassolar"]
    assert result["sample_id"].tolist() == [8, 9]
    assert result["lat"].tolist() == [22.7337, 22.7359]
    assert result["lon"].tolist() == [72.4351, 72.435]
