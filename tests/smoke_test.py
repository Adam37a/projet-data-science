from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import clean_data, default_data_path, load_data


def main() -> None:
    df = clean_data(load_data(default_data_path()))

    assert not df.empty, "Dataset should not be empty"
    assert "ROI" in df.columns, "ROI feature missing"
    assert "SalesClass" in df.columns, "SalesClass target missing"

    reg_model = ROOT / "models" / "best_regression_model.joblib"
    if reg_model.exists():
        _ = joblib.load(reg_model)

    print("Smoke test passed")


if __name__ == "__main__":
    main()


