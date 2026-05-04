from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(title: str, cmd: list[str]) -> None:
    print(f"\n--- {title} ---")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    run_step("EDA - Exploratory Data Analysis", [sys.executable, "src/eda.py"])

    run_step("Train all models", [sys.executable, "src/train.py"])
    run_step("Run budget optimization", [sys.executable, "src/optimize_roi.py"])

    print("\nProject artifacts generated:")
    print("- reports/regression_metrics.csv")
    print("- reports/classification_metrics.csv")
    print("- reports/budget_optimization.json")
    print("- models/best_regression_model.joblib")
    print("- models/best_classification_model.joblib")


if __name__ == "__main__":
    main()

