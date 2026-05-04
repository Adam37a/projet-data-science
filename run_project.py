from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent


def run_step(title: str, cmd: list[str]) -> None:
    print(f"\n--- {title} ---")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def generate_optimal_allocations() -> None:
    """Generate optimal budget allocations for all influencer tiers."""
    from src.optimize_roi import compute_optimal_allocations
    
    model_path = ROOT / "models" / "best_regression_model.joblib"
    output_path = ROOT / "reports" / "optimal_budget_allocation.json"
    
    if not model_path.exists():
        print("❌ Model not found. Cannot generate optimal allocations.")
        return
    
    print("\n--- Computing Optimal Budget Allocations ---")
    model = joblib.load(model_path)
    allocations = compute_optimal_allocations(model)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(allocations, f, indent=2)
    
    print(f"✅ Saved optimal allocations to: {output_path}")


def main() -> None:
    run_step("EDA - Exploratory Data Analysis", [sys.executable, "src/eda.py"])

    run_step("Train all models", [sys.executable, "src/train.py"])
    run_step("Run budget optimization", [sys.executable, "src/optimize_roi.py"])
    
    # Generate optimal allocations
    generate_optimal_allocations()

    print("\n✅ Project artifacts generated:")
    print("- reports/regression_metrics.csv")
    print("- reports/classification_metrics.csv")
    print("- reports/budget_optimization.json")
    print("- reports/optimal_budget_allocation.json (NEW)")
    print("- models/best_regression_model.joblib")
    print("- models/best_classification_model.joblib")


if __name__ == "__main__":
    main()

