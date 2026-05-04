from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_budget(
    model,
    total_budget: float,
    influencer: str = "Mega",
    lower_bounds: tuple[float, float, float] = (0.1, 0.1, 0.1),
    upper_bounds: tuple[float, float, float] = (0.8, 0.8, 0.8),
    random_trials: int = 300,
    grid_step: float = 0.05,
    seed: int = 42,
) -> dict:
    channels = ["TV", "Radio", "Social Media"]
    rng = np.random.default_rng(seed)

    if not (sum(lower_bounds) <= 1.0 <= sum(upper_bounds)):
        raise ValueError("Infeasible bounds: sum(lower_bounds) must be <= 1 and sum(upper_bounds) must be >= 1.")

    def is_feasible(ratios: np.ndarray) -> bool:
        return (
            np.isclose(np.sum(ratios), 1.0, atol=1e-6)
            and np.all(ratios >= np.array(lower_bounds) - 1e-8)
            and np.all(ratios <= np.array(upper_bounds) + 1e-8)
        )

    def predict_sales_from_ratios(ratios: np.ndarray) -> float:
        budgets = np.array(ratios, dtype=float) * total_budget
        row = pd.DataFrame(
            [{"TV": budgets[0], "Radio": budgets[1], "Social Media": budgets[2], "Influencer": influencer}]
        )
        return float(model.predict(row)[0])

    def objective(ratios):
        predicted_sales = predict_sales_from_ratios(np.array(ratios, dtype=float))
        predicted_roi = predicted_sales / max(total_budget, 1e-9)
        # Maximize ROI by minimizing the negative ROI.
        return -predicted_roi

    def grid_candidates(step: float) -> Iterable[np.ndarray]:
        values = np.arange(0.0, 1.0 + step / 2.0, step)
        for a in values:
            for b in values:
                c = 1.0 - a - b
                if c < 0.0:
                    continue
                ratios = np.array([a, b, c], dtype=float)
                if is_feasible(ratios):
                    yield ratios

    def random_candidates(n_trials: int) -> Iterable[np.ndarray]:
        for _ in range(n_trials):
            sample = rng.dirichlet(np.ones(3))
            if is_feasible(sample):
                yield sample

    def compute_marginal_impact(best_ratios: np.ndarray) -> dict:
        # Finite-difference style impact while preserving the total budget.
        base_sales = predict_sales_from_ratios(best_ratios)
        delta = min(0.05, max(0.01, (1.0 - max(lower_bounds)) / 4.0))
        impact = {}

        for i, channel in enumerate(channels):
            plus = best_ratios.copy()
            minus = best_ratios.copy()

            others = [idx for idx in range(3) if idx != i]
            plus[i] += delta
            plus[others] -= delta / 2.0

            minus[i] -= delta
            minus[others] += delta / 2.0

            if is_feasible(plus):
                plus_sales = predict_sales_from_ratios(plus)
                plus_delta = plus_sales - base_sales
            else:
                plus_delta = None

            if is_feasible(minus):
                minus_sales = predict_sales_from_ratios(minus)
                minus_delta = minus_sales - base_sales
            else:
                minus_delta = None

            impact[channel] = {
                "ratio_shift": float(delta),
                "delta_sales_if_ratio_increases": plus_delta,
                "delta_sales_if_ratio_decreases": minus_delta,
            }

        return impact

    constraints = [{"type": "eq", "fun": lambda r: np.sum(r) - 1.0}]
    bounds = list(zip(lower_bounds, upper_bounds))
    candidates = [np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)]
    candidates.extend(list(grid_candidates(grid_step)))
    candidates.extend(list(random_candidates(random_trials)))

    best_ratios = None
    best_score = float("-inf")
    best_origin = ""
    tried_points = 0

    for base in candidates:
        if not is_feasible(base):
            continue

        tried_points += 1
        base_sales = predict_sales_from_ratios(base)
        if base_sales > best_score:
            best_score = base_sales
            best_ratios = base.copy()
            best_origin = "candidate_search"

        result = minimize(
            objective,
            x0=base,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 100, "ftol": 1e-6},
        )
        if result.success and is_feasible(result.x):
            refined_ratios = np.array(result.x, dtype=float)
            refined_sales = predict_sales_from_ratios(refined_ratios)
            if refined_sales > best_score:
                best_score = refined_sales
                best_ratios = refined_ratios
                best_origin = "local_refinement"

    if best_ratios is None:
        raise RuntimeError("Optimization failed: no feasible solution found.")

    best_budgets = best_ratios * total_budget
    predicted_sales = predict_sales_from_ratios(best_ratios)
    marginal_impact = compute_marginal_impact(best_ratios)

    return {
        "total_budget": total_budget,
        "influencer": influencer,
        "recommended_budget": {channels[i]: float(best_budgets[i]) for i in range(3)},
        "recommended_ratio": {channels[i]: float(best_ratios[i]) for i in range(3)},
        "predicted_sales": predicted_sales,
        "predicted_roi": predicted_sales / max(total_budget, 1e-9),
        "search_diagnostics": {
            "candidate_points_tested": tried_points,
            "method": "grid + random + SLSQP multi-start",
            "best_solution_source": best_origin,
            "bounds_ratio": {channels[i]: [float(lower_bounds[i]), float(upper_bounds[i])] for i in range(3)},
        },
        "marginal_impact": marginal_impact,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "best_regression_model.joblib"
    output_path = root / "reports" / "budget_optimization.json"

    if not model_path.exists():
        raise FileNotFoundError(
            "Missing trained model. Run: python src/train.py"
        )

    model = joblib.load(model_path)

    recommendation = optimize_budget(
        model=model,
        total_budget=120.0,
        influencer="Mega",
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)

    print(f"Saved optimization result to: {output_path}")


if __name__ == "__main__":
    main()

