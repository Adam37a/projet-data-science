from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, clean_data, default_data_path, load_data, split_classification_data, split_regression_data


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def regression_models(training_mode: str) -> Dict[str, object]:
    is_nonlinear = training_mode == "nonlinear_friendly"
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=500 if is_nonlinear else 300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1 if is_nonlinear else 3,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=42,
            n_estimators=250 if is_nonlinear else 150,
            learning_rate=0.05 if is_nonlinear else 0.1,
        ),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=(128, 64) if is_nonlinear else (64, 32),
            max_iter=1200,
            random_state=42,
            early_stopping=True,
        ),
    }


def classification_models(training_mode: str) -> Dict[str, object]:
    is_nonlinear = training_mode == "nonlinear_friendly"
    return {
        "LogisticRegression": LogisticRegression(max_iter=300),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500 if is_nonlinear else 300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1 if is_nonlinear else 3,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=42,
            n_estimators=250 if is_nonlinear else 150,
            learning_rate=0.05 if is_nonlinear else 0.1,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(128, 64) if is_nonlinear else (64, 32),
            max_iter=1200,
            random_state=42,
            early_stopping=False,
        ),
    }


def correlation_analysis(df: pd.DataFrame, output_path: Path) -> Tuple[pd.DataFrame, float]:
    numeric_corr = df[NUMERIC_FEATURES + ["Sales"]].corr(numeric_only=True)["Sales"].drop("Sales")
    influencer_dummies = pd.get_dummies(df["Influencer"], prefix="Influencer")
    influencer_corr = influencer_dummies.apply(lambda col: col.corr(df["Sales"]))

    rows = []
    for feature, corr_value in numeric_corr.items():
        rows.append(
            {
                "Feature": feature,
                "CorrelationWithSales": float(corr_value),
                "AbsCorrelation": float(abs(corr_value)),
                "FeatureType": "numeric",
            }
        )

    for feature, corr_value in influencer_corr.items():
        safe_corr = 0.0 if pd.isna(corr_value) else float(corr_value)
        rows.append(
            {
                "Feature": feature,
                "CorrelationWithSales": safe_corr,
                "AbsCorrelation": float(abs(safe_corr)),
                "FeatureType": "categorical_dummy",
            }
        )

    corr_df = pd.DataFrame(rows).sort_values(by="AbsCorrelation", ascending=False)
    corr_df.to_csv(output_path, index=False)

    max_numeric_abs_corr = float(numeric_corr.abs().max()) if not numeric_corr.empty else 0.0
    return corr_df, max_numeric_abs_corr


def evaluate_regression(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_test, pred)),
    }


def evaluate_classification(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    pred = model.predict(X_test)
    return {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "F1_macro": float(f1_score(y_test, pred, average="macro")),
    }


def train_all() -> Tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_data(default_data_path()))
    corr_path = reports_dir / "correlation_with_sales.csv"
    corr_df, max_abs_corr = correlation_analysis(df, corr_path)

    training_mode = "linear_friendly" if max_abs_corr >= 0.7 else "nonlinear_friendly"

    reg_split = split_regression_data(df)
    clf_split = split_classification_data(df)
    preprocessor = make_preprocessor()

    reg_results = []
    best_reg_name = None
    best_reg_cv = float("-inf")
    best_reg_r2 = float("-inf")

    for name, estimator in regression_models(training_mode).items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(reg_split.X_train, reg_split.y_train)

        scores = evaluate_regression(pipe, reg_split.X_test, reg_split.y_test)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(pipe, reg_split.X_train, reg_split.y_train, cv=cv, scoring="r2").mean()

        row = {"Model": name, **scores, "CV_R2": float(cv_r2)}
        reg_results.append(row)

        if (cv_r2 > best_reg_cv) or (np.isclose(cv_r2, best_reg_cv) and scores["R2"] > best_reg_r2):
            best_reg_cv = float(cv_r2)
            best_reg_r2 = scores["R2"]
            best_reg_name = name
            joblib.dump(pipe, models_dir / "best_regression_model.joblib")

    clf_results = []
    best_clf_name = None
    best_clf_cv = float("-inf")
    best_clf_f1 = float("-inf")
    best_clf_report = ""

    for name, estimator in classification_models(training_mode).items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(clf_split.X_train, clf_split.y_train)

        scores = evaluate_classification(pipe, clf_split.X_test, clf_split.y_test)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(pipe, clf_split.X_train, clf_split.y_train, cv=cv, scoring="f1_macro").mean()

        pred = pipe.predict(clf_split.X_test)
        clf_report = classification_report(clf_split.y_test, pred)

        row = {"Model": name, **scores, "CV_F1_macro": float(cv_f1)}
        clf_results.append(row)

        if (cv_f1 > best_clf_cv) or (np.isclose(cv_f1, best_clf_cv) and scores["F1_macro"] > best_clf_f1):
            best_clf_cv = float(cv_f1)
            best_clf_f1 = scores["F1_macro"]
            best_clf_name = name
            best_clf_report = clf_report
            joblib.dump(pipe, models_dir / "best_classification_model.joblib")

    reg_df = pd.DataFrame(reg_results).sort_values(by="R2", ascending=False)
    clf_df = pd.DataFrame(clf_results).sort_values(by="F1_macro", ascending=False)

    reg_path = reports_dir / "regression_metrics.csv"
    clf_path = reports_dir / "classification_metrics.csv"
    cv_path = reports_dir / "cv_summary.csv"

    reg_df.to_csv(reg_path, index=False)
    clf_df.to_csv(clf_path, index=False)

    cv_df = pd.concat(
        [
            reg_df[["Model", "R2", "CV_R2"]].rename(columns={"R2": "HoldoutScore", "CV_R2": "CVScore"}).assign(Task="regression"),
            clf_df[["Model", "F1_macro", "CV_F1_macro"]]
            .rename(columns={"F1_macro": "HoldoutScore", "CV_F1_macro": "CVScore"})
            .assign(Task="classification"),
        ],
        ignore_index=True,
    )
    cv_df.to_csv(cv_path, index=False)

    top_corr = corr_df.head(3)[["Feature", "CorrelationWithSales"]].to_dict(orient="records")
    with (reports_dir / "model_selection_policy.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rule": "If max abs numeric correlation with Sales >= 0.7 then linear_friendly else nonlinear_friendly",
                "max_abs_numeric_correlation": max_abs_corr,
                "training_mode": training_mode,
                "top_correlated_features": top_corr,
                "selection_strategy": "Select best model by CV score, tie-break with holdout score",
            },
            f,
            indent=2,
        )

    with (reports_dir / "best_models.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_regression_model": best_reg_name,
                "best_regression_r2": best_reg_r2,
                "best_regression_cv_r2": best_reg_cv,
                "best_classification_model": best_clf_name,
                "best_classification_f1_macro": best_clf_f1,
                "best_classification_cv_f1_macro": best_clf_cv,
                "training_mode": training_mode,
            },
            f,
            indent=2,
        )

    with (reports_dir / "best_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(best_clf_report)

    return reg_path, clf_path


if __name__ == "__main__":
    reg_out, clf_out = train_all()
    print(f"Saved regression metrics to: {reg_out}")
    print(f"Saved classification metrics to: {clf_out}")


