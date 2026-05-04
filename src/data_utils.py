from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = ["TV", "Radio", "Social Media"]
CATEGORICAL_FEATURES = ["Influencer"]
TARGET_REGRESSION = "Sales"


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in NUMERIC_FEATURES + [TARGET_REGRESSION]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())

    out["Influencer"] = out["Influencer"].fillna("Unknown")

    out["TotalBudget"] = out[NUMERIC_FEATURES].sum(axis=1)
    out["ROI"] = out[TARGET_REGRESSION] / np.clip(out["TotalBudget"], 1e-9, None)

    # Convert continuous sales into classes for classification tasks.
    out["SalesClass"] = pd.qcut(
        out[TARGET_REGRESSION],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    )

    return out


def split_regression_data(df: pd.DataFrame, test_size: float = 0.2) -> SplitData:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET_REGRESSION]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )
    return SplitData(X_train, X_test, y_train, y_test)


def split_classification_data(df: pd.DataFrame, test_size: float = 0.2) -> SplitData:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df["SalesClass"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )
    return SplitData(X_train, X_test, y_train, y_test)


def default_data_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "raw" / "Dummy Data HSS.csv"

