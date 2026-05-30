"""Drift detection: feature drift (z-score on construction costs) and residual drift (rolling MAE)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

__all__ = ["DriftReport", "compute_rolling_mae", "detect_feature_drift", "detect_residual_drift"]

PREDICTION_LOG_PATH = Path("data/predictions.db")
FEATURES_PATH = Path("data/processed/features.parquet")

_RESIDUAL_DRIFT_MULTIPLIER = 1.5
_FEATURE_DRIFT_Z_THRESHOLD = 2.5


class DriftReport(BaseModel):
    """Summary of drift detection results."""

    baseline_mae: float
    current_mae: float
    mae_ratio: float
    residual_drift_detected: bool
    feature_drift_detected: bool
    cost_pressure_z_score: Optional[float]
    window_days: int
    n_predictions: int


def _load_prediction_log(db_path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    import sqlite_utils
    db = sqlite_utils.Database(str(db_path))
    if "predictions" not in db.table_names():
        return pd.DataFrame()
    rows = list(db["predictions"].rows)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["predicted_values"] = df["predicted_values"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    )
    df["actual_values"] = df["actual_values"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else None
    )
    return df


def compute_rolling_mae(
    window_days: int = 90,
    db_path: Path = PREDICTION_LOG_PATH,
) -> tuple[float, int]:
    """Compute mean absolute error over predictions with actuals in the last window_days.

    Returns (rolling_mae, n_predictions_with_actuals).
    """
    df = _load_prediction_log(db_path)
    if df.empty or "actual_values" not in df.columns:
        return 0.0, 0

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
    recent = df[df["timestamp"] >= cutoff].dropna(subset=["actual_values"])
    if recent.empty:
        return 0.0, 0

    errors = []
    for _, row in recent.iterrows():
        preds = row["predicted_values"]
        actuals = row["actual_values"]
        if preds and actuals:
            n = min(len(preds), len(actuals))
            errors.extend(np.abs(np.array(preds[:n]) - np.array(actuals[:n])))

    if not errors:
        return 0.0, 0
    return float(np.mean(errors)), len(recent)


def detect_feature_drift(
    features_path: Path = FEATURES_PATH,
    current_construction_cost_yoy: Optional[float] = None,
) -> tuple[bool, Optional[float]]:
    """Check if the current construction cost growth is outside the training distribution.

    Uses a z-score against the pre-Accord (pre-2022Q3) training distribution.
    Returns (drift_detected, z_score).
    """
    if current_construction_cost_yoy is None:
        return False, None

    df = pd.read_parquet(features_path)
    if "construction_cost_yoy" not in df.columns or "post_accord_2022" not in df.columns:
        return False, None

    baseline = df[df["post_accord_2022"] == 0]["construction_cost_yoy"].dropna()
    if baseline.empty:
        return False, None

    mean = baseline.mean()
    std = baseline.std()
    if std == 0:
        return False, 0.0

    z = (current_construction_cost_yoy - mean) / std
    return abs(z) > _FEATURE_DRIFT_Z_THRESHOLD, round(float(z), 3)


def detect_residual_drift(
    baseline_mae: float,
    window_days: int = 90,
    db_path: Path = PREDICTION_LOG_PATH,
) -> tuple[bool, float, int]:
    """Check if rolling MAE exceeds 1.5x the training-period baseline MAE.

    Returns (drift_detected, current_mae, n_predictions).
    """
    current_mae, n = compute_rolling_mae(window_days, db_path)
    if n == 0:
        return False, 0.0, 0
    drift = current_mae > baseline_mae * _RESIDUAL_DRIFT_MULTIPLIER
    return drift, current_mae, n


def generate_drift_report(
    baseline_mae: float,
    window_days: int = 90,
    current_construction_cost_yoy: Optional[float] = None,
    db_path: Path = PREDICTION_LOG_PATH,
    features_path: Path = FEATURES_PATH,
) -> DriftReport:
    """Produce a full drift report combining feature and residual drift signals."""
    feature_drift, z_score = detect_feature_drift(features_path, current_construction_cost_yoy)
    residual_drift, current_mae, n = detect_residual_drift(baseline_mae, window_days, db_path)
    return DriftReport(
        baseline_mae=baseline_mae,
        current_mae=current_mae,
        mae_ratio=round(current_mae / baseline_mae, 3) if baseline_mae > 0 else 0.0,
        residual_drift_detected=residual_drift,
        feature_drift_detected=feature_drift,
        cost_pressure_z_score=z_score,
        window_days=window_days,
        n_predictions=n,
    )
