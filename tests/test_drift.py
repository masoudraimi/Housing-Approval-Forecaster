"""Tests for monitoring/drift.py: threshold flags and no false positives."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    _RESIDUAL_DRIFT_MULTIPLIER,
    compute_rolling_mae,
    detect_feature_drift,
    detect_residual_drift,
)
from serving.logger import PredictionLogger


def _populate_log_with_actuals(db_path: Path, n: int, mae_target: float) -> None:
    logger = PredictionLogger(db_path=db_path)
    for _ in range(n):
        predicted = [100.0, 110.0]
        actual = [p + mae_target for p in predicted]
        logger.log("LGA001", 2, predicted, "v1", actual_values=actual)


def _make_features_parquet(tmp_path: Path) -> Path:
    """Create a synthetic features.parquet with construction cost and post_accord_2022 columns."""
    data = {
        "quarter": pd.period_range("2018Q1", periods=20, freq="Q"),
        "lga_code": ["LGA001"] * 20,
        "lga_name": ["Test LGA"] * 20,
        "dwellings_approved": [100.0] * 20,
        # Pre-Accord: low varying cost growth; post-Accord: elevated (needs variance for z-score)
        "construction_cost_yoy": [0.01 + i * 0.001 for i in range(16)] + [0.15, 0.18, 0.19, 0.20],
        "post_accord_2022": [0] * 16 + [1, 1, 1, 1],
    }
    df = pd.DataFrame(data)
    out = tmp_path / "features.parquet"
    df.to_parquet(out, index=False)
    return out


def test_residual_drift_triggered(tmp_path):
    db_path = tmp_path / "test.db"
    baseline_mae = 10.0
    high_mae = baseline_mae * (_RESIDUAL_DRIFT_MULTIPLIER + 0.5)
    _populate_log_with_actuals(db_path, n=20, mae_target=high_mae)
    drift, current_mae, n = detect_residual_drift(baseline_mae=baseline_mae, window_days=365, db_path=db_path)
    assert drift is True
    assert n > 0


def test_residual_drift_not_triggered_below_threshold(tmp_path):
    db_path = tmp_path / "test.db"
    baseline_mae = 10.0
    low_mae = baseline_mae * 0.9
    _populate_log_with_actuals(db_path, n=20, mae_target=low_mae)
    drift, _, _ = detect_residual_drift(baseline_mae=baseline_mae, window_days=365, db_path=db_path)
    assert drift is False


def test_feature_drift_triggered(tmp_path):
    features_path = _make_features_parquet(tmp_path)
    # 1.0 (100% YoY) is far outside training distribution (mean ~0.01)
    drift, z = detect_feature_drift(
        features_path=features_path, current_construction_cost_yoy=1.0
    )
    assert drift
    assert z is not None and z > 2.5


def test_feature_drift_not_triggered_within_training_range(tmp_path):
    features_path = _make_features_parquet(tmp_path)
    # 0.01 is exactly within training range
    drift, z = detect_feature_drift(
        features_path=features_path, current_construction_cost_yoy=0.01
    )
    assert not drift


def test_feature_drift_none_returns_false(tmp_path):
    features_path = _make_features_parquet(tmp_path)
    drift, z = detect_feature_drift(
        features_path=features_path, current_construction_cost_yoy=None
    )
    assert drift is False
    assert z is None
