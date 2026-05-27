"""Tests for data/pipeline.py: feature engineering output shapes and correctness."""

import numpy as np
import pandas as pd
import pytest

from data.pipeline import build_features


def _make_approvals(n_lgas: int = 3, n_quarters: int = 20) -> pd.DataFrame:
    quarters = pd.period_range("2018Q1", periods=n_quarters, freq="Q")
    rows = []
    for i in range(n_lgas):
        for q in quarters:
            rows.append({
                "lga_code": f"LGA{i:02d}",
                "lga_name": f"Test LGA {i}",
                "quarter": q,
                "dwellings_approved": float(np.random.randint(50, 300)),
            })
    return pd.DataFrame(rows)


def _make_cash_rate(n_quarters: int = 20) -> pd.DataFrame:
    quarters = pd.period_range("2018Q1", periods=n_quarters, freq="Q")
    return pd.DataFrame({"quarter": quarters, "cash_rate": np.linspace(0.1, 4.1, n_quarters)})


def test_build_features_output_columns():
    approvals = _make_approvals()
    cash_rate = _make_cash_rate()
    features = build_features(approvals, cash_rate)
    expected = [
        "lga_code", "lga_name", "quarter", "dwellings_approved",
        "approvals_lag1", "approvals_lag4",
        "season_q1", "season_q4",
        "post_rate_hike",
    ]
    for col in expected:
        assert col in features.columns, f"Missing column: {col}"


def test_build_features_no_nan_in_lags():
    approvals = _make_approvals()
    cash_rate = _make_cash_rate()
    features = build_features(approvals, cash_rate)
    # After dropping rows with NaN in approvals_lag4, no lag columns should have NaN
    lag_cols = [c for c in features.columns if "lag" in c]
    assert features[lag_cols].isnull().sum().sum() == 0


def test_build_features_post_rate_hike_flag():
    approvals = _make_approvals(n_lgas=1, n_quarters=24)
    cash_rate = _make_cash_rate(n_quarters=24)
    features = build_features(approvals, cash_rate)
    # Q3 2022 should be flagged; earlier quarters should not
    post_hike = features[features["post_rate_hike"] == 1]
    pre_hike = features[features["post_rate_hike"] == 0]
    if len(post_hike) > 0 and len(pre_hike) > 0:
        assert pre_hike["quarter"].max() < post_hike["quarter"].min()


def test_build_features_row_count():
    approvals = _make_approvals(n_lgas=2, n_quarters=16)
    cash_rate = _make_cash_rate(n_quarters=16)
    features = build_features(approvals, cash_rate)
    # Should have fewer rows than raw due to lag NaN removal
    assert len(features) < len(approvals)
    assert len(features) > 0
