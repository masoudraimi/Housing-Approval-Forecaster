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


def _make_population(n_lgas: int = 3, n_quarters: int = 20) -> pd.DataFrame:
    """Synthetic population data aligned with approvals quarters."""
    quarters = pd.period_range("2018Q1", periods=n_quarters, freq="Q")
    rows = []
    for i in range(n_lgas):
        for q in quarters:
            rows.append({
                "lga_code": f"LGA{i:02d}",
                "quarter": q,
                "population": float(np.random.randint(10_000, 200_000)),
            })
    return pd.DataFrame(rows)


def test_build_features_output_columns():
    approvals = _make_approvals()
    population = _make_population()
    features = build_features(approvals, population)
    expected = [
        "lga_code", "lga_name", "quarter", "dwellings_approved",
        "approvals_lag1",
        "population_growth_yoy", "construction_cost_yoy",
        "season_q1", "season_q4",
        "post_accord_2022",
    ]
    for col in expected:
        assert col in features.columns, f"Missing column: {col}"


def test_build_features_no_cash_rate():
    approvals = _make_approvals()
    population = _make_population()
    features = build_features(approvals, population)
    for col in ("cash_rate", "cash_rate_lag1", "cash_rate_lag2", "post_rate_hike"):
        assert col not in features.columns, f"Unexpected column: {col}"


def test_build_features_no_nan_in_lags():
    approvals = _make_approvals()
    population = _make_population()
    features = build_features(approvals, population)
    lag_cols = [c for c in features.columns if "lag" in c]
    assert features[lag_cols].isnull().sum().sum() == 0


def test_build_features_post_accord_flag():
    approvals = _make_approvals(n_lgas=1, n_quarters=24)
    population = _make_population(n_lgas=1, n_quarters=24)
    features = build_features(approvals, population)
    post = features[features["post_accord_2022"] == 1]
    pre = features[features["post_accord_2022"] == 0]
    if len(post) > 0 and len(pre) > 0:
        assert pre["quarter"].max() < post["quarter"].min()


def test_build_features_row_count():
    approvals = _make_approvals(n_lgas=2, n_quarters=16)
    population = _make_population(n_lgas=2, n_quarters=16)
    features = build_features(approvals, population)
    # Fewer rows than raw due to lag NaN removal (n_lags=1 drops first observation per LGA)
    assert len(features) < len(approvals)
    assert len(features) > 0


def test_build_features_ppi_none_gives_zero_cost():
    approvals = _make_approvals()
    population = _make_population()
    features = build_features(approvals, population, ppi_df=None)
    assert "construction_cost_yoy" in features.columns
    assert (features["construction_cost_yoy"] == 0.0).all()
