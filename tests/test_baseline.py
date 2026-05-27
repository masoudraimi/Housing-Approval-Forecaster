"""Tests for models/baseline.py: fit without error, finite predictions."""

import numpy as np
import pandas as pd
import pytest

from models.baseline import SARIMABaseline, SeasonalMeanBaseline


def _quarterly_series(n: int = 24) -> pd.Series:
    idx = pd.period_range("2018Q1", periods=n, freq="Q")
    values = 100 + np.sin(np.arange(n) * np.pi / 2) * 20 + np.random.rand(n) * 10
    return pd.Series(values, index=idx)


def test_seasonal_mean_fit_predict():
    y = _quarterly_series(24)
    model = SeasonalMeanBaseline(n_years=3)
    model.fit(y)
    preds = model.predict(horizon=4)
    assert len(preds) == 4
    assert np.all(np.isfinite(preds))


def test_seasonal_mean_predict_positive():
    y = _quarterly_series(20)
    y = y.abs()
    model = SeasonalMeanBaseline().fit(y)
    preds = model.predict(4)
    assert np.all(preds > 0)


def test_sarima_fit_predict():
    y = _quarterly_series(24)
    model = SARIMABaseline()
    model.fit(y)
    preds = model.predict(horizon=4)
    assert len(preds) == 4
    assert np.all(np.isfinite(preds))


def test_sarima_raises_before_fit():
    model = SARIMABaseline()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(4)
