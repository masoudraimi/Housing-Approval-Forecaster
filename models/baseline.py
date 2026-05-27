"""Baseline forecasting models: seasonal mean and SARIMA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["BaseForecaster", "SeasonalMeanBaseline", "SARIMABaseline"]


class BaseForecaster(ABC):
    """Abstract base class defining the forecaster interface."""

    @abstractmethod
    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None) -> "BaseForecaster":
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Return an array of length `horizon` with point forecasts."""
        ...


class SeasonalMeanBaseline(BaseForecaster):
    """Predict as the rolling seasonal mean of the same quarter over the last N years."""

    def __init__(self, n_years: int = 3) -> None:
        self.n_years = n_years
        self._seasonal_means: dict[int, float] = {}

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None) -> "SeasonalMeanBaseline":
        """Compute per-quarter mean over the last n_years of training data."""
        df = pd.DataFrame({"value": y_train})
        if isinstance(y_train.index, pd.PeriodIndex):
            df["quarter_num"] = y_train.index.quarter
        else:
            df["quarter_num"] = pd.to_datetime(y_train.index).quarter

        cutoff = len(df) - self.n_years * 4
        recent = df.iloc[max(0, cutoff):]
        self._seasonal_means = recent.groupby("quarter_num")["value"].mean().to_dict()
        self._last_quarter_num = df["quarter_num"].iloc[-1]
        return self

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Repeat seasonal mean for each forecast quarter."""
        preds = []
        q = self._last_quarter_num
        for _ in range(horizon):
            q = q % 4 + 1
            preds.append(self._seasonal_means.get(q, np.mean(list(self._seasonal_means.values()))))
        return np.array(preds)


class SARIMABaseline(BaseForecaster):
    """Per-series SARIMA(1,1,1)(0,1,1,4) baseline using statsmodels."""

    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = (0, 1, 1, 4)) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None) -> "SARIMABaseline":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        return self

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        forecast = self._result.forecast(steps=horizon)
        return np.array(forecast)
