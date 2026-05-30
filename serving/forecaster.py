"""Forecaster: loads the champion model from MLflow and runs inference."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import mlflow.pytorch
import numpy as np
import torch
from pydantic import BaseModel

__all__ = ["ForecastFeatures", "ForecastResult", "Forecaster"]

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "housing-forecast-lstm")
SEQ_LEN = 12
HORIZON = 4

FEATURE_COLS = [
    "approvals_lag1",
    "population_growth_yoy",
    "construction_cost_yoy",
    "season_q1",
    "season_q2",
    "season_q3",
    "season_q4",
    "post_accord_2022",
]


class ForecastFeatures(BaseModel):
    """Input features for a single LGA forecast request."""

    approvals_lag1: float
    population_growth_yoy: float = 0.0
    construction_cost_yoy: float = 0.0
    season_q1: int = 0
    season_q2: int = 0
    season_q3: int = 0
    season_q4: int = 0
    post_accord_2022: int = 0


class ForecastResult(BaseModel):
    """Response schema for a dwelling approvals forecast."""

    lga_code: str
    horizon_quarters: int
    predicted_approvals: list[float]
    model_version: str
    confidence_note: str


class Forecaster:
    """Lazy-loaded singleton that wraps the MLflow champion model for inference."""

    _instance: Optional["Forecaster"] = None

    def __init__(self) -> None:
        self._model = None
        self._x_scaler = None
        self._y_scaler = None
        self._model_version = "unknown"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get(cls) -> "Forecaster":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        uri = f"models:/{MODEL_NAME}@champion"
        self._model = mlflow.pytorch.load_model(uri, map_location=self._device)
        self._model.eval()

        # Resolve version from registry
        client = mlflow.tracking.MlflowClient()
        try:
            alias_info = client.get_model_version_by_alias(MODEL_NAME, "champion")
            self._model_version = alias_info.version
        except Exception:
            self._model_version = "champion"

        # Load scalers from the same run's artefacts
        try:
            scaler_local = Path("data/processed/scalers.pkl")
            if scaler_local.exists():
                with open(scaler_local, "rb") as f:
                    scalers = pickle.load(f)
                    self._x_scaler = scalers["x_scaler"]
                    self._y_scaler = scalers["y_scaler"]
        except Exception as e:
            print(f"Warning: could not load scalers ({e}). Predictions will be unscaled.")

    @property
    def model_version(self) -> str:
        if self._model is None:
            self._load()
        return self._model_version

    def predict(self, lga_code: str, features: ForecastFeatures) -> ForecastResult:
        """Run inference for a single LGA. Loads the model lazily on first call."""
        if self._model is None:
            self._load()

        feature_vec = np.array([getattr(features, col, 0.0) for col in FEATURE_COLS], dtype=np.float32)

        if self._x_scaler is not None:
            feature_vec = self._x_scaler.transform(feature_vec.reshape(1, -1)).ravel()

        # Repeat the feature vector to fill the sequence window (no history available from request)
        x_seq = torch.tensor(
            np.tile(feature_vec, (SEQ_LEN, 1))[np.newaxis, :, :],
            dtype=torch.float32,
        ).to(self._device)

        with torch.no_grad():
            preds_scaled = self._model(x_seq).cpu().numpy().ravel()

        if self._y_scaler is not None:
            preds = self._y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        else:
            preds = preds_scaled

        return ForecastResult(
            lga_code=lga_code,
            horizon_quarters=HORIZON,
            predicted_approvals=[round(float(p), 1) for p in preds],
            model_version=str(self._model_version),
            confidence_note=(
                "Prediction based on supplied feature values replicated across the sequence window. "
                "For highest accuracy, provide a full 12-quarter history."
            ),
        )
