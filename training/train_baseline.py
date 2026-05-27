"""Fit baseline models and log results to MLflow."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from models.baseline import SARIMABaseline, SeasonalMeanBaseline
from models.registry import register_model

load_dotenv()

FEATURES_PATH = Path("data/raw/approvals_clean.parquet")
TRAIN_END = "2022Q2"
VAL_END = "2023Q2"
HORIZON = 1
SEASONAL_MODEL_NAME = os.getenv("MLFLOW_BASELINE_NAME", "housing-forecast-baseline")


def _hit_rate(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.15) -> float:
    """Fraction of predictions within ±tolerance of actuals."""
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8) <= tolerance))


def train_baselines(features_path: Path = FEATURES_PATH) -> None:
    df = pd.read_parquet(features_path)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

    train_mask = df["quarter"] <= TRAIN_END
    val_mask = (df["quarter"] > TRAIN_END) & (df["quarter"] <= VAL_END)

    lgas = df["lga_code"].unique()

    all_true, sm_preds, sarima_preds = [], [], []

    for lga in lgas:
        lga_df = df[df["lga_code"] == lga].sort_values("quarter")
        y_train = lga_df.loc[lga_df["quarter"] <= TRAIN_END, "dwellings_approved"]
        y_val = lga_df.loc[(lga_df["quarter"] > TRAIN_END) & (lga_df["quarter"] <= VAL_END), "dwellings_approved"]

        if len(y_train) < 1 or len(y_val) == 0:
            continue

        sm = SeasonalMeanBaseline(n_years=3).fit(y_train)
        try:
            sarima = SARIMABaseline().fit(y_train)
        except Exception:
            continue

        n = min(len(y_val), HORIZON)
        sm_pred = sm.predict(n)[:n]
        sarima_pred = sarima.predict(n)[:n]
        true_vals = y_val.values[:n]

        all_true.extend(true_vals)
        sm_preds.extend(sm_pred)
        sarima_preds.extend(sarima_pred)

    all_true = np.array(all_true)
    sm_preds = np.array(sm_preds)
    sarima_preds = np.array(sarima_preds)

    if len(all_true) == 0:
        print("No LGAs had sufficient data for evaluation — check TRAIN_END/VAL_END constants.")
        return

    mlflow.set_experiment("housing-approvals-baselines")

    with mlflow.start_run(run_name="seasonal_mean") as run:
        mae = mean_absolute_error(all_true, sm_preds)
        mape = mean_absolute_percentage_error(all_true, sm_preds)
        hit = _hit_rate(all_true, sm_preds)
        mlflow.log_params({"model_type": "seasonal_mean", "n_years": 3, "horizon": HORIZON})
        mlflow.log_metrics({"val_mae": mae, "val_mape": mape, "hit_rate_15pct": hit})
        print(f"Seasonal Mean  MAE={mae:.1f}  MAPE={mape:.3f}  Hit={hit:.3f}")

    with mlflow.start_run(run_name="sarima") as run:
        mae = mean_absolute_error(all_true, sarima_preds)
        mape = mean_absolute_percentage_error(all_true, sarima_preds)
        hit = _hit_rate(all_true, sarima_preds)
        mlflow.log_params({"model_type": "sarima", "order": "(1,1,1)", "seasonal_order": "(0,1,1,4)", "horizon": HORIZON})
        mlflow.log_metrics({"val_mae": mae, "val_mape": mape, "hit_rate_15pct": hit})
        print(f"SARIMA         MAE={mae:.1f}  MAPE={mape:.3f}  Hit={hit:.3f}")


if __name__ == "__main__":
    train_baselines()
