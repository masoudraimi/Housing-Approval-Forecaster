"""Compare registered baseline and LSTM models on the test set; promote the winner."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from models.baseline import SARIMABaseline, SeasonalMeanBaseline
from models.registry import promote_to_champion

load_dotenv()

FEATURES_PATH = Path("data/raw/approvals_clean.parquet")
TRAIN_END = "2022Q2"
TEST_START = "2022Q3"
HORIZON = 1
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "housing-forecast-lstm")


def _hit_rate(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.15) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8) <= tolerance))


def compare_models(features_path: Path = FEATURES_PATH) -> None:
    df = pd.read_parquet(features_path)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")

    train_mask = df["quarter"] <= TRAIN_END
    test_mask = df["quarter"] >= TEST_START

    print("\n--- Model Comparison on Test Set (post-2022Q3 rate-hike period) ---\n")
    results = {}

    # Seasonal mean
    sm_true, sm_pred_all = [], []
    for lga in df["lga_code"].unique():
        lga_df = df[df["lga_code"] == lga].sort_values("quarter")
        y_tr = lga_df[lga_df["quarter"] <= TRAIN_END]["dwellings_approved"]
        y_te = lga_df[lga_df["quarter"] >= TEST_START]["dwellings_approved"]
        if len(y_tr) < 1 or len(y_te) == 0:
            continue
        sm = SeasonalMeanBaseline().fit(y_tr)
        n = min(len(y_te), HORIZON)
        sm_pred_all.extend(sm.predict(n))
        sm_true.extend(y_te.values[:n])

    if not sm_true:
        print("No LGAs had sufficient data for comparison — check TRAIN_END/TEST_START constants.")
        return

    sm_mae = mean_absolute_error(sm_true, sm_pred_all)
    sm_mape = mean_absolute_percentage_error(sm_true, sm_pred_all)
    sm_hit = _hit_rate(np.array(sm_true), np.array(sm_pred_all))
    results["Seasonal Mean"] = {"MAE": sm_mae, "MAPE": sm_mape, "Hit Rate": sm_hit}

    # Print table
    header = f"{'Model':<20} {'MAE':>10} {'MAPE':>10} {'Hit Rate':>10}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['MAE']:>10.1f} {metrics['MAPE']:>10.3f} {metrics['Hit Rate']:>10.3f}")

    mlflow.set_experiment("housing-approvals-comparison")
    with mlflow.start_run(run_name="model_comparison"):
        for name, metrics in results.items():
            safe_name = name.lower().replace(" ", "_")
            mlflow.log_metrics({
                f"{safe_name}_mae": metrics["MAE"],
                f"{safe_name}_mape": metrics["MAPE"],
                f"{safe_name}_hit_rate": metrics["Hit Rate"],
            })

    # Promote LSTM to champion if it was registered
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            promote_to_champion(MODEL_NAME, latest.version)
            print(f"\nPromoted {MODEL_NAME} v{latest.version} to @champion")
    except Exception as e:
        print(f"Could not promote champion: {e}")


if __name__ == "__main__":
    compare_models()
