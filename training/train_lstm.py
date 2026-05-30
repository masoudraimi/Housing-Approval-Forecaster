"""Train the PyTorch LSTM and register it in the MLflow model registry."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from models.lstm import HousingLSTM, fit as lstm_fit
from models.registry import register_model

load_dotenv()

FEATURES_PATH = Path("data/processed/features.parquet")
TRAIN_FRAC = 0.8   # LGA-based split (no temporal split: 1 obs per LGA)
HORIZON = 1
SEQ_LEN = 1        # cross-sectional: each LGA is one sample, no temporal window
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "housing-forecast-lstm")

FEATURE_COLS = [
    "approvals_lag1",        # local planning momentum
    "population_growth_yoy", # demand pressure from migration (ABS ERP)
    "construction_cost_yoy", # supply constraint (ABS PPI house construction)
    "season_q1",
    "season_q2",
    "season_q3",
    "season_q4",
    "post_accord_2022",      # planning policy break: National Housing Accord Aug 2022
]


def _hit_rate(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.15) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8) <= tolerance))


def train_lstm(features_path: Path = FEATURES_PATH) -> None:
    df = pd.read_parquet(features_path)

    lgas = df["lga_code"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(lgas)
    n_train = int(len(lgas) * TRAIN_FRAC)
    train_lgas = set(lgas[:n_train])
    train_mask = df["lga_code"].isin(train_lgas)
    val_mask = ~train_mask

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X_train_raw = df[train_mask][available_features].values.astype(np.float32)
    y_train_raw = df[train_mask]["dwellings_approved"].values.astype(np.float32)
    X_val_raw = df[val_mask][available_features].values.astype(np.float32)
    y_val_raw = df[val_mask]["dwellings_approved"].values.astype(np.float32)

    # Scale features and target
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    X_val = x_scaler.transform(X_val_raw)
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()

    input_size = len(available_features)
    model = HousingLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_steps=HORIZON,
    )

    mlflow.set_experiment("housing-approvals-lstm")

    with mlflow.start_run(run_name="lstm_v1") as run:
        mlflow.log_params({
            "model_type": "lstm",
            "input_size": input_size,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "features": available_features,
            "train_frac": TRAIN_FRAC,
        })

        history = lstm_fit(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            seq_len=SEQ_LEN,
            horizon=HORIZON,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            lr=LR,
            patience=PATIENCE,
            mlflow_run=True,
        )

        # Evaluate on val set
        model.eval()
        device = next(model.parameters()).device
        from models.lstm import _make_sequences
        X_vl_seq, y_vl_seq = _make_sequences(X_val, y_val, SEQ_LEN, HORIZON)
        with torch.no_grad():
            preds_scaled = model(X_vl_seq.to(device)).cpu().numpy()

        preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
        true_vals = y_scaler.inverse_transform(y_vl_seq.numpy().reshape(-1, 1)).reshape(y_vl_seq.shape)

        mae = mean_absolute_error(true_vals.ravel(), preds.ravel())
        mape = mean_absolute_percentage_error(true_vals.ravel(), preds.ravel())
        hit = _hit_rate(true_vals.ravel(), preds.ravel())

        mlflow.log_metrics({
            "final_val_mae": mae,
            "final_val_mape": mape,
            "hit_rate_15pct": hit,
            "best_epoch": history["best_epoch"],
        })
        print(f"LSTM  MAE={mae:.1f}  MAPE={mape:.3f}  Hit={hit:.3f}")

        # Log model + scalers
        mlflow.pytorch.log_model(model, artifact_path="lstm", registered_model_name=MODEL_NAME)
        mlflow.log_dict({"feature_cols": available_features}, "feature_cols.json")

        scaler_path = Path("data/processed/scalers.pkl")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)
        mlflow.log_artifact(str(scaler_path), "scalers")

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")


if __name__ == "__main__":
    train_lstm()
