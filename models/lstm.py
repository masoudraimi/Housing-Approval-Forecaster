"""PyTorch LSTM for quarterly dwelling approvals forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["HousingLSTM", "train_epoch", "validate_epoch", "fit"]

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 10
_DEFAULT_DROPOUT = 0.2


class HousingLSTM(nn.Module):
    """LSTM-based sequence model for multi-step dwelling approvals forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        output_steps: int = 4,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, seq_len, input_size) -> (batch, output_steps)."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slide a window over X/y to produce (n_samples, seq_len, features) tensors."""
    xs, ys = [], []
    for i in range(len(X) - seq_len - horizon + 1):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + horizon])
    return (
        torch.tensor(np.array(xs), dtype=torch.float32),
        torch.tensor(np.array(ys), dtype=torch.float32),
    )


def train_epoch(
    model: HousingLSTM,
    loader: DataLoader,
    optimiser: AdamW,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def validate_epoch(
    model: HousingLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation; return (mean_loss, MAE)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(X_batch)
            total_mae += torch.abs(preds - y_batch).mean().item() * len(X_batch)
    n = len(loader.dataset)
    return total_loss / n, total_mae / n


def fit(
    model: HousingLSTM,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seq_len: int = 12,
    horizon: int = 4,
    batch_size: int = 32,
    max_epochs: int = 100,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    device: Optional[torch.device] = None,
    mlflow_run: bool = False,
) -> dict:
    """Full training loop with early stopping and optional MLflow metric logging.

    Returns a dict with keys: train_losses, val_losses, val_maes, best_epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    X_tr_seq, y_tr_seq = _make_sequences(X_train, y_train, seq_len, horizon)
    X_vl_seq, y_vl_seq = _make_sequences(X_val, y_val, seq_len, horizon)

    train_loader = DataLoader(TensorDataset(X_tr_seq, y_tr_seq), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_vl_seq, y_vl_seq), batch_size=batch_size, shuffle=False)

    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    train_losses, val_losses, val_maes = [], [], []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        if mlflow_run:
            import mlflow
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae},
                step=epoch,
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val MAE={best_val_mae:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_epoch = len(train_losses) - no_improve
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maes": val_maes,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
    }
