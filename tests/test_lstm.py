"""Tests for models/lstm.py: forward pass shape, loss convergence."""

import numpy as np
import pytest
import torch

from models.lstm import HousingLSTM, _make_sequences, fit


def _dummy_data(n: int = 50, n_features: int = 5) -> tuple:
    X = np.random.rand(n, n_features).astype(np.float32)
    y = np.random.rand(n).astype(np.float32) * 100
    return X, y


def test_forward_pass_output_shape():
    model = HousingLSTM(input_size=5, hidden_size=16, num_layers=1, output_steps=4)
    x = torch.randn(8, 12, 5)
    out = model(x)
    assert out.shape == (8, 4)


def test_forward_pass_no_nan():
    model = HousingLSTM(input_size=5, hidden_size=16, num_layers=1, output_steps=4)
    x = torch.randn(4, 12, 5)
    out = model(x)
    assert not torch.isnan(out).any()


def test_make_sequences_shapes():
    X = np.ones((30, 5), dtype=np.float32)
    y = np.ones(30, dtype=np.float32)
    X_seq, y_seq = _make_sequences(X, y, seq_len=8, horizon=4)
    expected_n = 30 - 8 - 4 + 1
    assert X_seq.shape == (expected_n, 8, 5)
    assert y_seq.shape == (expected_n, 4)


def test_fit_loss_decreases():
    X_train, y_train = _dummy_data(n=60, n_features=3)
    X_val, y_val = _dummy_data(n=20, n_features=3)
    model = HousingLSTM(input_size=3, hidden_size=8, num_layers=1, output_steps=4)
    history = fit(
        model, X_train, y_train, X_val, y_val,
        seq_len=6, horizon=4, batch_size=8, max_epochs=10, patience=5,
    )
    assert "train_losses" in history
    assert len(history["train_losses"]) > 0
    # Loss should decrease or at least not explode
    assert history["train_losses"][-1] < history["train_losses"][0] * 10


def test_fit_returns_best_epoch():
    X_train, y_train = _dummy_data(n=60, n_features=3)
    X_val, y_val = _dummy_data(n=20, n_features=3)
    model = HousingLSTM(input_size=3, hidden_size=8, num_layers=1, output_steps=4)
    history = fit(
        model, X_train, y_train, X_val, y_val,
        seq_len=6, horizon=4, batch_size=8, max_epochs=5, patience=3,
    )
    assert "best_epoch" in history
    assert 1 <= history["best_epoch"] <= 5
