"""Tests for serving/app.py FastAPI endpoints via httpx test client."""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from serving.forecaster import ForecastResult

_MOCK_RESULT = ForecastResult(
    lga_code="LGA12345",
    horizon_quarters=4,
    predicted_approvals=[120.0, 115.0, 130.0, 125.0],
    model_version="1",
    confidence_note="Test prediction",
)


@pytest.fixture
def client() -> Generator:
    mock_logger = MagicMock()
    mock_logger.total_count.return_value = 5

    mock_forecaster = MagicMock()
    mock_forecaster.model_version = "1"
    mock_forecaster.predict.return_value = _MOCK_RESULT

    with patch("serving.app.PredictionLogger", return_value=mock_logger), \
         patch("serving.app.Forecaster.get", return_value=mock_forecaster):
        from serving.app import app
        with TestClient(app) as c:
            yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "uptime_seconds" in data


def test_forecast_valid_request(client):
    payload = {
        "lga_code": "LGA12345",
        "features": {
            "approvals_lag1": 120.0,
            "population_growth_yoy": 0.015,
            "construction_cost_yoy": 0.05,
            "season_q2": 1,
            "post_accord_2022": 1,
        },
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["lga_code"] == "LGA12345"
    assert len(data["predicted_approvals"]) == 4
    assert "model_version" in data


def test_forecast_returns_model_version(client):
    payload = {
        "lga_code": "LGA99999",
        "features": {
            "approvals_lag1": 50.0,
            "population_growth_yoy": 0.005,
            "construction_cost_yoy": 0.02,
        },
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    assert response.json()["model_version"] == "1"


def test_info_endpoint(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "feature_columns" in data
    assert "horizon_quarters" in data
    assert data["horizon_quarters"] == 4


def test_batch_forecast_exceeds_limit(client):
    requests = [
        {
            "lga_code": f"LGA{i:05d}",
            "features": {
                "approvals_lag1": 100.0,
                "population_growth_yoy": 0.01,
                "construction_cost_yoy": 0.04,
            },
        }
        for i in range(25)
    ]
    response = client.post("/batch_forecast", json={"requests": requests})
    assert response.status_code == 400
