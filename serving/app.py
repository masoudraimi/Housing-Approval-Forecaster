"""FastAPI application: /forecast, /batch_forecast, /health, /info endpoints."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from serving.forecaster import ForecastFeatures, ForecastResult, Forecaster
from serving.logger import PredictionLogger

app = FastAPI(
    title="Housing Approvals Forecaster",
    description="PyTorch LSTM forecaster for Australian quarterly dwelling approvals",
    version="0.1.0",
)

_start_time = time.time()
_logger: Optional[PredictionLogger] = None
_forecaster: Optional[Forecaster] = None


@app.on_event("startup")
def startup() -> None:
    global _logger, _forecaster
    _logger = PredictionLogger()
    _forecaster = Forecaster.get()


class ForecastRequest(BaseModel):
    lga_code: str
    features: ForecastFeatures


class BatchForecastRequest(BaseModel):
    requests: list[ForecastRequest]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float
    total_predictions: int


class InfoResponse(BaseModel):
    target_variable: str
    feature_columns: list[str]
    horizon_quarters: int
    model_name: str


@app.post("/forecast", response_model=ForecastResult)
def forecast(request: ForecastRequest) -> ForecastResult:
    """Return a 4-quarter dwelling approvals forecast for a single LGA."""
    try:
        result = _forecaster.predict(request.lga_code, request.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _logger.log(
        lga_code=request.lga_code,
        horizon_quarters=result.horizon_quarters,
        predicted_values=result.predicted_approvals,
        model_version=result.model_version,
    )
    return result


@app.post("/batch_forecast", response_model=list[ForecastResult])
def batch_forecast(request: BatchForecastRequest) -> list[ForecastResult]:
    """Return forecasts for up to 20 LGAs in a single request."""
    if len(request.requests) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 LGAs per batch request.")
    results = []
    for req in request.requests:
        try:
            result = _forecaster.predict(req.lga_code, req.features)
            _logger.log(
                lga_code=req.lga_code,
                horizon_quarters=result.horizon_quarters,
                predicted_values=result.predicted_approvals,
                model_version=result.model_version,
            )
            results.append(result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error for LGA {req.lga_code}: {exc}")
    return results


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check with model version and uptime."""
    return HealthResponse(
        status="ok",
        model_version=_forecaster.model_version if _forecaster else "not_loaded",
        uptime_seconds=round(time.time() - _start_time, 1),
        total_predictions=_logger.total_count() if _logger else 0,
    )


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Return model metadata: target variable, features, horizon."""
    from serving.forecaster import FEATURE_COLS, HORIZON, MODEL_NAME
    return InfoResponse(
        target_variable="dwellings_approved (quarterly, per LGA)",
        feature_columns=FEATURE_COLS,
        horizon_quarters=HORIZON,
        model_name=MODEL_NAME,
    )
