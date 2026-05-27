# Housing Approvals Forecaster

## The Problem

At PIA Congress 2026, Auckland planner John Duguid reported that the Auckland Unitary Plan lifted consented dwellings per capita from 5.9 to 9.3. Australia's National Housing Accord targets 1.2 million new homes over five years. The question every planner, banker, and infrastructure analyst is now asking: are Australian LGAs on track?

That is a forecasting problem. This project builds a production-grade answer: a PyTorch LSTM trained on ABS building approvals data, tracked with MLflow, served via FastAPI, and monitored for model drift. It demonstrates the full MLOps lifecycle that sits between a research model and a production deployment.

## What This Predicts

Quarterly dwelling approvals per Local Government Area (LGA), 4-quarter horizon.

- **Input:** 12-quarter history of dwelling approvals + macro features (RBA cash rate, ABS construction cost index, estimated resident population)
- **Output:** Predicted dwelling approvals for the next 4 quarters
- **Granularity:** LGA-level (consistent with Duguid's local government reform framing)
- **Evaluation metrics:** MAE, MAPE, and a custom on-target hit rate (within 15% of actuals)

## Architecture

```
ABS 8731.0 (building approvals)        RBA cash rate CSV
         |                                      |
         +-------------> data/pipeline.py <------+
                                |
                         features.parquet
                                |
              +-----------------+------------------+
              |                                    |
   training/train_baseline.py        training/train_lstm.py
   (SARIMA + seasonal mean)          (PyTorch LSTM custom loop)
              |                                    |
              +---------> MLflow Registry <--------+
                                |
                         @champion alias
                                |
                     serving/forecaster.py
                                |
                       FastAPI (port 8000)
                       /forecast, /health, /info
                                |
                       serving/logger.py
                       (SQLite prediction log)
                                |
                  monitoring/drift.py + report.py
                  (feature drift + residual drift)
```

## Model Comparison

Results on held-out test set (Q3 2022 onwards, post rate-hike period):

| Model | MAE | MAPE | Hit Rate (15%) |
|---|---|---|---|
| Seasonal Mean | TBD after training | TBD | TBD |
| SARIMA(1,1,1)(0,1,1,4) | TBD | TBD | TBD |
| PyTorch LSTM | TBD | TBD | TBD |

*Run `uv run python -m training.compare` to populate this table after training.*

## The 2022 Rate-Hike Drift Event

The RBA raised the cash rate 13 times between May 2022 and November 2023, from 0.1% to 4.35%. Any model trained on pre-2022 data has a structural break at Q3 2022 -- forecasts that assumed stable financing conditions began systematically missing actuals by Q1 2023.

This project's monitoring layer catches that in two ways:

1. **Feature drift** (immediate): the cash rate moves outside the training distribution. The z-score breaches the 2.5 threshold as early as Q3 2022.
2. **Residual drift** (lagged): rolling MAE exceeds 1.5x the training baseline. This typically triggers 1--2 quarters after the feature drift flag, once actual approvals data is available.

See `notebooks/02_drift_case_study.ipynb` for the full case study with annotated charts.

## Quick Start

### With Docker Compose

```bash
# Copy and edit environment variables
cp .env.example .env

# Start MLflow server and FastAPI service
docker compose up

# MLflow UI: http://localhost:5000
# API:       http://localhost:8000
```

### Local Development

```bash
# Install dependencies
uv sync

# Download data
uv run python -m data.download

# Build feature set
uv run python -m data.pipeline

# Train baselines
uv run python -m training.train_baseline

# Train LSTM
uv run python -m training.train_lstm

# Compare and promote champion
uv run python -m training.compare

# Start API server
uv run uvicorn serving.app:app --reload --port 8000
```

### Run Tests

```bash
uv run pytest --cov=. --cov-report=term-missing
```

## API Usage

Forecast next 4 quarters for a single LGA:

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "lga_code": "LGA24600",
    "features": {
      "approvals_lag1": 320,
      "approvals_lag2": 305,
      "approvals_lag3": 290,
      "approvals_lag4": 310,
      "cash_rate_lag1": 4.35,
      "cash_rate_lag2": 4.35,
      "post_rate_hike": 1,
      "season_q1": 0,
      "season_q2": 1,
      "season_q3": 0,
      "season_q4": 0
    }
  }'
```

Health check:

```bash
curl http://localhost:8000/health
```

## Data

| Source | Content | Granularity |
|---|---|---|
| ABS 8731.0 | Quarterly dwelling approvals | LGA |
| RBA cash rate | Monthly target rate | National |
| ABS PPI 6427.0 | House construction cost index | National |
| ABS ERP 3218.0 | Estimated resident population | LGA |

All sources are free, publicly available, and require no authentication.

## Why This Exists

Deploying an ML model is not the same as training one. This project is the production engineering layer that comes after the research: experiment tracking to compare multiple models objectively, a model registry to manage versioning and promotion, an API to serve predictions at request time, a prediction log to accumulate ground truth, and a monitoring layer to detect when the model has stopped working.

The 2022 rate-hike event is a real, dateable example of production model failure. The feature drift signal triggers before performance visibly degrades -- that is the production value of monitoring: early warning, not post-mortem.
