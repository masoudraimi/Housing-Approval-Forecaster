# Housing Approvals Forecaster

## The Problem

At PIA Congress 2026, Auckland planner John Duguid reported that the Auckland Unitary Plan lifted consented dwellings per capita from 5.9 to 9.3 — a measurable supply response to planning reform. Australia's National Housing Accord targets 1.2 million new homes over five years. The question every planner, analyst, and infrastructure team is now asking: are Australian LGAs on track?

That is a forecasting problem. This project builds a production-grade answer: a PyTorch LSTM trained on ABS building approvals data, tracked with MLflow, served via FastAPI, and monitored for model drift. It demonstrates the full MLOps lifecycle that sits between a research model and a production deployment.

**Feature design is grounded in the supply-demand framework from housing research:** population growth (migration-driven demand), construction costs (delivery constraint), planning policy shifts, and autoregressive approval momentum. Monetary policy is absent by design — cash rate is not a model feature.

## What This Predicts

Quarterly dwelling approvals per Local Government Area (LGA), 4-quarter horizon.

- **Input:** 8 features per LGA — `approvals_lag1`, `population_growth_yoy` (ABS ERP), `construction_cost_yoy` (ABS PPI), seasonal dummies, `post_accord_2022`
- **Output:** Predicted dwelling approvals for the next 4 quarters
- **Granularity:** LGA-level (528 LGAs, 2019–2025)
- **Evaluation metrics:** MAE, MAPE, and a custom on-target hit rate (within 15% of actuals)

## Architecture

```
ABS LGA2020/2021             ABS ERP (population)       ABS PPI (construction)
(building approvals)         measure ERP_P_20            Table 17: house construction
        |                           |                           |
        +-----------> data/pipeline.py <-----------------------+
                               |
                        features.parquet
                        3,371 rows × 13 cols
                               |
             +-----------------+------------------+
             |                 |                  |
training/train_baseline.py     |     training/train_lstm.py
(SARIMA + seasonal mean)       |     (PyTorch LSTM, 8 features)
             |                 |                  |
             +---------+-------+----------+--------+
                       |
                MLflow Registry
                @champion alias
                       |
              serving/forecaster.py
                       |
              FastAPI (port 8000)
              /forecast  /health  /info
                       |
              serving/logger.py
              (SQLite prediction log)
                       |
           monitoring/drift.py
           feature drift (construction cost z-score)
           + residual drift (rolling MAE vs baseline)
```

## Model Comparison

Results on held-out test set (post 2022Q2, post-Accord period):

| Model | MAE (dwellings) | MAPE | Hit Rate (±15%) |
|---|---|---|---|
| Seasonal Mean | **101.9** | **0.548** | **28.0%** |
| SARIMA(1,1,1)(0,1,1,4) | 163.0 | 0.834 | 21.2% |
| PyTorch LSTM (8 features) | 192.6 | 4.822 | 13.6% |

The seasonal mean baseline wins on all three metrics. See `notebooks/03_project_showcase.ipynb` for a full explanation of why — evaluation asymmetry, dominant lag signal, and cross-sectional LSTM training are the main factors.

## The 2022 Policy-Shift Drift Event

The National Housing Accord was announced in August 2022 (2022Q3). Around the same time, post-COVID construction cost inflation peaked, with house construction PPI growing at ~15–20% YoY — far outside the pre-2022 training distribution. Any model trained on pre-Accord data faces a structural break at this point.

This project's monitoring layer catches that in two ways:

1. **Feature drift** (immediate): construction cost YoY moves outside the training distribution. The z-score can breach the 2.5σ threshold in the same quarter as the cost shock, without needing ground truth.
2. **Residual drift** (lagged): rolling MAE exceeds 1.5× the training baseline. This typically triggers 1–2 quarters after the feature drift flag, once actual approvals data accumulates.

See `notebooks/02_drift_case_study.ipynb` for the full case study with annotated charts.

## Quick Start

### With Docker Compose

```bash
cp .env.example .env
docker compose up
# MLflow UI: http://localhost:5000
# API:       http://localhost:8000
```

### Local Development

```bash
uv sync

# Download all data (ABS building approvals, population, PPI construction)
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
    "lga_code": "24600",
    "features": {
      "approvals_lag1": 320,
      "population_growth_yoy": 0.015,
      "construction_cost_yoy": 0.05,
      "season_q1": 0,
      "season_q2": 1,
      "season_q3": 0,
      "season_q4": 0,
      "post_accord_2022": 1
    }
  }'
```

Health check:

```bash
curl http://localhost:8000/health
```

## Data Sources

| Source | Content | Granularity | Used for |
|---|---|---|---|
| ABS Regional LGA2020/2021 | Quarterly dwelling approvals (BUILDING_4) | LGA | Target variable + `approvals_lag1` |
| ABS Regional LGA2020/2021 | Estimated Resident Population (ERP_P_20) | LGA | `population_growth_yoy` |
| ABS PPI 6427.0 Table 17 | House construction output price index | National | `construction_cost_yoy` |

All sources are free, publicly available, and require no authentication. The ABS Regional CSVs are large (~300 MB each) but are downloaded once and reused for both approvals and population.

## Why This Exists

Deploying an ML model is not the same as training one. This project is the production engineering layer that comes after the research: experiment tracking to compare multiple models objectively, a model registry to manage versioning and promotion, a typed API to serve predictions at request time, a prediction log to accumulate ground truth, and a monitoring layer to detect when the model has stopped working.

The 2022 policy-shift event is a real, dateable structural break. The feature drift signal (construction cost z-score) can trigger in the same quarter as the break — before performance visibly degrades. That is the production value of monitoring: early warning, not post-mortem.
