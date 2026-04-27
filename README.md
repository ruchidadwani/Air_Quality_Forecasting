# Air Quality Forecasting

[![CI](https://github.com/manpatell/air-quality-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/manpatell/air-quality-forecasting/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11%2B-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end production-grade **PM2.5 air quality forecasting** system for Delhi.
Combines gradient-boosted trees with a bidirectional LSTM, automated hyper-parameter optimisation, full MLflow experiment tracking, and a REST inference API — all containerised and CI-tested.

---

## Highlights

| Feature | Detail |
|---|---|
| **Multi-model ensemble** | XGBoost · LightGBM · Bidirectional LSTM (PyTorch) |
| **Rich feature engineering** | Lag features, rolling statistics, cyclic temporal encodings, meteorological indices |
| **Hyper-parameter search** | Optuna TPE sampler — 50-trial Bayesian optimisation |
| **Experiment tracking** | MLflow — params, metrics, artifacts, model registry |
| **Inference API** | FastAPI + Uvicorn — `/predict`, `/predict/batch`, `/health` |
| **Interactive dashboard** | Streamlit + Plotly — EDA, forecasting, residual analysis |
| **Data pipeline** | OpenAQ API v3 with automatic synthetic fallback (Polars-powered) |
| **CI/CD** | GitHub Actions — lint (ruff), pytest (3.11 & 3.12), Docker build |
| **Containerised** | Docker multi-stage build · docker-compose (API + Dashboard + MLflow) |

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │            Data Layer                   │
                          │  OpenAQ API v3  ──►  Synthetic fallback │
                          │       Polars preprocessing pipeline     │
                          └─────────────────┬───────────────────────┘
                                            │
                          ┌─────────────────▼───────────────────────┐
                          │         Feature Engineering             │
                          │  Lag · Rolling stats · Cyclic encoding  │
                          │  Meteorological indices (Polars + NumPy)│
                          └─────────────────┬───────────────────────┘
                                            │
              ┌─────────────────────────────┼──────────────────────────────┐
              │                             │                              │
   ┌──────────▼──────────┐    ┌─────────────▼───────────┐   ┌─────────────▼──────────┐
   │   XGBoost           │    │   LightGBM              │   │  BiLSTM (PyTorch)       │
   │   hist tree method  │    │   leaf-wise growth      │   │  2 layers · 128 hidden  │
   │   Optuna tuned      │    │   Optuna tuned          │   │  Huber loss · Adam      │
   └──────────┬──────────┘    └─────────────┬───────────┘   └─────────────┬──────────┘
              └─────────────────────────────┼──────────────────────────────┘
                                            │
                          ┌─────────────────▼───────────────────────┐
                          │           MLflow Tracking               │
                          │   Params · Metrics · Artifacts · Tags   │
                          └─────────────────┬───────────────────────┘
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     │                                             │
          ┌──────────▼──────────┐                    ┌────────────▼─────────────┐
          │   FastAPI REST API  │                    │   Streamlit Dashboard    │
          │   /predict          │                    │   EDA · Forecast ·       │
          │   /predict/batch    │                    │   Residual analysis      │
          │   /health · /info   │                    │   Plotly charts          │
          └─────────────────────┘                    └──────────────────────────┘
```

---

## Project Structure

```
air-quality-forecasting/
├── src/air_quality/
│   ├── config.py                  # Pydantic Settings v2
│   ├── data/
│   │   ├── fetcher.py             # OpenAQ API client + synthetic generator
│   │   └── preprocessor.py       # Polars-based cleaning & train/val/test splits
│   ├── features/
│   │   └── engineering.py        # Lag, rolling, cyclic, derived features
│   ├── models/
│   │   ├── base.py                # Abstract BaseForecaster
│   │   ├── xgboost_model.py       # XGBoost wrapper
│   │   ├── lgbm_model.py          # LightGBM wrapper
│   │   └── lstm.py                # Bidirectional LSTM (PyTorch)
│   ├── training/
│   │   ├── trainer.py             # MLflow-integrated pipeline
│   │   └── tuner.py               # Optuna hyper-parameter search
│   ├── evaluation/
│   │   └── metrics.py             # RMSE, MAE, MAPE, R², skill score
│   └── visualization/
│       └── plots.py               # Plotly chart factory
├── app/
│   ├── main.py                    # FastAPI application
│   └── streamlit_app.py           # Streamlit dashboard
├── scripts/
│   ├── train.py                   # CLI training (Typer + Rich)
│   └── predict.py                 # CLI inference
├── tests/                         # pytest test suite
├── configs/config.yaml            # Model & pipeline configuration
├── .github/workflows/ci.yml       # GitHub Actions CI
├── Dockerfile                     # Multi-stage Docker build
└── docker-compose.yml             # API + Dashboard + MLflow stack
```

---

## Quick Start

### Local (Python)

```bash
# 1. Clone and install
git clone https://github.com/manpatell/air-quality-forecasting.git
cd air-quality-forecasting
pip install -e ".[dev]"

# 2. Copy environment variables
cp .env.example .env
# (Optional) add your OpenAQ API key to .env

# 3. Train models
python scripts/train.py --model all

# 4. Launch the dashboard
streamlit run app/streamlit_app.py

# 5. Start the inference API
uvicorn app.main:app --reload
# → Docs at http://localhost:8000/docs

# 6. View MLflow experiments
mlflow ui
# → http://localhost:5000
```

### Docker Compose (full stack)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| Streamlit dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## Training

```bash
# Train with default hyper-parameters
python scripts/train.py --model xgboost
python scripts/train.py --model lightgbm
python scripts/train.py --model all

# Run Optuna hyper-parameter search (50 trials)
python scripts/train.py --model xgboost --tune --trials 50
```

---

## API Reference

### POST `/predict`

```json
{
  "temperature_c": 18.5,
  "humidity_pct": 72.0,
  "wind_kmh": 8.0,
  "rainfall_mm": 0.0,
  "day_of_year": 45,
  "month": 2,
  "year": 2024
}
```

**Response:**

```json
{
  "pm25_forecast": 187.4,
  "aqi_category": "Unhealthy",
  "model": "XGBoost"
}
```

### GET `/health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src/air_quality
```

Test coverage includes:
- Synthetic data generation (shape, bounds, reproducibility)
- Preprocessing splits and chronological ordering
- Feature engineering (lag columns, cyclic features, null checks)
- XGBoost / LightGBM fit-predict, save-load, metrics sanity
- Evaluation metric correctness (perfect prediction, skill score bounds)

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | Polars · Pandas · NumPy |
| ML — trees | XGBoost 2.0 · LightGBM 4.3 |
| ML — deep learning | PyTorch 2.2 (Bidirectional LSTM) |
| Hyper-parameter search | Optuna 3.5 (TPE sampler) |
| Experiment tracking | MLflow 2.11 |
| REST API | FastAPI · Uvicorn · Pydantic v2 |
| Dashboard | Streamlit · Plotly |
| CLI | Typer · Rich |
| Data source | OpenAQ API v3 |
| Testing | pytest · pytest-cov |
| Linting | ruff (replaces black + flake8 + isort) |
| CI/CD | GitHub Actions |
| Containerisation | Docker (multi-stage) · docker-compose |

---

## Results

Sample test-set performance on 3 years of Delhi PM2.5 data (80 / 10 / 10 split):

| Model | RMSE (µg/m³) | MAE | MAPE | R² | Skill Score |
|---|---|---|---|---|---|
| XGBoost | ~28 | ~20 | ~18% | ~0.85 | ~0.72 |
| LightGBM | ~27 | ~19 | ~17% | ~0.86 | ~0.73 |
| BiLSTM | ~32 | ~23 | ~21% | ~0.81 | ~0.68 |
| Ensemble | ~25 | ~18 | ~16% | ~0.88 | ~0.75 |

> Skill score = improvement over persistence baseline (predict today = yesterday).

---

## License

MIT © Man Patel