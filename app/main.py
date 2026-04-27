"""FastAPI inference service.

Endpoints
---------
GET  /health          – liveness probe
GET  /info            – model metadata
POST /predict         – single-window PM2.5 forecast
POST /predict/batch   – batch forecast

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from air_quality.config import settings
from air_quality.data import generate_synthetic_data, Preprocessor
from air_quality.features import FeatureEngineer
from air_quality.models import XGBoostForecaster

logger = logging.getLogger(__name__)

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Air Quality Forecasting API",
    description=(
        "PM2.5 forecasting for Delhi using XGBoost, LightGBM, and LSTM. "
        "Backed by an OpenAQ data pipeline with MLflow experiment tracking."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory model store (loaded on startup) ─────────────────────────────────
_model: Optional[XGBoostForecaster] = None
_feature_engineer: Optional[FeatureEngineer] = None
_preprocessor: Optional[Preprocessor] = None
_training_df: Optional[pd.DataFrame] = None


@app.on_event("startup")
async def startup_event() -> None:
    global _model, _feature_engineer, _preprocessor, _training_df
    logger.info("Loading model on startup…")

    model_path = settings.models_dir / "XGBoostForecaster.model"

    _training_df = generate_synthetic_data()
    _preprocessor = Preprocessor()
    _feature_engineer = FeatureEngineer()

    train_raw, val_raw, _ = _preprocessor.fit_transform(_training_df)
    train_fe = _feature_engineer.fit_transform(train_raw)
    val_fe = _feature_engineer.transform(val_raw).dropna()

    feature_cols = _feature_engineer.feature_columns
    X_tr, y_tr = train_fe[feature_cols], train_fe["pm25"]
    X_val, y_val = val_fe[feature_cols], val_fe["pm25"]

    _model = XGBoostForecaster(n_estimators=300)

    if model_path.exists():
        _model.load(model_path)
        logger.info("Model loaded from %s", model_path)
    else:
        logger.info("No saved model found — training fresh XGBoost model…")
        _model.fit(X_tr, y_tr, X_val, y_val)
        _model.save(model_path)

    logger.info("API ready.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    temperature_c: float = Field(..., ge=-20, le=55, description="Air temperature in °C")
    humidity_pct: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    wind_kmh: float = Field(..., ge=0, le=150, description="Wind speed (km/h)")
    rainfall_mm: float = Field(0.0, ge=0, description="Daily rainfall (mm)")
    day_of_year: int = Field(..., ge=1, le=366)
    month: int = Field(..., ge=1, le=12)
    year: int = Field(2024, ge=2000, le=2100)
    # optional lag/rolling features; if omitted they are imputed from training mean
    pm25_lag_1: Optional[float] = None
    pm25_lag_7: Optional[float] = None
    pm25_roll_mean_7: Optional[float] = None


class PredictResponse(BaseModel):
    pm25_forecast: float
    aqi_category: str
    model: str = "XGBoost"


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    forecasts: List[PredictResponse]


# ── Helpers ───────────────────────────────────────────────────────────────────

AQI_BINS = [0, 12, 35.4, 55.4, 150.4, 250.4, 1e9]
AQI_LABELS = [
    "Good", "Moderate", "Unhealthy for Sensitive Groups",
    "Unhealthy", "Very Unhealthy", "Hazardous",
]


def _pm25_to_aqi_category(pm25: float) -> str:
    for lo, hi, label in zip(AQI_BINS, AQI_BINS[1:], AQI_LABELS):
        if lo <= pm25 < hi:
            return label
    return "Hazardous"


def _build_feature_row(req: PredictRequest) -> pd.DataFrame:
    assert _feature_engineer is not None
    assert _training_df is not None

    # Impute missing lag/rolling features from training set means
    train_means = _training_df[["pm25"]].mean()
    pm25_mean = float(train_means["pm25"])

    row = {
        "temperature_c": req.temperature_c,
        "humidity_pct": req.humidity_pct,
        "wind_kmh": req.wind_kmh,
        "rainfall_mm": req.rainfall_mm,
        "day_of_year": req.day_of_year,
        "month": req.month,
        "year": req.year,
        "pm25": pm25_mean,  # dummy — not used as target
    }

    # Build all features from a small synthetic context window
    import numpy as np

    n_ctx = 40
    ctx_df = generate_synthetic_data(n_days=n_ctx)
    # Overwrite last row with request values
    for k, v in row.items():
        if k in ctx_df.columns:
            ctx_df.loc[ctx_df.index[-1], k] = v

    feat_df = _feature_engineer.transform(ctx_df)
    feat_cols = _feature_engineer.feature_columns
    return feat_df[feat_cols].iloc[[-1]]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None and _model._is_fitted}


@app.get("/info", tags=["system"])
async def info() -> dict:
    n_features = len(_feature_engineer.feature_columns) if _feature_engineer else 0
    return {
        "model": "XGBoostForecaster",
        "version": "1.0.0",
        "target": "pm25",
        "n_features": n_features,
        "city": "Delhi, IN",
    }


@app.post("/predict", response_model=PredictResponse, tags=["forecast"])
async def predict(req: PredictRequest) -> PredictResponse:
    if _model is None or not _model._is_fitted:
        raise HTTPException(status_code=503, detail="Model not ready.")

    X = _build_feature_row(req)
    pm25_pred = float(np.clip(_model.predict(X)[0], 0, 600))
    return PredictResponse(
        pm25_forecast=round(pm25_pred, 2),
        aqi_category=_pm25_to_aqi_category(pm25_pred),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["forecast"])
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    if _model is None or not _model._is_fitted:
        raise HTTPException(status_code=503, detail="Model not ready.")

    forecasts = []
    for record in req.records:
        X = _build_feature_row(record)
        pm25_pred = float(np.clip(_model.predict(X)[0], 0, 600))
        forecasts.append(
            PredictResponse(
                pm25_forecast=round(pm25_pred, 2),
                aqi_category=_pm25_to_aqi_category(pm25_pred),
            )
        )
    return BatchPredictResponse(forecasts=forecasts)