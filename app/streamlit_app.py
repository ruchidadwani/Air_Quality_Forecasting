"""Streamlit dashboard — Air Quality Forecasting System.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from src/ without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from air_quality.data import generate_synthetic_data, Preprocessor
from air_quality.evaluation import evaluate
from air_quality.features import FeatureEngineer
from air_quality.models import XGBoostForecaster, LightGBMForecaster
from air_quality.visualization import (
    plot_time_series,
    plot_forecast_vs_actual,
    plot_feature_importance,
    plot_residuals,
    plot_aqi_distribution,
    plot_correlation_heatmap,
    plot_monthly_seasonality,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecasting · Delhi",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .metric-card {
            background: #F8F9FA;
            border-radius: 12px;
            padding: 1rem 1.4rem;
            border-left: 4px solid #1565C0;
        }
        .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        "Forecasting model",
        ["XGBoost", "LightGBM", "Both (ensemble)"],
    )
    year_filter = st.multiselect(
        "Filter years",
        options=[2021, 2022, 2023],
        default=[2021, 2022, 2023],
    )
    n_estimators = st.slider("n_estimators", 100, 800, 300, step=50)
    max_depth = st.slider("max_depth (XGBoost)", 3, 10, 6)
    num_leaves = st.slider("num_leaves (LightGBM)", 20, 150, 63, step=5)

    st.markdown("---")
    st.markdown(
        "**Data source:** OpenAQ synthetic proxy  \n"
        "**City:** Delhi, India  \n"
        "**Pollutant:** PM2.5 (µg/m³)"
    )


# ── data + caching ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return generate_synthetic_data(n_days=1095)


@st.cache_resource(show_spinner="Training models…")
def train_models(
    _df: pd.DataFrame,
    model_choice: str,
    n_estimators: int,
    max_depth: int,
    num_leaves: int,
) -> dict:
    prep = Preprocessor()
    fe = FeatureEngineer()

    train_raw, val_raw, test_raw = prep.fit_transform(_df)
    train_fe = fe.fit_transform(train_raw)
    val_fe = fe.transform(val_raw).dropna()
    test_fe = fe.transform(test_raw).dropna()

    feat_cols = fe.feature_columns
    target = "pm25"

    X_tr, y_tr = train_fe[feat_cols], train_fe[target]
    X_val, y_val = val_fe[feat_cols], val_fe[target]
    X_te, y_te = test_fe[feat_cols], test_fe[target]

    results: dict = {
        "X_test": X_te,
        "y_test": y_te,
        "dates_test": test_fe["date"] if "date" in test_fe.columns else pd.Series(),
        "feature_cols": feat_cols,
        "predictions": {},
        "metrics": {},
        "importances": {},
    }

    models_to_train = []
    if model_choice in ("XGBoost", "Both (ensemble)"):
        models_to_train.append(("XGBoost", XGBoostForecaster(
            n_estimators=n_estimators, max_depth=max_depth)))
    if model_choice in ("LightGBM", "Both (ensemble)"):
        models_to_train.append(("LightGBM", LightGBMForecaster(
            n_estimators=n_estimators, num_leaves=num_leaves)))

    for name, model in models_to_train:
        model.fit(X_tr, y_tr, X_val, y_val)
        y_pred = model.predict(X_te)
        results["predictions"][name] = y_pred
        results["metrics"][name] = evaluate(y_te.values, y_pred)
        results["importances"][name] = model.feature_importance()

    if model_choice == "Both (ensemble)":
        preds = list(results["predictions"].values())
        ensemble = np.mean(preds, axis=0)
        results["predictions"]["Ensemble"] = ensemble
        results["metrics"]["Ensemble"] = evaluate(y_te.values, ensemble)

    return results


# ── load everything ──────────────────────────────────────────────────────────
df_full = load_data()
if year_filter:
    df = df_full[df_full["year"].isin(year_filter)].copy()
else:
    df = df_full.copy()

cache_key = (model_choice, n_estimators, max_depth, num_leaves)
results = train_models(df_full, model_choice, n_estimators, max_depth, num_leaves)

# ── header ───────────────────────────────────────────────────────────────────
st.title("🌫️ Air Quality Forecasting · Delhi")
st.markdown(
    "End-to-end ML pipeline — **XGBoost · LightGBM · Bidirectional LSTM** · "
    "Optuna tuning · MLflow tracking · FastAPI inference"
)
st.markdown("---")

# ── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg PM2.5", f"{df['pm25'].mean():.1f} µg/m³")
c2.metric("Max PM2.5", f"{df['pm25'].max():.1f} µg/m³")
hazardous = int((df["aqi_category"] == "Hazardous").sum())
c3.metric("Hazardous days", hazardous)
good = int((df["aqi_category"] == "Good").sum())
c4.metric("Good days", good)
# Best model R²
best_r2 = max(m.r2 for m in results["metrics"].values()) if results["metrics"] else 0.0
c5.metric("Best Model R²", f"{best_r2:.3f}")

st.markdown("---")

# ── tabs ─────────────────────────────────────────────────────────────────────
tab_eda, tab_forecast, tab_analysis, tab_about = st.tabs(
    ["📊 Exploratory Analysis", "🔮 Forecasting", "🔬 Model Analysis", "ℹ️ About"]
)

# ── EDA TAB ──────────────────────────────────────────────────────────────────
with tab_eda:
    st.plotly_chart(
        plot_time_series(df, title="Delhi PM2.5 Daily Levels"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_monthly_seasonality(df),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            plot_aqi_distribution(df),
            use_container_width=True,
        )

    st.plotly_chart(
        plot_correlation_heatmap(
            df[["pm25", "temperature_c", "humidity_pct", "wind_kmh", "rainfall_mm"]]
        ),
        use_container_width=True,
    )

# ── FORECASTING TAB ──────────────────────────────────────────────────────────
with tab_forecast:
    if not results["metrics"]:
        st.warning("No models trained. Check sidebar configuration.")
    else:
        # Metrics table
        metrics_rows = []
        for model_name, m in results["metrics"].items():
            row = {"Model": model_name} | {k.upper(): round(v, 3) for k, v in m.to_dict().items()}
            metrics_rows.append(row)
        metrics_df = pd.DataFrame(metrics_rows).set_index("Model")

        st.subheader("Test-set Performance")
        st.dataframe(
            metrics_df.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color="#C8E6C9")
                             .highlight_max(subset=["R2", "SKILL_SCORE"], color="#C8E6C9"),
            use_container_width=True,
        )

        # Forecast chart
        st.plotly_chart(
            plot_forecast_vs_actual(
                results["dates_test"].reset_index(drop=True),
                results["y_test"].values,
                results["predictions"],
            ),
            use_container_width=True,
        )

# ── MODEL ANALYSIS TAB ───────────────────────────────────────────────────────
with tab_analysis:
    if not results["importances"]:
        st.info("Train a tree-based model to see feature importance.")
    else:
        for model_name, imp in results["importances"].items():
            if imp is not None:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(
                        plot_feature_importance(imp, top_n=15, title=f"{model_name} — Feature Importance"),
                        use_container_width=True,
                    )
                with col_b:
                    y_pred = results["predictions"][model_name]
                    st.plotly_chart(
                        plot_residuals(results["y_test"].values, y_pred, model_name),
                        use_container_width=True,
                    )

# ── ABOUT TAB ────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        ## About this project

        This dashboard is the frontend for a production-grade **air quality
        forecasting** system targeting Delhi's PM2.5 pollution.

        ### Stack
        | Layer | Technology |
        |---|---|
        | Data ingestion | OpenAQ API v3 / synthetic fallback |
        | Data processing | Polars + Pandas |
        | Feature engineering | Lag features, rolling stats, cyclic temporal encodings |
        | ML models | XGBoost · LightGBM · Bidirectional LSTM (PyTorch) |
        | Hyper-parameter search | Optuna (TPE sampler) |
        | Experiment tracking | MLflow |
        | Inference API | FastAPI + Uvicorn |
        | Dashboard | Streamlit + Plotly |
        | CI/CD | GitHub Actions |
        | Containerisation | Docker + docker-compose |

        ### Metrics legend
        - **RMSE** — Root Mean Squared Error (µg/m³)
        - **MAE** — Mean Absolute Error (µg/m³)
        - **MAPE** — Mean Absolute Percentage Error (%)
        - **R²** — Coefficient of determination
        - **Skill score** — Improvement over persistence baseline (higher = better)
        """
    )