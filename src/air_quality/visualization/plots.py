"""Plotly chart factory for the Streamlit dashboard and notebooks."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

AQI_PALETTE = {
    "Good": "#00C853",
    "Moderate": "#FFD600",
    "Unhealthy for Sensitive": "#FF6D00",
    "Unhealthy": "#D50000",
    "Very Unhealthy": "#8E24AA",
    "Hazardous": "#37474F",
}

BRAND_BLUE = "#1565C0"
BRAND_RED = "#B71C1C"
BRAND_GREY = "#546E7A"


def plot_time_series(df: pd.DataFrame, title: str = "PM2.5 Time Series") -> go.Figure:
    """Full-width time-series with WHO and NAAQS reference lines."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["pm25"],
            mode="lines",
            line=dict(color=BRAND_BLUE, width=1.5),
            name="PM2.5 (daily)",
            hovertemplate="%{x|%d %b %Y}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        )
    )

    # 7-day rolling average
    roll7 = df["pm25"].rolling(7, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=roll7,
            mode="lines",
            line=dict(color="#FFA726", width=2.5, dash="dot"),
            name="7-day rolling avg",
        )
    )

    # Reference lines
    for level, color, label in [
        (15, "#00C853", "WHO guideline (15)"),
        (60, "#FFD600", "NAAQS 24-hr (60)"),
        (250, BRAND_RED, "Hazardous (250)"),
    ]:
        fig.add_hline(
            y=level, line_dash="dash", line_color=color,
            annotation_text=label, annotation_position="top right",
            annotation_font_size=11,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    return fig


def plot_forecast_vs_actual(
    dates: pd.Series,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> go.Figure:
    """Overlay multiple model forecasts against actual values."""
    colours = [BRAND_BLUE, BRAND_RED, "#2E7D32", "#6A1B9A"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates, y=y_true,
            mode="lines",
            line=dict(color=BRAND_GREY, width=2),
            name="Actual PM2.5",
        )
    )

    for (name, y_pred), colour in zip(predictions.items(), colours):
        mask = ~np.isnan(y_pred)
        fig.add_trace(
            go.Scatter(
                x=dates[mask], y=y_pred[mask],
                mode="lines",
                line=dict(color=colour, width=1.8, dash="dot"),
                name=name,
                opacity=0.85,
            )
        )

    fig.update_layout(
        title="Forecast vs. Actual PM2.5 (Test Period)",
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    return fig


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importances",
) -> go.Figure:
    top = importance.nlargest(top_n).sort_values()
    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
            marker_color=BRAND_BLUE,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance score",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, top_n * 22),
    )
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    residuals = y_pred - y_true
    mask = ~np.isnan(residuals)
    r, p = residuals[mask], y_pred[mask]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residuals over Predicted", "Residual Distribution"],
    )
    fig.add_trace(
        go.Scatter(
            x=p, y=r, mode="markers",
            marker=dict(color=BRAND_BLUE, opacity=0.5, size=4),
            name="Residuals",
            hovertemplate="Pred: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=BRAND_RED, row=1, col=1)  # type: ignore

    fig.add_trace(
        go.Histogram(x=r, nbinsx=40, marker_color=BRAND_BLUE, opacity=0.75, name="Histogram"),
        row=1, col=2,
    )

    fig.update_layout(
        title_text=f"Residual Analysis — {model_name}",
        showlegend=False,
        template="plotly_white",
    )
    return fig


def plot_aqi_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["aqi_category"].value_counts()
    colours = [AQI_PALETTE.get(str(c), "#888") for c in counts.index]
    fig = px.pie(
        values=counts.values,
        names=counts.index.astype(str),
        color=counts.index.astype(str),
        color_discrete_map=AQI_PALETTE,
        title="AQI Category Distribution",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white", showlegend=True)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont_size=10,
            hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_monthly_seasonality(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.groupby("month")["pm25"]
        .agg(["mean", "std"])
        .reset_index()
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly["month"].map(lambda m: month_labels[m - 1]),
            y=monthly["mean"],
            error_y=dict(type="data", array=monthly["std"], visible=True),
            marker_color=[
                AQI_PALETTE["Hazardous"] if v > 250 else
                AQI_PALETTE["Very Unhealthy"] if v > 150 else
                AQI_PALETTE["Unhealthy"] if v > 55 else
                AQI_PALETTE["Moderate"] if v > 35 else
                AQI_PALETTE["Good"]
                for v in monthly["mean"]
            ],
            name="Monthly mean ± 1 SD",
            hovertemplate="%{x}: %{y:.1f} µg/m³<extra></extra>",
        )
    )
    fig.update_layout(
        title="Monthly PM2.5 Seasonality",
        xaxis_title="Month",
        yaxis_title="Mean PM2.5 (µg/m³)",
        template="plotly_white",
    )
    return fig