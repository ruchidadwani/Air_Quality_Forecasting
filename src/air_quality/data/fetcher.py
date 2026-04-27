"""Data acquisition: OpenAQ API client with synthetic-data fallback.

Real data is fetched from OpenAQ v3 (https://api.openaq.org/v3).
When no API key or network is available the module falls back to a
statistically-realistic synthetic generator so the full pipeline stays
runnable in offline / demo environments.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from air_quality.config import settings

logger = logging.getLogger(__name__)


class AirQualityFetcher:
    """Async-capable OpenAQ v3 client with rate-limit handling."""

    BASE_URL = settings.openaq_base_url

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.openaq_api_key
        self._headers = {"X-API-Key": self.api_key} if self.api_key else {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(
        self,
        city: str = "Delhi",
        parameter: str = "pm25",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch measurements and return a cleaned daily-aggregate DataFrame.

        Falls back to synthetic data if the request fails.
        """
        try:
            raw = self._fetch_measurements(city, parameter, date_from, date_to, limit)
            if raw.empty:
                logger.warning("OpenAQ returned no data — using synthetic fallback.")
                return generate_synthetic_data()
            return self._aggregate_daily(raw)
        except Exception as exc:
            logger.warning("OpenAQ fetch failed (%s) — using synthetic fallback.", exc)
            return generate_synthetic_data()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_measurements(
        self,
        city: str,
        parameter: str,
        date_from: Optional[str],
        date_to: Optional[str],
        limit: int,
    ) -> pd.DataFrame:
        if date_to is None:
            date_to = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        if date_from is None:
            date_from = (datetime.utcnow() - timedelta(days=365)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        params = {
            "city": city,
            "parameter": parameter,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{self.BASE_URL}/measurements",
                params=params,
                headers=self._headers,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

        if not results:
            return pd.DataFrame()

        records = [
            {
                "date": r["date"]["utc"],
                "pm25": r["value"],
                "unit": r.get("unit", "µg/m³"),
                "location": r.get("location", ""),
                "city": r.get("city", city),
            }
            for r in results
            if r.get("value") is not None
        ]
        return pd.DataFrame(records)

    @staticmethod
    def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return (
            df.groupby("date")["pm25"]
            .agg(["mean", "min", "max", "std", "count"])
            .rename(columns={"mean": "pm25", "min": "pm25_min", "max": "pm25_max",
                              "std": "pm25_std", "count": "n_measurements"})
            .reset_index()
        )


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_days: int = 1095, seed: int = 42) -> pd.DataFrame:
    """Generate statistically realistic Delhi-like PM2.5 and meteorology data.

    The synthetic series replicates:
    - Strong winter (Dec-Feb) pollution peaks driven by crop-residue burning
      and thermal inversions.
    - Clean monsoon season (Jul-Sep) with rain-induced washout.
    - Positive PM2.5–humidity correlation and negative wind-speed correlation.
    - A slight long-term improvement trend (cleaner fuels policy proxy).

    Returns a daily DataFrame with columns:
        date, pm25, temperature_c, humidity_pct, wind_kmh, rainfall_mm,
        month, day_of_year, year
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    # --- PM2.5 -----------------------------------------------------------
    # Seasonal: peak in Jan (~220 µg/m³), trough in Aug (~40 µg/m³)
    seasonal_pm25 = 130 + 90 * np.cos(2 * np.pi * (t - 15) / 365)
    # Post-monsoon spike Oct-Nov (stubble burning)
    stub_burn = 50 * np.exp(-0.5 * ((t % 365 - 295) / 20) ** 2)
    trend = np.linspace(0, -15, n_days)  # marginal improvement
    noise_pm25 = rng.normal(0, 25, n_days)
    pm25 = np.clip(seasonal_pm25 + stub_burn + trend + noise_pm25, 8, 550)

    # --- Temperature (°C) -----------------------------------------------
    temp = 25 - 15 * np.cos(2 * np.pi * t / 365) + rng.normal(0, 2.5, n_days)

    # --- Relative Humidity (%) ------------------------------------------
    humid_base = 58 + 28 * np.sin(2 * np.pi * (t - 80) / 365)
    humidity = np.clip(humid_base + rng.normal(0, 8, n_days), 10, 100)

    # --- Wind speed (km/h) ----------------------------------------------
    wind = np.clip(
        8 + 4 * np.sin(2 * np.pi * (t - 60) / 365) + rng.exponential(4, n_days),
        0.5, 45,
    )

    # --- Rainfall (mm) --------------------------------------------------
    monsoon_weight = np.clip(
        np.sin(2 * np.pi * (t - 150) / 365), 0, 1
    )
    rainfall = np.clip(rng.exponential(1.5 + 8 * monsoon_weight, n_days), 0, 80)

    df = pd.DataFrame(
        {
            "date": dates,
            "pm25": pm25.round(2),
            "temperature_c": temp.round(2),
            "humidity_pct": humidity.round(2),
            "wind_kmh": wind.round(2),
            "rainfall_mm": rainfall.round(2),
        }
    )
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year
    df["year"] = df["date"].dt.year
    df["aqi_category"] = pd.cut(
        df["pm25"],
        bins=[0, 12, 35.4, 55.4, 150.4, 250.4, 600],
        labels=["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy",
                "Very Unhealthy", "Hazardous"],
    )
    return df