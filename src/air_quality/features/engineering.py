"""Feature engineering: lag features, rolling statistics, cyclic encodings."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LAGS = [1, 2, 3, 7, 14, 21]
DEFAULT_WINDOWS = [3, 7, 14, 30]


class FeatureEngineer:
    """Transform raw daily data into a rich ML feature matrix.

    Features created
    ----------------
    Lag features        pm25_lag_{k}           k in lag_days
    Rolling mean        pm25_roll_mean_{w}     w in rolling_windows
    Rolling std         pm25_roll_std_{w}
    Rolling min/max     pm25_roll_min/max_{w}
    Cyclic day-of-year  doy_sin, doy_cos
    Cyclic month        month_sin, month_cos
    Meteorological      temperature_c, humidity_pct, wind_kmh, rainfall_mm
    Derived             temp_humidity_idx, wind_pollution_idx
    """

    def __init__(
        self,
        target_col: str = "pm25",
        lag_days: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ) -> None:
        self.target_col = target_col
        self.lag_days = lag_days or DEFAULT_LAGS
        self.rolling_windows = rolling_windows or DEFAULT_WINDOWS
        self.feature_names_: List[str] = []

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features and drop rows with NaN (warm-up period)."""
        return self._build(df).dropna().reset_index(drop=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same transforms without dropping NaNs (inference path)."""
        return self._build(df)

    @property
    def feature_columns(self) -> List[str]:
        return [c for c in self.feature_names_ if c not in (self.target_col, "date")]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.sort_values("date").reset_index(drop=True)
        target = out[self.target_col]

        # ── lag features ────────────────────────────────────────────────
        for k in self.lag_days:
            out[f"pm25_lag_{k}"] = target.shift(k)

        # ── rolling statistics ──────────────────────────────────────────
        for w in self.rolling_windows:
            roll = target.shift(1).rolling(w, min_periods=w // 2)
            out[f"pm25_roll_mean_{w}"] = roll.mean()
            out[f"pm25_roll_std_{w}"] = roll.std()
            out[f"pm25_roll_min_{w}"] = roll.min()
            out[f"pm25_roll_max_{w}"] = roll.max()

        # ── cyclic temporal encodings ────────────────────────────────────
        doy = out["day_of_year"] if "day_of_year" in out.columns else out["date"].dt.day_of_year
        month = out["month"] if "month" in out.columns else out["date"].dt.month
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)

        # ── derived meteorological indices ───────────────────────────────
        if "temperature_c" in out.columns and "humidity_pct" in out.columns:
            # Heat-index proxy (higher = more stable boundary layer)
            out["temp_humidity_idx"] = out["temperature_c"] * out["humidity_pct"] / 100
        if "wind_kmh" in out.columns:
            # Dispersion index: higher wind → better pollutant dispersion
            out["wind_pollution_idx"] = 1 / (out["wind_kmh"].clip(lower=0.5))

        self.feature_names_ = [
            c for c in out.columns if c not in ("date", "aqi_category")
        ]
        return out