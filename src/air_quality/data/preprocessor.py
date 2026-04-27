"""Data cleaning and preprocessing with Polars for performance."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class Preprocessor:
    """Clean, validate, and split raw air-quality data.

    Uses Polars for fast column operations and returns pandas DataFrames
    for downstream scikit-learn / PyTorch compatibility.
    """

    PM25_FLOOR = 0.5        # physical minimum µg/m³
    PM25_CEILING = 600.0    # instrument saturation ceiling

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        target_col: str = "pm25",
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_col = target_col

    # ------------------------------------------------------------------
    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Clean *df* and return (train, val, test) splits (no shuffle)."""
        lf = pl.from_pandas(df).lazy()
        lf = self._clean(lf)
        clean_df = lf.collect().to_pandas()
        return self._split(clean_df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same cleaning without splitting (for inference)."""
        lf = pl.from_pandas(df).lazy()
        return self._clean(lf).collect().to_pandas()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply cleaning rules in a single lazy pass."""
        pm = self.target_col

        lf = (
            lf
            # Parse date column to Date type
            .with_columns(pl.col("date").cast(pl.Date))
            # Sort chronologically
            .sort("date")
            # Clamp extreme PM2.5 values
            .with_columns(
                pl.col(pm).clip(self.PM25_FLOOR, self.PM25_CEILING).alias(pm)
            )
            # Drop rows where target is null
            .filter(pl.col(pm).is_not_null())
            # Forward-fill meteorological nulls (at most 2 consecutive days)
            .with_columns(
                [
                    pl.col(c).forward_fill(limit=2)
                    for c in ["temperature_c", "humidity_pct", "wind_kmh", "rainfall_mm"]
                    if c in lf.columns
                ]
            )
            # Drop any remaining rows with nulls in key columns
            .drop_nulls(
                subset=[pm]
            )
        )
        return lf

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train = df.iloc[:n_train].copy()
        val = df.iloc[n_train : n_train + n_val].copy()
        test = df.iloc[n_train + n_val :].copy()

        logger.info(
            "Split → train=%d  val=%d  test=%d  (total=%d)",
            len(train), len(val), len(test), n,
        )
        return train, val, test