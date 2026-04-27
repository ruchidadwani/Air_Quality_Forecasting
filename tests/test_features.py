"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from air_quality.features import FeatureEngineer


class TestFeatureEngineer:
    def test_lag_columns_created(self, engineered):
        fe, train_fe, *_ = engineered
        for k in fe.lag_days:
            assert f"pm25_lag_{k}" in train_fe.columns

    def test_rolling_columns_created(self, engineered):
        fe, train_fe, *_ = engineered
        for w in fe.rolling_windows:
            assert f"pm25_roll_mean_{w}" in train_fe.columns
            assert f"pm25_roll_std_{w}" in train_fe.columns

    def test_cyclic_features(self, engineered):
        _, train_fe, *_ = engineered
        for col in ["doy_sin", "doy_cos", "month_sin", "month_cos"]:
            assert col in train_fe.columns
            # cyclic features should lie in [-1, 1]
            assert train_fe[col].between(-1.001, 1.001).all()

    def test_no_nulls_after_fit_transform(self, engineered):
        _, train_fe, *_ = engineered
        feat_cols = [c for c in train_fe.columns if c not in ("date", "aqi_category")]
        nulls = train_fe[feat_cols].isnull().sum().sum()
        assert nulls == 0, f"Found {nulls} NaN values after fit_transform"

    def test_feature_column_count(self, engineered):
        fe, train_fe, *_ = engineered
        assert len(fe.feature_columns) > 10, "Expected a rich feature set (>10 columns)"

    def test_derived_features(self, engineered):
        _, train_fe, *_ = engineered
        assert "temp_humidity_idx" in train_fe.columns
        assert "wind_pollution_idx" in train_fe.columns