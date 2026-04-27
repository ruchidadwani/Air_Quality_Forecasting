"""Tests for data fetching and preprocessing."""

import pandas as pd
import pytest

from air_quality.data import generate_synthetic_data, Preprocessor


class TestSyntheticData:
    def test_shape(self):
        df = generate_synthetic_data(n_days=365)
        assert len(df) == 365

    def test_required_columns(self):
        df = generate_synthetic_data(n_days=100)
        for col in ["date", "pm25", "temperature_c", "humidity_pct", "wind_kmh", "rainfall_mm"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_pm25_bounds(self):
        df = generate_synthetic_data(n_days=500)
        assert df["pm25"].min() >= 0
        assert df["pm25"].max() <= 600

    def test_humidity_bounds(self):
        df = generate_synthetic_data(n_days=500)
        assert df["humidity_pct"].min() >= 0
        assert df["humidity_pct"].max() <= 100

    def test_aqi_categories_present(self):
        df = generate_synthetic_data(n_days=1095)
        cats = df["aqi_category"].dropna().unique()
        assert len(cats) > 1, "Expected multiple AQI categories"

    def test_reproducibility(self):
        df1 = generate_synthetic_data(n_days=100, seed=99)
        df2 = generate_synthetic_data(n_days=100, seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestPreprocessor:
    def test_split_sizes(self, raw_df):
        prep = Preprocessor(train_ratio=0.7, val_ratio=0.15)
        train, val, test = prep.fit_transform(raw_df)
        total = len(train) + len(val) + len(test)
        assert abs(total - len(raw_df)) <= 2  # allow minor rounding

    def test_no_nulls_in_target(self, splits):
        for split in splits:
            assert split["pm25"].isnull().sum() == 0

    def test_chronological_order(self, splits):
        for split in splits:
            assert split["date"].is_monotonic_increasing

    def test_transform_no_split(self, raw_df):
        prep = Preprocessor()
        result = prep.transform(raw_df)
        assert len(result) == len(raw_df)