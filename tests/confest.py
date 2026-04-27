"""Pytest shared fixtures."""

import sys
from pathlib import Path

# Allow imports from src/ without editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from air_quality.data import generate_synthetic_data, Preprocessor
from air_quality.features import FeatureEngineer


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return generate_synthetic_data(n_days=400, seed=0)


@pytest.fixture(scope="session")
def splits(raw_df):
    prep = Preprocessor(train_ratio=0.7, val_ratio=0.15)
    return prep.fit_transform(raw_df)


@pytest.fixture(scope="session")
def engineered(splits):
    train_raw, val_raw, test_raw = splits
    fe = FeatureEngineer()
    train_fe = fe.fit_transform(train_raw)
    val_fe = fe.transform(val_raw).dropna()
    test_fe = fe.transform(test_raw).dropna()
    return fe, train_fe, val_fe, test_fe