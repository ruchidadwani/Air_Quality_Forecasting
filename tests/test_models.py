"""Tests for ML model wrappers."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from air_quality.evaluation import evaluate
from air_quality.models import LightGBMForecaster, XGBoostForecaster


def _xy(engineered, split_idx: int = 1):
    fe, train_fe, val_fe, test_fe = engineered
    cols = fe.feature_columns
    splits = [train_fe, val_fe, test_fe]
    df = splits[split_idx]
    return df[cols], df["pm25"]


class TestXGBoostForecaster:
    def test_fit_predict(self, engineered):
        fe, train_fe, val_fe, test_fe = engineered
        cols = fe.feature_columns
        model = XGBoostForecaster(n_estimators=50)
        model.fit(train_fe[cols], train_fe["pm25"], val_fe[cols], val_fe["pm25"])
        preds = model.predict(test_fe[cols])
        assert len(preds) == len(test_fe)
        assert np.isfinite(preds).all()

    def test_metrics_reasonable(self, engineered):
        fe, train_fe, val_fe, test_fe = engineered
        cols = fe.feature_columns
        model = XGBoostForecaster(n_estimators=100)
        model.fit(train_fe[cols], train_fe["pm25"], val_fe[cols], val_fe["pm25"])
        preds = model.predict(test_fe[cols])
        metrics = evaluate(test_fe["pm25"].values, preds)
        assert metrics.r2 > 0.5, f"R² too low: {metrics.r2}"
        assert metrics.rmse < 100, f"RMSE too high: {metrics.rmse}"

    def test_feature_importance_shape(self, engineered):
        fe, train_fe, val_fe, *_ = engineered
        cols = fe.feature_columns
        model = XGBoostForecaster(n_estimators=50)
        model.fit(train_fe[cols], train_fe["pm25"])
        imp = model.feature_importance()
        assert len(imp) == len(cols)

    def test_save_load(self, engineered, tmp_path):
        fe, train_fe, val_fe, test_fe = engineered
        cols = fe.feature_columns
        model = XGBoostForecaster(n_estimators=50)
        model.fit(train_fe[cols], train_fe["pm25"])
        preds_before = model.predict(test_fe[cols])

        model_path = tmp_path / "xgb.model"
        model.save(model_path)

        loaded = XGBoostForecaster()
        loaded.load(model_path)
        preds_after = loaded.predict(test_fe[cols])

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)

    def test_not_fitted_raises(self):
        model = XGBoostForecaster()
        with pytest.raises(RuntimeError):
            model.predict(None)  # type: ignore


class TestLightGBMForecaster:
    def test_fit_predict(self, engineered):
        fe, train_fe, val_fe, test_fe = engineered
        cols = fe.feature_columns
        model = LightGBMForecaster(n_estimators=50)
        model.fit(train_fe[cols], train_fe["pm25"], val_fe[cols], val_fe["pm25"])
        preds = model.predict(test_fe[cols])
        assert len(preds) == len(test_fe)
        assert np.isfinite(preds).all()

    def test_metrics_reasonable(self, engineered):
        fe, train_fe, val_fe, test_fe = engineered
        cols = fe.feature_columns
        model = LightGBMForecaster(n_estimators=100)
        model.fit(train_fe[cols], train_fe["pm25"], val_fe[cols], val_fe["pm25"])
        preds = model.predict(test_fe[cols])
        metrics = evaluate(test_fe["pm25"].values, preds)
        assert metrics.r2 > 0.5, f"R² too low: {metrics.r2}"


class TestEvaluationMetrics:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        m = evaluate(y, y)
        assert m.rmse == pytest.approx(0.0, abs=1e-6)
        assert m.r2 == pytest.approx(1.0, abs=1e-6)
        assert m.bias == pytest.approx(0.0, abs=1e-6)

    def test_skill_score_bounds(self):
        y = np.random.default_rng(0).normal(100, 20, 100)
        pred = y + np.random.default_rng(1).normal(0, 10, 100)
        m = evaluate(y, pred)
        assert -2 < m.skill_score <= 1.0