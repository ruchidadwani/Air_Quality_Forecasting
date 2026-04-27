"""Optuna-based hyper-parameter optimisation for XGBoost and LightGBM."""

from __future__ import annotations

import logging
from typing import Dict, Literal

import optuna
import pandas as pd

from air_quality.config import settings
from air_quality.evaluation import evaluate
from air_quality.models.lgbm_model import LightGBMForecaster
from air_quality.models.xgboost_model import XGBoostForecaster

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

ModelType = Literal["xgboost", "lightgbm"]


class HyperparameterTuner:
    """Run Optuna TPE search to find optimal hyper-parameters.

    Example
    -------
    >>> tuner = HyperparameterTuner("xgboost", n_trials=50)
    >>> best = tuner.tune(train_df, val_df, feature_cols)
    >>> print(best)
    """

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        n_trials: int = 50,
        target_metric: str = "rmse",
    ) -> None:
        self.model_type = model_type
        self.n_trials = n_trials
        self.target_metric = target_metric
        self.best_params_: Dict = {}
        self.study_: optuna.Study | None = None

    def tune(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "pm25",
    ) -> Dict:
        """Return best hyper-parameter dict after Optuna search."""
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        def objective(trial: optuna.Trial) -> float:
            model = self._build_model(trial)
            model.fit(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_val)
            metrics = evaluate(y_val.values, y_pred)
            return getattr(metrics, self.target_metric)

        self.study_ = optuna.create_study(
            direction="minimize",
            study_name=f"{self.model_type}_tuning",
            sampler=optuna.samplers.TPESampler(seed=settings.random_state),
        )
        self.study_.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params_ = self.study_.best_params
        logger.info(
            "Optuna %s tuning complete | best %s=%.4f",
            self.model_type,
            self.target_metric,
            self.study_.best_value,
        )
        return self.best_params_

    # ------------------------------------------------------------------

    def _build_model(self, trial: optuna.Trial) -> XGBoostForecaster | LightGBMForecaster:
        if self.model_type == "xgboost":
            return XGBoostForecaster(
                n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
                max_depth=trial.suggest_int("max_depth", 3, 9),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                early_stopping_rounds=30,
            )
        else:  # lightgbm
            return LightGBMForecaster(
                n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
                num_leaves=trial.suggest_int("num_leaves", 20, 150),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                early_stopping_rounds=30,
            )