"""LightGBM gradient-boosted tree forecaster."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class LightGBMForecaster(BaseForecaster):
    """LightGBM wrapper — generally faster than XGBoost on large feature sets."""

    def __init__(
        self,
        n_estimators: int = 500,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        target_col: str = "pm25",
        random_state: int = 42,
    ) -> None:
        super().__init__(target_col=target_col, random_state=random_state)
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self._model: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: List[str] = []

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMForecaster":
        self.feature_names_ = list(X_train.columns)
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                     lgb.log_evaluation(period=-1)]

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if eval_set else [lgb.log_evaluation(period=-1)],
        )
        self._is_fitted = True
        logger.info("LightGBM fitted | best_iteration=%s",
                    self._model.best_iteration_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)  # type: ignore[union-attr]

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self._model.feature_importances_,  # type: ignore[union-attr]
            index=self.feature_names_,
            name="importance",
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(path))  # type: ignore[union-attr]
        logger.info("LightGBM model saved → %s", path)

    def load(self, path: Path) -> "LightGBMForecaster":
        booster = lgb.Booster(model_file=str(path))
        self._model = lgb.LGBMRegressor()
        self._model._Booster = booster  # type: ignore[attr-defined]
        self._is_fitted = True
        return self

    def get_params(self) -> Dict:
        return {
            "model": "lightgbm",
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }