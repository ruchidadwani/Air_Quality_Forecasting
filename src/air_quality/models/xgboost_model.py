"""XGBoost gradient-boosted tree forecaster."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost regressor wrapper with early-stopping and feature importance."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        target_col: str = "pm25",
        random_state: int = 42,
    ) -> None:
        super().__init__(target_col=target_col, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self._model: Optional[xgb.XGBRegressor] = None
        self.feature_names_: List[str] = []

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostForecaster":
        self.feature_names_ = list(X_train.columns)
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        early_stopping = self.early_stopping_rounds if eval_set else None

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            early_stopping_rounds=early_stopping,
            random_state=self.random_state,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self._is_fitted = True
        logger.info(
            "XGBoost fitted | best_iteration=%s",
            getattr(self._model, "best_iteration", self.n_estimators),
        )
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
        self._model.save_model(str(path))  # type: ignore[union-attr]
        logger.info("XGBoost model saved → %s", path)

    def load(self, path: Path) -> "XGBoostForecaster":
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(path))
        self._is_fitted = True
        return self

    def get_params(self) -> Dict:
        return {
            "model": "xgboost",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }