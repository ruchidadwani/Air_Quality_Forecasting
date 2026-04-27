"""Abstract base class for all forecasting models."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


class BaseForecaster(abc.ABC):
    """Uniform interface shared by every forecaster in this project."""

    def __init__(self, target_col: str = "pm25", random_state: int = 42) -> None:
        self.target_col = target_col
        self.random_state = random_state
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseForecaster":
        """Train the model."""

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return point predictions."""

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to *path*."""

    @abc.abstractmethod
    def load(self, path: Path) -> "BaseForecaster":
        """Load a persisted model from *path*."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet.")

    def get_params(self) -> Dict:
        """Return hyper-parameters as a flat dict (for MLflow logging)."""
        return {}