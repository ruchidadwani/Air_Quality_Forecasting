"""Regression metrics for PM2.5 forecasting evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    mape: float
    r2: float
    bias: float         # mean error (positive = over-forecast)
    skill_score: float  # vs. persistence baseline

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"RMSE={self.rmse:.2f}  MAE={self.mae:.2f}  MAPE={self.mape:.1f}%  "
            f"R²={self.r2:.3f}  Bias={self.bias:+.2f}  Skill={self.skill_score:.3f}"
        )


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    baseline_pred: np.ndarray | pd.Series | None = None,
) -> RegressionMetrics:
    """Compute a comprehensive set of regression metrics.

    Parameters
    ----------
    y_true : ground-truth PM2.5 values
    y_pred : model predictions
    baseline_pred : persistence/naive baseline predictions; if *None* the
        persistence model (y_pred = y_true shifted by 1) is used to compute
        the skill score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Mask NaN predictions (LSTM warm-up rows)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    # Skill score vs. persistence
    if baseline_pred is not None:
        baseline_rmse = float(
            np.sqrt(mean_squared_error(y_true, np.asarray(baseline_pred)[mask]))
        )
    else:
        # Persistence: predict today = yesterday
        baseline_rmse = float(np.sqrt(mean_squared_error(y_true[1:], y_true[:-1])))

    skill_score = 1.0 - (rmse / baseline_rmse) if baseline_rmse > 0 else 0.0

    return RegressionMetrics(
        rmse=rmse, mae=mae, mape=mape, r2=r2, bias=bias, skill_score=skill_score
    )