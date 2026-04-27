"""End-to-end training pipeline with MLflow experiment tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd

from air_quality.config import settings
from air_quality.data import generate_synthetic_data, Preprocessor
from air_quality.evaluation import evaluate
from air_quality.features import FeatureEngineer
from air_quality.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the full train → evaluate → log → save pipeline.

    Usage
    -----
    >>> trainer = Trainer(model=XGBoostForecaster())
    >>> metrics = trainer.run()
    """

    def __init__(
        self,
        model: BaseForecaster,
        experiment_name: str = "air-quality-forecasting",
        run_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.experiment_name = experiment_name
        self.run_name = run_name or type(model).__name__
        self._fe = FeatureEngineer()
        self._prep = Preprocessor(
            train_ratio=settings.train_ratio,
            val_ratio=settings.val_ratio,
        )

    # ------------------------------------------------------------------

    def run(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Execute full pipeline and return evaluation metrics dict."""
        if df is None:
            logger.info("No DataFrame provided — generating synthetic data.")
            df = generate_synthetic_data()

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name) as run:
            logger.info("MLflow run started: %s", run.info.run_id)

            # 1. Pre-process
            train_raw, val_raw, test_raw = self._prep.fit_transform(df)

            # 2. Feature engineering
            train_fe = self._fe.fit_transform(train_raw)
            val_fe = self._fe.transform(val_raw).dropna()
            test_fe = self._fe.transform(test_raw).dropna()

            feature_cols = self._fe.feature_columns
            target = settings.target_column

            X_train = train_fe[feature_cols]
            y_train = train_fe[target]
            X_val = val_fe[feature_cols]
            y_val = val_fe[target]
            X_test = test_fe[feature_cols]
            y_test = test_fe[target]

            # 3. Train
            logger.info("Training %s…", type(self.model).__name__)
            self.model.fit(X_train, y_train, X_val, y_val)

            # 4. Evaluate on test set
            y_pred = self.model.predict(X_test)
            metrics = evaluate(y_test.values, y_pred)
            logger.info("Test metrics: %s", metrics)

            # 5. Log to MLflow
            mlflow.log_params(self.model.get_params())
            mlflow.log_metrics(metrics.to_dict())
            mlflow.set_tag("city", "Delhi")
            mlflow.set_tag("target", target)
            mlflow.set_tag("n_features", len(feature_cols))

            # 6. Save model artifact
            model_path = settings.models_dir / f"{type(self.model).__name__}.model"
            self.model.save(model_path)
            mlflow.log_artifact(str(model_path))

            logger.info("Run complete. run_id=%s", run.info.run_id)
            return metrics.to_dict()

    # ------------------------------------------------------------------

    def load_data_and_engineer(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Exposed for use by the Optuna tuner."""
        if df is None:
            df = generate_synthetic_data()
        train_raw, val_raw, test_raw = self._prep.fit_transform(df)
        train_fe = self._fe.fit_transform(train_raw)
        val_fe = self._fe.transform(val_raw).dropna()
        test_fe = self._fe.transform(test_raw).dropna()
        return train_fe, val_fe, test_fe