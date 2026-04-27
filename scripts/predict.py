"""CLI prediction script.

Usage
-----
    python scripts/predict.py --days 14
    python scripts/predict.py --model lightgbm --days 30
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from air_quality.config import settings
from air_quality.data import generate_synthetic_data, Preprocessor
from air_quality.evaluation import evaluate
from air_quality.features import FeatureEngineer
from air_quality.models import LightGBMForecaster, XGBoostForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
console = Console()
app = typer.Typer(help="Run PM2.5 inference from a saved model.")


@app.command()
def predict(
    model: str = typer.Option("xgboost", help="Model to use: xgboost | lightgbm"),
    days: int = typer.Option(14, help="Number of test-period days to show"),
) -> None:
    """Load a saved model and print a forecast table for recent test days."""
    model_map = {
        "xgboost": (XGBoostForecaster, "XGBoostForecaster.model"),
        "lightgbm": (LightGBMForecaster, "LightGBMForecaster.model"),
    }
    if model not in model_map:
        console.print(f"[red]Unknown model: {model}. Choose xgboost or lightgbm.[/red]")
        raise typer.Exit(1)

    ModelClass, fname = model_map[model]
    model_path = settings.models_dir / fname

    df = generate_synthetic_data()
    prep = Preprocessor()
    fe = FeatureEngineer()

    _, _, test_raw = prep.fit_transform(df)
    test_fe = fe.fit_transform(test_raw)  # fit on test for demo; normally use train-fitted fe
    feat_cols = fe.feature_columns

    forecaster = ModelClass()
    if not model_path.exists():
        console.print(f"[yellow]No saved model at {model_path} — training fresh…[/yellow]")
        train_raw, val_raw, _ = prep.fit_transform(df)
        train_fe = fe.fit_transform(train_raw)
        val_fe = fe.transform(val_raw).dropna()
        forecaster.fit(train_fe[feat_cols], train_fe["pm25"],
                       val_fe[feat_cols], val_fe["pm25"])
    else:
        forecaster.load(model_path)

    X_test = test_fe[feat_cols]
    y_test = test_fe["pm25"].values
    y_pred = forecaster.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    console.print(f"\n[bold cyan]{model.upper()} test metrics:[/bold cyan] {metrics}\n")

    table = Table(title=f"Last {days} predictions", show_header=True,
                  header_style="bold magenta")
    table.add_column("Date")
    table.add_column("Actual PM2.5", justify="right")
    table.add_column("Predicted PM2.5", justify="right")
    table.add_column("Error", justify="right")

    mask = ~np.isnan(y_pred)
    dates = (
        test_fe["date"].values if "date" in test_fe.columns
        else pd.date_range("2023-01-01", periods=len(y_test))
    )

    for date, actual, pred in zip(
        dates[mask][-days:], y_test[mask][-days:], y_pred[mask][-days:]
    ):
        error = pred - actual
        colour = "red" if abs(error) > 30 else "green"
        table.add_row(
            str(date)[:10],
            f"{actual:.1f}",
            f"{pred:.1f}",
            f"[{colour}]{error:+.1f}[/{colour}]",
        )

    console.print(table)


if __name__ == "__main__":
    app()