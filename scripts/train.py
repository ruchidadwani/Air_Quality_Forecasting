"""CLI training script.

Usage
-----
    python scripts/train.py --model xgboost
    python scripts/train.py --model lightgbm --tune --trials 50
    python scripts/train.py --model all
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging

import typer
from rich.console import Console
from rich.table import Table

from air_quality.data import generate_synthetic_data
from air_quality.models import LightGBMForecaster, XGBoostForecaster
from air_quality.training import HyperparameterTuner, Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
console = Console()
app = typer.Typer(help="Train PM2.5 forecasting models.")


@app.command()
def train(
    model: str = typer.Option("xgboost", help="Model to train: xgboost | lightgbm | all"),
    tune: bool = typer.Option(False, "--tune", help="Run Optuna hyper-parameter search"),
    trials: int = typer.Option(30, help="Number of Optuna trials"),
    n_days: int = typer.Option(1095, help="Days of synthetic data to generate"),
) -> None:
    """Train one or all models and log results to MLflow."""
    df = generate_synthetic_data(n_days=n_days)
    console.print(f"[cyan]Dataset:[/cyan] {len(df)} days of synthetic Delhi PM2.5 data")

    models_to_train = []

    if model in ("xgboost", "all"):
        if tune:
            console.print("[yellow]Running Optuna tuning for XGBoost…[/yellow]")
            tuner = HyperparameterTuner("xgboost", n_trials=trials)
            trainer_tmp = Trainer(XGBoostForecaster())
            train_fe, val_fe, _ = trainer_tmp.load_data_and_engineer(df)
            best = tuner.tune(train_fe, val_fe, trainer_tmp._fe.feature_columns)
            console.print(f"[green]Best XGBoost params:[/green] {best}")
            models_to_train.append(("XGBoost", XGBoostForecaster(**best)))
        else:
            models_to_train.append(("XGBoost", XGBoostForecaster()))

    if model in ("lightgbm", "all"):
        if tune:
            console.print("[yellow]Running Optuna tuning for LightGBM…[/yellow]")
            tuner = HyperparameterTuner("lightgbm", n_trials=trials)
            trainer_tmp = Trainer(LightGBMForecaster())
            train_fe, val_fe, _ = trainer_tmp.load_data_and_engineer(df)
            best = tuner.tune(train_fe, val_fe, trainer_tmp._fe.feature_columns)
            console.print(f"[green]Best LightGBM params:[/green] {best}")
            models_to_train.append(("LightGBM", LightGBMForecaster(**best)))
        else:
            models_to_train.append(("LightGBM", LightGBMForecaster()))

    table = Table(title="Training Results", show_header=True, header_style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("RMSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Skill", justify="right")

    for name, m in models_to_train:
        console.print(f"[bold cyan]Training {name}…[/bold cyan]")
        trainer = Trainer(m, run_name=name)
        metrics = trainer.run(df)
        table.add_row(
            name,
            f"{metrics['rmse']:.2f}",
            f"{metrics['mae']:.2f}",
            f"{metrics['r2']:.3f}",
            f"{metrics['skill_score']:.3f}",
        )

    console.print(table)
    console.print("[green]✓ Training complete. Results logged to MLflow.[/green]")


if __name__ == "__main__":
    app()