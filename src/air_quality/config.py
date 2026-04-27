"""Centralised configuration using Pydantic Settings v2."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── paths ────────────────────────────────────────────────────────────────
    root_dir: Path = ROOT_DIR
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"
    models_dir: Path = ROOT_DIR / "models"
    mlflow_tracking_uri: str = Field(
        default=f"file://{ROOT_DIR / 'mlruns'}", alias="MLFLOW_TRACKING_URI"
    )

    # ── OpenAQ API ───────────────────────────────────────────────────────────
    openaq_api_key: Optional[str] = Field(default=None, alias="OPENAQ_API_KEY")
    openaq_base_url: str = "https://api.openaq.org/v3"

    # ── model defaults ───────────────────────────────────────────────────────
    target_column: str = "pm25"
    random_state: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure output directories exist
        for d in [self.data_raw_dir, self.data_processed_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()