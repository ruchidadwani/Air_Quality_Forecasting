"""Bidirectional LSTM forecaster (PyTorch).

Architecture
------------
Input  → BiLSTM (2 layers, 128 hidden, dropout 0.2)
       → Linear projection → PM2.5 prediction

The model consumes a fixed-length look-back window (default 30 days) and
predicts the next day's PM2.5 value.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base import BaseForecaster

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for supervised sequence modelling."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 30,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return x_seq, y_val


# ---------------------------------------------------------------------------
# PyTorch Module
# ---------------------------------------------------------------------------

class _BiLSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, T, 2*H)
        last = out[:, -1, :]           # take last time-step
        last = self.norm(last)
        return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# Forecaster wrapper
# ---------------------------------------------------------------------------

class LSTMForecaster(BaseForecaster):
    """Bidirectional LSTM with Adam + cosine-annealing LR schedule."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 30,
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        target_col: str = "pm25",
        random_state: int = 42,
    ) -> None:
        super().__init__(target_col=target_col, random_state=random_state)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._net: Optional[_BiLSTMNet] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self.train_losses_: list = []

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LSTMForecaster":
        torch.manual_seed(self.random_state)

        # Z-score normalisation (fit on train, apply to val)
        X_np = X_train.values.astype(np.float32)
        self._feature_mean = X_np.mean(axis=0)
        self._feature_std = X_np.std(axis=0) + 1e-8
        X_norm = (X_np - self._feature_mean) / self._feature_std

        y_np = y_train.values.astype(np.float32)

        train_ds = TimeSeriesDataset(X_norm, y_np, self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = (X_val.values.astype(np.float32) - self._feature_mean) / self._feature_std
            val_ds = TimeSeriesDataset(X_val_norm, y_val.values.astype(np.float32), self.seq_len)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self._net = _BiLSTMNet(
            input_size=X_norm.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(DEVICE)

        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs
        )
        criterion = nn.HuberLoss(delta=10.0)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimiser.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item() * len(xb)

            scheduler.step()
            avg_loss = epoch_loss / max(len(train_ds), 1)
            self.train_losses_.append(avg_loss)

            if val_loader and epoch % 5 == 0:
                val_loss = self._eval_loss(val_loader, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in self._net.state_dict().items()}
                logger.debug("Epoch %d/%d | train=%.3f val=%.3f",
                             epoch, self.epochs, avg_loss, val_loss)

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        logger.info("LSTM training complete | device=%s", DEVICE)
        return self

    def _eval_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        assert self._net is not None
        self._net.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += criterion(self._net(xb), yb).item() * len(xb)
                n += len(xb)
        return total / max(n, 1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        assert self._net is not None
        assert self._feature_mean is not None and self._feature_std is not None

        X_norm = (X.values.astype(np.float32) - self._feature_mean) / self._feature_std
        preds = []

        self._net.eval()
        with torch.no_grad():
            for i in range(self.seq_len, len(X_norm)):
                seq = torch.tensor(
                    X_norm[i - self.seq_len : i], dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)
                preds.append(self._net(seq).item())

        # Pad the warm-up period with NaN to keep length consistent
        padding = [np.nan] * self.seq_len
        return np.array(padding + preds)

    def save(self, path: Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),  # type: ignore[union-attr]
                "feature_mean": self._feature_mean,
                "feature_std": self._feature_std,
                "config": self.get_params(),
            },
            str(path),
        )
        logger.info("LSTM model saved → %s", path)

    def load(self, path: Path) -> "LSTMForecaster":
        ckpt = torch.load(str(path), map_location=DEVICE)
        cfg = ckpt["config"]
        self._net = _BiLSTMNet(
            input_size=ckpt["feature_mean"].shape[0],
            hidden_size=cfg.get("hidden_size", self.hidden_size),
            num_layers=cfg.get("num_layers", self.num_layers),
            dropout=cfg.get("dropout", self.dropout),
        ).to(DEVICE)
        self._net.load_state_dict(ckpt["state_dict"])
        self._feature_mean = ckpt["feature_mean"]
        self._feature_std = ckpt["feature_std"]
        self._is_fitted = True
        return self

    def get_params(self) -> Dict:
        return {
            "model": "lstm_bidirectional",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }