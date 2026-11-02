"""Post-hoc calibration (per-class temperature scaling) and calibration metrics."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """One learnable temperature per class, fitted on held-out logits by minimising NLL.

    Temperature scaling preserves the ranking of predictions, so AUROC is unchanged while
    the predicted probabilities become better calibrated.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(num_classes))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_t.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 200) -> "TemperatureScaler":
        logits = logits.detach().float()
        targets = targets.detach().float()
        opt = torch.optim.LBFGS([self.log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
        bce = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = bce(self(logits), targets)
            loss.backward()
            return loss

        opt.step(closure)
        return self


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Binary ECE with equal-width bins, computed over all (sample, class) pairs."""
    probs = np.asarray(probs).ravel()
    targets = np.asarray(targets).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(probs[sel].mean() - targets[sel].mean())
    return float(ece)


def reliability_curve(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin confidence, bin accuracy, bin count) for a reliability diagram."""
    probs = np.asarray(probs).ravel()
    targets = np.asarray(targets).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = idx == b
        count[b] = sel.sum()
        if count[b]:
            conf[b] = probs[sel].mean()
            acc[b] = targets[sel].mean()
    return conf, acc, count


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(targets)) ** 2))


def calibration_report(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {"ece": expected_calibration_error(probs, targets), "brier": brier_score(probs, targets)}
