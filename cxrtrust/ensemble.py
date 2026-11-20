"""Deep ensembles and anatomy-safe test-time augmentation.

Averaging several networks is among the most reliable ways to improve both accuracy and
calibration (Lakshminarayanan et al., NeurIPS 2017), and the disagreement between members
is a useful *epistemic* uncertainty signal that a single network lacks.

* :func:`combine_logits` averages members in logit space (a geometric pooling of odds, which
  keeps the result a valid logit for temperature scaling) or in probability space.
* :func:`epistemic_uncertainty` is the per-finding mutual information
  ``H(mean p) - mean H(p_m)``: zero when members agree, large when they split.
* :class:`TTA` evaluates a model under small deterministic scale and shift changes.
  Horizontal flips are deliberately excluded: they move the heart to the wrong side of the
  image and would teach the ensemble anatomy that does not exist.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def combine_logits(members: Sequence[np.ndarray], mode: str = "logit") -> np.ndarray:
    """Pool member logits ``(M, N, K)``; ``mode='prob'`` averages probabilities then maps back to logits."""
    stack = np.stack([np.asarray(m, float) for m in members])
    if mode == "logit":
        return stack.mean(0)
    if mode == "prob":
        p = np.clip(_sigmoid(stack).mean(0), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    raise ValueError("mode must be 'logit' or 'prob'")


def epistemic_uncertainty(members: Sequence[np.ndarray]) -> np.ndarray:
    """Mutual information between prediction and ensemble member, per (sample, finding)."""
    p = _sigmoid(np.stack([np.asarray(m, float) for m in members]))
    return np.maximum(_entropy(p.mean(0)) - _entropy(p).mean(0), 0.0)


class TTA:
    """Deterministic scale / shift test-time augmentation for ``(B, C, H, W)`` batches.

    Each view zooms by ``scale`` and translates by ``(dx, dy)`` (fractions of the image size)
    with an affine resampling; the identity view is always included.
    """

    DEFAULT_VIEWS: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (1.05, 0.0, 0.0),
        (0.95, 0.0, 0.0),
        (1.0, 0.03, 0.0),
        (1.0, -0.03, 0.0),
        (1.0, 0.0, 0.03),
        (1.0, 0.0, -0.03),
    )

    def __init__(self, views: Optional[Iterable[Tuple[float, float, float]]] = None) -> None:
        self.views = tuple(views) if views is not None else self.DEFAULT_VIEWS
        if (1.0, 0.0, 0.0) not in self.views:
            self.views = ((1.0, 0.0, 0.0), *self.views)

    @staticmethod
    def transform(x: torch.Tensor, scale: float, dx: float, dy: float) -> torch.Tensor:
        if scale == 1.0 and dx == 0.0 and dy == 0.0:
            return x
        theta = x.new_tensor([[1.0 / scale, 0.0, -2 * dx], [0.0, 1.0 / scale, -2 * dy]]).expand(len(x), 2, 3)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    @torch.no_grad()
    def logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Mean logits over all views, shape ``(B, K)``."""
        return torch.stack([model(self.transform(x, *v)) for v in self.views]).float().mean(0)


@torch.no_grad()
def predict_members(
    models: Sequence[nn.Module],
    loader: DataLoader,
    device: torch.device,
    tta: Optional[TTA] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Logits of every member, ``(M, N, K)``, plus targets ``(N, K)``; one pass over the data."""
    for m in models:
        m.eval()
    outs: List[List[torch.Tensor]] = [[] for _ in models]
    targets = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        for i, m in enumerate(models):
            outs[i].append((tta.logits(m, x) if tta else m(x)).float().cpu())
        targets.append(batch["target"])
    return np.stack([torch.cat(o).numpy() for o in outs]), torch.cat(targets).numpy()


def summarize_members(members: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Mean single-member loss versus ensemble loss (binary cross-entropy) and mean disagreement."""

    def bce(logits: np.ndarray) -> float:
        p = np.clip(_sigmoid(logits), 1e-7, 1 - 1e-7)
        return float(-(targets * np.log(p) + (1 - targets) * np.log(1 - p)).mean())

    return {
        "bce_single_mean": float(np.mean([bce(m) for m in members])),
        "bce_ensemble": bce(combine_logits(members)),
        "epistemic_mean": float(epistemic_uncertainty(members).mean()),
    }
