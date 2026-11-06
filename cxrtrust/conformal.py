"""Distribution-free uncertainty for multi-label predictions.

Two complementary guarantees, both fitted on a held-out calibration split and valid for any
model under exchangeability (no retraining):

* :class:`MondrianConformal` builds, for every finding, a *prediction set* over
  ``{absent, present}`` with **label-conditional** coverage: for each class ``k`` and each true
  label ``y``, ``P(y in C_k(x) | y) >= 1 - alpha``. Conditioning on the label matters for rare
  findings, where marginal coverage would be met by always predicting "absent". A set
  containing both labels means "the model cannot decide at this risk level": a natural
  trigger for deferring the case to a radiologist.
* :class:`FNRControl` implements conformal risk control (Angelopoulos et al., 2024) for the
  **false-negative rate** of the set of reported findings: it picks one threshold ``lambda``
  such that ``E[FNR] <= alpha`` on new images when every finding with ``p >= lambda`` is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of calibration scores.

    Returns ``inf`` when there are too few scores for the requested level (the set must then
    contain every label to keep the guarantee).
    """
    scores = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or rank > n:
        return float("inf")
    return float(scores[max(rank, 1) - 1])


@dataclass
class MondrianConformal:
    """Per-class, label-conditional split conformal prediction for binary findings.

    Nonconformity of label 1 is ``1 - p`` and of label 0 is ``p``; one quantile is fitted per
    (class, label) cell.
    """

    alpha: float = 0.1
    q_pos: np.ndarray = field(default=None, repr=False)
    q_neg: np.ndarray = field(default=None, repr=False)

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> "MondrianConformal":
        probs, targets = np.asarray(probs, float), np.asarray(targets) > 0.5
        k = probs.shape[1]
        self.q_pos = np.array([conformal_quantile(1 - probs[targets[:, c], c], self.alpha) for c in range(k)])
        self.q_neg = np.array([conformal_quantile(probs[~targets[:, c], c], self.alpha) for c in range(k)])
        return self

    def predict(self, probs: np.ndarray) -> Dict[str, np.ndarray]:
        """Boolean arrays ``present`` / ``absent`` (label included in the set) and ``ambiguous``."""
        if self.q_pos is None:
            raise RuntimeError("call fit() first")
        probs = np.asarray(probs, float)
        present = 1 - probs <= self.q_pos
        absent = probs <= self.q_neg
        return {"present": present, "absent": absent, "ambiguous": present & absent, "empty": ~present & ~absent}

    def report(self, probs: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-class empirical coverage (overall, on positives, on negatives) and ambiguity rate."""
        sets = self.predict(probs)
        y = np.asarray(targets) > 0.5
        covered = np.where(y, sets["present"], sets["absent"])

        def masked_mean(v, m):
            return np.array([v[m[:, c], c].mean() if m[:, c].any() else np.nan for c in range(v.shape[1])])

        return {
            "coverage": covered.mean(0),
            "coverage_pos": masked_mean(covered, y),
            "coverage_neg": masked_mean(covered, ~y),
            "ambiguity": sets["ambiguous"].mean(0),
        }


def false_negative_rate(probs: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
    """Per-image FNR of reporting ``{k : p_k >= lam}``; images without findings contribute 0."""
    y = np.asarray(targets) > 0.5
    missed = (np.asarray(probs) < lam) & y
    npos = y.sum(1)
    return np.where(npos > 0, missed.sum(1) / np.maximum(npos, 1), 0.0)


@dataclass
class FNRControl:
    """Conformal risk control of the per-image false-negative rate (loss bounded by B = 1).

    ``lambda_hat = sup { lam : (n R_n(lam) + 1) / (n + 1) <= alpha }``, where ``R_n`` is the
    empirical FNR on the calibration set. FNR decreases monotonically as ``lam`` decreases,
    so the guarantee ``E[FNR(lambda_hat)] <= alpha`` holds for a new exchangeable image.
    """

    alpha: float = 0.1
    grid_size: int = 2001
    threshold: float = field(default=None)

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> "FNRControl":
        n = len(probs)
        for lam in np.linspace(1.0, 0.0, self.grid_size):  # largest threshold first
            risk = false_negative_rate(probs, targets, lam).mean()
            if (n * risk + 1.0) / (n + 1.0) <= self.alpha:
                self.threshold = float(lam)
                return self
        self.threshold = 0.0  # report everything
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("call fit() first")
        return np.asarray(probs) >= self.threshold

    def report(self, probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        reported = self.predict(probs)
        return {
            "threshold": self.threshold,
            "fnr": float(false_negative_rate(probs, targets, self.threshold).mean()),
            "mean_findings_reported": float(reported.sum(1).mean()),
        }
