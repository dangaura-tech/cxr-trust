"""Selective prediction: how much does error drop if the model may abstain on uncertain cases?

A *confidence score* ranks images; keeping the most confident fraction ``c`` (coverage) of the
test set gives a selective risk ``r(c)``. The risk-coverage curve and its area (AURC, Geifman
and El-Yaniv 2017) summarise how useful the score is for triage; E-AURC subtracts the area of
an oracle that ranks by the true loss, isolating the quality of the ranking itself.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def image_confidence(probs: np.ndarray, method: str = "entropy") -> np.ndarray:
    """Per-image confidence from per-finding probabilities (higher = more certain).

    ``entropy``: one minus the mean binary entropy over findings; ``min_margin``: the least
    confident finding's distance to 0.5, scaled to [0, 1].
    """
    probs = np.asarray(probs, float)
    if method == "entropy":
        return 1.0 - binary_entropy(probs).mean(1)
    if method == "min_margin":
        return 2 * np.abs(probs - 0.5).min(1)
    raise ValueError("method must be 'entropy' or 'min_margin'")


def hamming_loss(probs: np.ndarray, targets: np.ndarray, thresholds=0.5) -> np.ndarray:
    """Per-image fraction of findings predicted wrongly at ``thresholds`` (scalar or per class)."""
    pred = np.asarray(probs) >= np.asarray(thresholds)
    return (pred != (np.asarray(targets) > 0.5)).mean(1)


def risk_coverage(confidence: np.ndarray, loss: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(coverage, selective risk) after keeping the ``1..N`` most confident samples."""
    order = np.argsort(-np.asarray(confidence), kind="stable")
    loss = np.asarray(loss, float)[order]
    n = np.arange(1, len(loss) + 1)
    return n / len(loss), np.cumsum(loss) / n


def aurc(confidence: np.ndarray, loss: np.ndarray) -> float:
    return float(risk_coverage(confidence, loss)[1].mean())


def selective_report(confidence: np.ndarray, loss: np.ndarray, coverages=(1.0, 0.9, 0.8, 0.5)) -> Dict[str, float]:
    cov, risk = risk_coverage(confidence, loss)
    oracle = aurc(-np.asarray(loss, float), loss)
    out = {"aurc": aurc(confidence, loss), "e_aurc": aurc(confidence, loss) - oracle}
    for c in coverages:
        out[f"risk@{int(round(100 * c))}"] = float(risk[max(int(np.ceil(c * len(risk))) - 1, 0)])
    return out
