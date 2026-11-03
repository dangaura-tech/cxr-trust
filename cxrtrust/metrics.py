from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def multilabel_report(probs: np.ndarray, targets: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """Per-class AUROC / AUPRC plus macro averages. Classes with a single target value are skipped."""
    out: Dict[str, float] = {}
    aucs, aps = [], []
    for k, name in enumerate(labels):
        y = targets[:, k]
        if y.min() == y.max():
            continue
        auc = roc_auc_score(y, probs[:, k])
        ap = average_precision_score(y, probs[:, k])
        out[f"auroc/{name}"] = auc
        out[f"auprc/{name}"] = ap
        aucs.append(auc)
        aps.append(ap)
    out["auroc_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    out["auprc_macro"] = float(np.mean(aps)) if aps else float("nan")
    return out


def best_f1_thresholds(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-class threshold maximising F1 on a validation set."""
    thresholds = np.full(probs.shape[1], 0.5)
    grid = np.linspace(0.01, 0.99, 99)
    for k in range(probs.shape[1]):
        y = targets[:, k].astype(bool)
        if not y.any():
            continue
        p = probs[:, k][None, :] >= grid[:, None]
        tp = (p & y).sum(1)
        fp = (p & ~y).sum(1)
        fn = (~p & y).sum(1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        thresholds[k] = grid[int(f1.argmax())]
    return thresholds
