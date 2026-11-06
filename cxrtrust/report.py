"""One evaluation protocol for any source of logits (trained CNN or zero-shot VLM).

The validation split is divided at random into two disjoint halves: the first fits the per-class
temperatures, the second calibrates the conformal procedures on the *temperature-scaled*
probabilities. Keeping them disjoint preserves the finite-sample conformal guarantee.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from .calibration import TemperatureScaler, calibration_report, reliability_curve
from .conformal import FNRControl, MondrianConformal
from .metrics import multilabel_report
from .selective import hamming_loss, image_confidence, selective_report


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def trust_report(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    logits: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    alpha: float = 0.1,
    seed: int = 0,
) -> Dict:
    perm = np.random.default_rng(seed).permutation(len(val_logits))
    fit_idx, cal_idx = perm[: len(perm) // 2], perm[len(perm) // 2 :]
    scaler = TemperatureScaler(len(labels)).fit(
        torch.from_numpy(val_logits[fit_idx]).float(), torch.from_numpy(val_targets[fit_idx]).float()
    )
    with torch.no_grad():
        cal_val = sigmoid(scaler(torch.from_numpy(val_logits[cal_idx]).float()).numpy())
        cal_test = sigmoid(scaler(torch.from_numpy(logits).float()).numpy())
    probs = sigmoid(logits)

    report: Dict = multilabel_report(probs, targets, labels)
    report.update({f"uncalibrated/{k}": v for k, v in calibration_report(probs, targets).items()})
    report.update({f"calibrated/{k}": v for k, v in calibration_report(cal_test, targets).items()})
    report["temperatures"] = dict(zip(labels, scaler.temperature.detach().numpy().round(4).tolist(), strict=True))
    conf, acc, count = reliability_curve(cal_test, targets)
    report["reliability"] = {"confidence": conf.tolist(), "accuracy": acc.tolist(), "count": count.tolist()}

    cp = MondrianConformal(alpha).fit(cal_val, val_targets[cal_idx]).report(cal_test, targets)
    report["conformal"] = {
        "alpha": alpha,
        **{f"{k}/{lab}": float(v) for k, vals in cp.items() for lab, v in zip(labels, vals, strict=True)},
        "coverage_pos_min": float(np.nanmin(cp["coverage_pos"])),
        "coverage_neg_min": float(np.nanmin(cp["coverage_neg"])),
        "ambiguity_mean": float(np.mean(cp["ambiguity"])),
    }
    crc = FNRControl(alpha).fit(cal_val, val_targets[cal_idx])
    report["fnr_control"] = {"alpha": alpha, **crc.report(cal_test, targets)}
    loss = hamming_loss(cal_test, targets)
    report["selective"] = {
        method: selective_report(image_confidence(cal_test, method), loss) for method in ("entropy", "min_margin")
    }
    return report


def print_report(report: Dict, indent: str = "") -> None:
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{indent}{k:<44} {v:.4f}")
        elif isinstance(v, dict) and k not in ("temperatures", "reliability") and not k.startswith("conformal"):
            print(f"{indent}{k}")
            print_report(v, indent + "  ")
    if "conformal" in report:
        c = report["conformal"]
        print(f"{indent}conformal (alpha={c['alpha']}): min coverage pos {c['coverage_pos_min']:.4f} / "
              f"neg {c['coverage_neg_min']:.4f}, ambiguity {c['ambiguity_mean']:.4f}")  # fmt: skip
