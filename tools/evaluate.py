#!/usr/bin/env python
"""Evaluate a checkpoint: AUROC / AUPRC per finding, ECE and Brier score before and after
temperature scaling (temperatures are fitted on the validation split).

Example:
    python tools/evaluate.py --checkpoint runs/chexpert_densenet121/best.pth --out results/chexpert.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxrtrust.calibration import TemperatureScaler, calibration_report, reliability_curve  # noqa: E402
from cxrtrust.data import build_dataloaders  # noqa: E402
from cxrtrust.engine import predict  # noqa: E402
from cxrtrust.metrics import multilabel_report  # noqa: E402
from cxrtrust.models import build_model  # noqa: E402


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg, labels = ckpt["config"], ckpt["labels"]
    device = torch.device(args.device)
    model = build_model({**cfg, "model": {**cfg.get("model", {}), "pretrained": False}}, len(labels))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    loaders = build_dataloaders(cfg)

    val_logits, val_targets = predict(model, loaders["val"], device)
    logits, targets = predict(model, loaders[args.split], device)
    scaler = TemperatureScaler(len(labels)).fit(torch.from_numpy(val_logits), torch.from_numpy(val_targets))
    with torch.no_grad():
        cal_logits = scaler(torch.from_numpy(logits)).numpy()

    probs, cal_probs = sigmoid(logits), sigmoid(cal_logits)
    report = multilabel_report(probs, targets, labels)
    report.update({f"uncalibrated/{k}": v for k, v in calibration_report(probs, targets).items()})
    report.update({f"calibrated/{k}": v for k, v in calibration_report(cal_probs, targets).items()})
    report["temperatures"] = dict(zip(labels, scaler.temperature.detach().numpy().round(4).tolist(), strict=True))
    conf, acc, count = reliability_curve(cal_probs, targets)
    report["reliability"] = {"confidence": conf.tolist(), "accuracy": acc.tolist(), "count": count.tolist()}

    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k:<40} {v:.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=lambda o: None))


if __name__ == "__main__":
    main()
