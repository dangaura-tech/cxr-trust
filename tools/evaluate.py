#!/usr/bin/env python
"""Evaluate a checkpoint: AUROC / AUPRC per finding, ECE and Brier score before and after
temperature scaling, label-conditional conformal coverage, false-negative-rate control and
risk-coverage (selective prediction). Temperatures and conformal thresholds are fitted on two
disjoint halves of the validation split.

Example:
    python tools/evaluate.py --checkpoint runs/chexpert_densenet121/best.pth --alpha 0.1 \
        --out results/chexpert.json --save-logits results/chexpert_logits.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxrtrust.data import build_dataloaders  # noqa: E402
from cxrtrust.engine import predict  # noqa: E402
from cxrtrust.models import build_model  # noqa: E402
from cxrtrust.report import print_report, trust_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    p.add_argument("--alpha", type=float, default=0.1, help="risk level for conformal sets and FNR control")
    p.add_argument("--save-logits", default=None, help="store val/test logits and targets (.npz) for later analysis")
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
    report = trust_report(val_logits, val_targets, logits, targets, labels, alpha=args.alpha, seed=cfg.get("seed", 0))
    print_report(report)
    if args.save_logits:
        Path(args.save_logits).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_logits, val_logits=val_logits, val_targets=val_targets, logits=logits,
                            targets=targets, labels=np.array(labels))  # fmt: skip
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=lambda o: None))


if __name__ == "__main__":
    main()
