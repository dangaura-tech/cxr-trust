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
from cxrtrust.ensemble import TTA, combine_logits, predict_members, summarize_members  # noqa: E402
from cxrtrust.models import build_model  # noqa: E402
from cxrtrust.report import print_report, trust_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, nargs="+", help="one checkpoint, or several for a deep ensemble")
    p.add_argument("--tta", action="store_true", help="average logits over small scale / shift views (no flips)")
    p.add_argument("--pool", choices=["logit", "prob"], default="logit", help="how ensemble members are averaged")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    p.add_argument("--alpha", type=float, default=0.1, help="risk level for conformal sets and FNR control")
    p.add_argument("--save-logits", default=None, help="store val/test logits and targets (.npz) for later analysis")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    models, cfg, labels = [], None, None
    for path in args.checkpoint:
        ckpt = torch.load(path, map_location="cpu")
        if labels is not None and ckpt["labels"] != labels:
            raise SystemExit(f"{path}: label set differs from the first checkpoint")
        cfg, labels = cfg or ckpt["config"], ckpt["labels"]
        mcfg = {**ckpt["config"], "model": {**ckpt["config"].get("model", {}), "pretrained": False}}
        model = build_model(mcfg, len(labels))
        model.load_state_dict(ckpt["model"])
        models.append(model.to(device).eval())
    loaders = build_dataloaders(cfg)
    tta = TTA() if args.tta else None
    val_members, val_targets = predict_members(models, loaders["val"], device, tta)
    members, targets = predict_members(models, loaders[args.split], device, tta)
    val_logits, logits = combine_logits(val_members, args.pool), combine_logits(members, args.pool)
    report = trust_report(val_logits, val_targets, logits, targets, labels, alpha=args.alpha, seed=cfg.get("seed", 0))
    if len(models) > 1:
        report["ensemble"] = summarize_members(members, targets)
        report["ensemble"]["members"] = len(models)
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
