#!/usr/bin/env python
"""Zero-shot evaluation of a vision-language model (default: BiomedCLIP) on a dataset config.

No training: findings are scored by positive-vs-negative prompt ensembles, and the resulting
logits go through the same protocol as trained models (temperature scaling, conformal sets,
FNR control, selective prediction).

Example:
    pip install -e ".[vlm]"
    python tools/zeroshot.py --config configs/chexpert_densenet121.yaml --out results/zeroshot_biomedclip.json
    python tools/zeroshot.py --config configs/nih_densenet121.yaml --model hf-hub:<org>/<open_clip model>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxrtrust.data.datasets import build_datasets  # noqa: E402
from cxrtrust.report import print_report, trust_report  # noqa: E402
from cxrtrust.utils import apply_overrides, load_config  # noqa: E402
from cxrtrust.zeroshot import OpenCLIPBackend, ZeroShotClassifier  # noqa: E402


def dataset_logits(clf: ZeroShotClassifier, ds, batch_size: int, limit: int | None):
    ds.transform = None  # the VLM applies its own preprocessing to the PIL images
    n = len(ds) if limit is None else min(limit, len(ds))
    logits, targets = [], []
    for s in range(0, n, batch_size):
        items = [ds[i] for i in range(s, min(s + batch_size, n))]
        logits.append(clf.logits([it["image"] for it in items]))
        targets.append(np.stack([it["target"].numpy() for it in items]))
        print(f"\r{min(s + batch_size, n)}/{n}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return np.concatenate(logits).astype(np.float32), np.concatenate(targets).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="dataset config (model / train sections are ignored)")
    p.add_argument("--model", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--split", default="test")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="evaluate at most this many images per split")
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    datasets = build_datasets(cfg)
    labels = datasets[args.split].labels
    clf = ZeroShotClassifier(OpenCLIPBackend(args.model), labels)
    val_logits, val_targets = dataset_logits(clf, datasets["val"], args.batch_size, args.limit)
    logits, targets = dataset_logits(clf, datasets[args.split], args.batch_size, args.limit)
    report = trust_report(val_logits, val_targets, logits, targets, labels, alpha=args.alpha, seed=cfg.get("seed", 0))
    report["model"] = args.model
    print_report(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=lambda o: None))


if __name__ == "__main__":
    main()
