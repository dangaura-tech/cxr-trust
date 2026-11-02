#!/usr/bin/env python
"""Train a multi-label chest X-ray classifier.

Example:
    python tools/train.py --config configs/chexpert_densenet121.yaml
    python tools/train.py --config configs/nih_convnext.yaml train.epochs=20 model.dropout=0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxrtrust.data import build_dataloaders  # noqa: E402
from cxrtrust.engine import Trainer  # noqa: E402
from cxrtrust.losses import build_loss  # noqa: E402
from cxrtrust.models import build_model  # noqa: E402
from cxrtrust.utils import apply_overrides, get_logger, load_config, seed_everything  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    seed_everything(cfg.get("seed", 0))
    work_dir = args.work_dir or f"runs/{Path(args.config).stem}"
    loaders = build_dataloaders(cfg)
    train_ds = loaders["train"].dataset
    labels = train_ds.labels
    logger = get_logger()
    logger.info(f"labels: {len(labels)} | train {len(train_ds)} | val {len(loaders['val'].dataset)}")

    model = build_model(cfg, len(labels))
    criterion = build_loss(cfg, train_ds.pos_weight())
    Trainer(model, criterion, loaders, labels, cfg, work_dir, torch.device(args.device)).fit()


if __name__ == "__main__":
    main()
