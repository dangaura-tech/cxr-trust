from __future__ import annotations

import copy
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base_", None)
    if base:
        parent = load_config(str(path.parent / base))
        cfg = _merge(parent, cfg)
    return cfg


def _merge(a: Dict, b: Dict) -> Dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg: Dict, overrides: Iterable[str]) -> Dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, raw = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(raw)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("cxrtrust")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        logger.addHandler(h)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def _parse_value(raw: str) -> Any:
    """Parse an override value as YAML, also accepting floats like ``5e-4`` (YAML 1.1 reads them as strings)."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value
