"""Chest X-ray datasets (CheXpert, NIH ChestX-ray14) with uncertainty-label policies.

CheXpert encodes labels as 1 (positive), 0 (negative), -1 (uncertain) and blank (not
mentioned). Following Irvin et al. (2019) we support the policies:

* ``ones``   - map uncertain labels to 1 (U-Ones)
* ``zeros``  - map uncertain labels to 0 (U-Zeros)
* ``ignore`` - mask uncertain labels out of the loss (U-Ignore)

Not-mentioned labels are treated as negative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transforms

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices",
]  # fmt: skip

# The five competition pathologies used for the CheXpert leaderboard.
CHEXPERT_COMPETITION = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]  # fmt: skip


def apply_uncertainty_policy(values: np.ndarray, policy: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(targets, mask)`` where ``mask == 0`` marks labels excluded from the loss."""
    values = np.nan_to_num(values.astype(np.float32), nan=0.0)
    mask = np.ones_like(values)
    uncertain = values == -1
    if policy == "ones":
        values[uncertain] = 1.0
    elif policy == "zeros":
        values[uncertain] = 0.0
    elif policy == "ignore":
        values[uncertain] = 0.0
        mask[uncertain] = 0.0
    else:
        raise ValueError(f"unknown uncertainty policy '{policy}'")
    return values, mask


def load_chexpert_csv(csv_path: str, labels: Sequence[str], frontal_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if frontal_only and "Frontal/Lateral" in df:
        df = df[df["Frontal/Lateral"] == "Frontal"]
    # CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg -> patient00001
    df = df.assign(patient_id=df["Path"].str.extract(r"(patient\d+)")[0])
    return df[["Path", "patient_id", *labels]].reset_index(drop=True)


def load_nih_csv(csv_path: str, labels: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    findings = df["Finding Labels"].str.split("|")
    out = pd.DataFrame({"Path": "images/" + df["Image Index"], "patient_id": df["Patient ID"].astype(str)})
    for lab in labels:
        out[lab] = findings.apply(lambda f, lab=lab: float(lab in f))
    return out


def patient_split(
    df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.0, seed: int = 0
) -> Dict[str, pd.DataFrame]:
    """Split by patient so that no patient appears in more than one subset (avoids leakage)."""
    patients = np.array(sorted(df["patient_id"].astype(str).unique()), dtype=object)
    patients = patients[np.random.default_rng(seed).permutation(len(patients))]
    n_val = int(len(patients) * val_frac)
    n_test = int(len(patients) * test_frac)
    val_p = set(patients[:n_val])
    test_p = set(patients[n_val : n_val + n_test])
    is_val = df["patient_id"].astype(str).isin(val_p)
    is_test = df["patient_id"].astype(str).isin(test_p)
    out = {"train": df[~is_val & ~is_test], "val": df[is_val]}
    if n_test:
        out["test"] = df[is_test]
    return {k: v.reset_index(drop=True) for k, v in out.items()}


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        labels: Sequence[str],
        policy: str = "ones",
        transform: Optional[Callable] = None,
    ) -> None:
        self.paths: List[str] = df["Path"].tolist()
        self.root = Path(image_root)
        self.labels = list(labels)
        self.targets, self.masks = apply_uncertainty_policy(df[self.labels].to_numpy(), policy)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def pos_weight(self) -> torch.Tensor:
        """Negative/positive ratio per class, for class-balanced BCE."""
        pos = (self.targets * self.masks).sum(0)
        neg = ((1 - self.targets) * self.masks).sum(0)
        return torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = Image.open(self.root / self.paths[idx]).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": torch.from_numpy(self.targets[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
            "path": self.paths[idx],
        }


def build_datasets(cfg: Dict) -> Dict[str, ChestXrayDataset]:
    d = cfg["data"]
    labels = d.get("labels") or (CHEXPERT_LABELS if d["name"] == "chexpert" else NIH_LABELS)
    if d["name"] == "chexpert":
        train_df = load_chexpert_csv(d["train_csv"], labels, d.get("frontal_only", True))
        splits = patient_split(train_df, d.get("val_frac", 0.05), 0.0, cfg.get("seed", 0))
        # The official CheXpert validation set (radiologist-labelled) is used as the test set.
        splits["test"] = load_chexpert_csv(d["valid_csv"], labels, d.get("frontal_only", True))
    elif d["name"] == "nih":
        df = load_nih_csv(d["csv"], labels)
        splits = patient_split(df, d.get("val_frac", 0.1), d.get("test_frac", 0.2), cfg.get("seed", 0))
    else:
        raise ValueError(f"unknown dataset {d['name']}")
    policy = d.get("uncertainty_policy", "ones")
    out = {}
    for name, df in splits.items():
        tf = build_transforms(d, train=name == "train")
        # Evaluation sets never use U-Ignore masking: every label counts.
        out[name] = ChestXrayDataset(df, d["image_root"], labels, policy if name == "train" else "zeros", tf)
    return out


def build_dataloaders(cfg: Dict) -> Dict[str, DataLoader]:
    t = cfg["train"]
    datasets = build_datasets(cfg)
    return {
        name: DataLoader(
            ds,
            batch_size=t["batch_size"] if name == "train" else t.get("eval_batch_size", t["batch_size"]),
            shuffle=name == "train",
            num_workers=t.get("num_workers", 8),
            pin_memory=torch.cuda.is_available(),
            drop_last=name == "train",
        )
        for name, ds in datasets.items()
    }
