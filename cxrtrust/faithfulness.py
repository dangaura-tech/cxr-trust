"""Quantitative checks of saliency maps: are they *faithful* to the model, and *plausible*?

* **Deletion / insertion** (Petsiuk et al., RISE, BMVC 2018): pixels are removed from the image
  (or inserted into a blurred copy) in order of decreasing saliency, and the class probability
  is tracked. A faithful map makes the probability collapse quickly under deletion (low AUC)
  and recover quickly under insertion (high AUC). No annotations are needed.
* **Pointing game** (Zhang et al., IJCV 2018) and its energy-based variant (Wang et al.,
  Score-CAM 2020) compare the map with expert bounding boxes, such as the NIH ChestX-ray14
  ``BBox_List_2017.csv`` subset, measuring localisation plausibility.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

Box = Tuple[float, float, float, float]  # x, y, w, h in pixels of the saliency map
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2 only has trapz


def _baseline(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "blur":
        k = max(3, (x.shape[-1] // 8) | 1)
        return TF.gaussian_blur(x, [k, k], sigma=x.shape[-1] / 16)
    if kind == "mean":
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x)
    if kind == "zero":
        return torch.zeros_like(x)
    raise ValueError("baseline must be 'blur', 'mean' or 'zero'")


@torch.no_grad()
def deletion_insertion(
    model: nn.Module,
    x: torch.Tensor,
    saliency: torch.Tensor,
    class_idx: int,
    steps: int = 32,
    baseline: str = "blur",
    batch_size: int = 33,
) -> Dict[str, np.ndarray | float]:
    """Deletion and insertion curves for one image ``x`` (1, C, H, W) and map (H, W)."""
    model.eval()
    h, w = x.shape[-2:]
    sal = saliency.reshape(-1).to(x.device)
    if sal.numel() != h * w:
        sal = F.interpolate(saliency[None, None].float(), size=(h, w), mode="bilinear", align_corners=False).reshape(-1)
    order = torch.argsort(sal, descending=True)
    base = _baseline(x, baseline)
    fracs = np.linspace(0, 1, steps + 1)
    masks = torch.zeros(steps + 1, 1, h * w, device=x.device)
    for i, f in enumerate(fracs):
        masks[i, 0, order[: int(round(f * h * w))]] = 1.0
    masks = masks.view(steps + 1, 1, h, w)

    def run(images: torch.Tensor) -> np.ndarray:
        chunks = [images[s : s + batch_size] for s in range(0, len(images), batch_size)]
        out = [torch.sigmoid(model(c)[:, class_idx]) for c in chunks]
        return torch.cat(out).float().cpu().numpy()

    deletion = run(x * (1 - masks) + base * masks)
    insertion = run(base * (1 - masks) + x * masks)
    return {
        "fractions": fracs,
        "deletion": deletion,
        "insertion": insertion,
        "deletion_auc": float(_trapezoid(deletion, fracs)),
        "insertion_auc": float(_trapezoid(insertion, fracs)),
    }


def pointing_game(saliency: np.ndarray, boxes: Sequence[Box], tolerance: int = 15) -> bool:
    """Hit if the saliency maximum falls inside any box enlarged by ``tolerance`` pixels."""
    sal = np.asarray(saliency)
    y, x = np.unravel_index(int(np.argmax(sal)), sal.shape)
    return any(bx - tolerance <= x <= bx + bw + tolerance and by - tolerance <= y <= by + bh + tolerance
               for bx, by, bw, bh in boxes)  # fmt: skip


def energy_in_boxes(saliency: np.ndarray, boxes: Sequence[Box]) -> float:
    """Fraction of (non-negative) saliency mass inside the union of boxes."""
    sal = np.clip(np.asarray(saliency, float), 0, None)
    inside = np.zeros(sal.shape, bool)
    for bx, by, bw, bh in boxes:
        inside[max(int(by), 0) : int(np.ceil(by + bh)), max(int(bx), 0) : int(np.ceil(bx + bw))] = True
    total = sal.sum()
    return float(sal[inside].sum() / total) if total > 0 else 0.0


def load_nih_boxes(csv_path: str, image_size: int | None = None, original_size: int = 1024) -> Dict[str, List]:
    """Read ``BBox_List_2017.csv`` into ``{image name: [(label, (x, y, w, h)), ...]}``.

    Boxes are rescaled from the original 1024 x 1024 resolution to ``image_size`` if given.
    """
    df = pd.read_csv(csv_path)
    # header is "Image Index,Finding Label,Bbox [x,y,w,h],,,", so the box occupies columns 2-5
    xs, ys, ws, hs = (df.iloc[:, k].to_numpy(float) for k in (2, 3, 4, 5))
    s = (image_size / original_size) if image_size else 1.0
    out: Dict[str, List] = {}
    for name, label, x, y, w, h in zip(df["Image Index"], df["Finding Label"], xs, ys, ws, hs, strict=True):
        out.setdefault(name, []).append((label, (x * s, y * s, w * s, h * s)))
    return out
