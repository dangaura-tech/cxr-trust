from __future__ import annotations

from typing import Dict

from torchvision import transforms as T


def build_transforms(cfg: Dict, train: bool) -> T.Compose:
    """Grayscale radiograph -> 3-channel normalised tensor (ImageNet statistics)."""
    size = cfg.get("image_size", 320)
    ops = [T.Grayscale(num_output_channels=3)]
    if train:
        ops += [
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        # Horizontal flips are deliberately NOT used: they move the heart to the wrong side.
    else:
        ops += [T.Resize(size), T.CenterCrop(size)]
    ops += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(ops)
