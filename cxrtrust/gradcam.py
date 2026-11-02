"""Grad-CAM and Grad-CAM++ saliency maps for any convolutional target layer."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Usage::

    with GradCAM(model, model.target_layer) as cam:
        heatmap = cam(image, class_idx=2)    # (B, H, W) in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, plus_plus: bool = False) -> None:
        self.model = model
        self.plus_plus = plus_plus
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, out: torch.Tensor) -> None:
        self._acts = out.detach()

    def _save_grad(self, _m, _gi, grad_out) -> None:
        self._grads = grad_out[0].detach()

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()
        self.model.train(was_training)
        a, g = self._acts, self._grads
        if self.plus_plus:
            g2, g3 = g.pow(2), g.pow(3)
            denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        else:
            weights = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo = flat.min(1, keepdim=True)[0][..., None]
        hi = flat.max(1, keepdim=True)[0][..., None]
        return (cam - lo) / (hi - lo).clamp_min(1e-8)


def overlay_heatmap(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap onto a grayscale uint8 image using a jet-like colour map."""
    cam = np.clip(cam, 0, 1)
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    heat = np.stack([r, g, b], -1) * 255
    base = np.repeat(gray[..., None], 3, -1).astype(np.float32)
    return ((1 - alpha) * base + alpha * heat).astype(np.uint8)
