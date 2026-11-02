"""Multi-label losses supporting per-label masks (for the U-Ignore policy)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(loss: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class MaskedBCE(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight, reduction="none")
        return _masked_mean(loss, mask)


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification (Ridnik et al., ICCV 2021).

    Down-weights easy negatives with a stronger focusing parameter ``gamma_neg`` and
    discards very easy negatives with probability shifting ``clip``.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        loss_pos = target * torch.log(p.clamp(min=self.eps)) * (1 - p) ** self.gamma_pos
        loss_neg = (1 - target) * torch.log((1 - p_neg).clamp(min=self.eps)) * p_neg**self.gamma_neg
        return _masked_mean(-(loss_pos + loss_neg), mask)


def build_loss(cfg: Dict, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    lcfg = dict(cfg.get("loss", {"name": "bce"}))
    name = lcfg.pop("name", "bce")
    if name == "bce":
        return MaskedBCE(pos_weight if lcfg.get("class_balanced", False) else None)
    if name == "asl":
        lcfg.pop("class_balanced", None)
        return AsymmetricLoss(**lcfg)
    raise ValueError(f"unknown loss {name}")
