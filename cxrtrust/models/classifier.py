"""Multi-label classifier on top of torchvision backbones."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torchvision

# backbone name -> (feature attribute, classifier attribute, feature dim attr)
_SUPPORTED = {"densenet121", "resnet50", "efficientnet_b0", "convnext_tiny"}


def _strip_head(name: str, model: nn.Module) -> tuple[nn.Module, int]:
    if name.startswith("densenet"):
        dim = model.classifier.in_features
        return nn.Sequential(model.features, nn.ReLU(inplace=True)), dim
    if name.startswith("resnet"):
        dim = model.fc.in_features
        return nn.Sequential(*list(model.children())[:-2]), dim
    if name.startswith("efficientnet"):
        dim = model.classifier[-1].in_features
        return model.features, dim
    if name.startswith("convnext"):
        dim = model.classifier[-1].in_features
        return nn.Sequential(model.features, nn.GroupNorm(1, dim)), dim
    raise ValueError(name)


class CXRClassifier(nn.Module):
    """Backbone -> feature map -> global pooling -> dropout -> linear (one logit per finding).

    The last convolutional feature map is exposed as ``self.target_layer`` for Grad-CAM.
    """

    def __init__(
        self,
        backbone: str = "densenet121",
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.0,
        pooling: str = "avg",
    ) -> None:
        super().__init__()
        if backbone not in _SUPPORTED:
            raise ValueError(f"backbone must be one of {sorted(_SUPPORTED)}")
        weights = "DEFAULT" if pretrained else None
        base = torchvision.models.get_model(backbone, weights=weights)
        self.features, dim = _strip_head(backbone, base)
        self.target_layer = self.features
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError("pooling must be 'avg' or 'max'")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)
        self.num_classes = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.pool(self.forward_features(x)).flatten(1)
        return self.head(self.dropout(f))


def build_model(cfg: Dict, num_classes: int) -> CXRClassifier:
    m = cfg.get("model", {})
    return CXRClassifier(
        backbone=m.get("backbone", "densenet121"),
        num_classes=num_classes,
        pretrained=m.get("pretrained", True),
        dropout=m.get("dropout", 0.0),
        pooling=m.get("pooling", "avg"),
    )
