#!/usr/bin/env python
"""Generate Grad-CAM / Grad-CAM++ explanations for the top-k predicted findings of an image.

Example:
    python tools/explain.py --checkpoint runs/chexpert_densenet121/best.pth \
        --image path/to/cxr.jpg --topk 3 --plus-plus --output explanations/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxrtrust.data.transforms import build_transforms  # noqa: E402
from cxrtrust.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from cxrtrust.models import build_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--plus-plus", action="store_true")
    p.add_argument("--output", default="explanations")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg, labels = ckpt["config"], ckpt["labels"]
    model = build_model({**cfg, "model": {**cfg.get("model", {}), "pretrained": False}}, len(labels))
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    pil = Image.open(args.image).convert("L")
    x = build_transforms(cfg["data"], train=False)(pil)[None].to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu().numpy()
    order = np.argsort(-probs)[: args.topk]

    size = x.shape[-1]
    gray = np.asarray(pil.resize((size, size)))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with GradCAM(model, model.target_layer, plus_plus=args.plus_plus) as cam:
        for k in order:
            heat = cam(x, int(k))[0].cpu().numpy()
            name = labels[k].replace(" ", "_").replace("/", "-")
            Image.fromarray(overlay_heatmap(gray, heat)).save(out / f"{Path(args.image).stem}_{name}.png")
            print(f"{labels[k]:<28} p={probs[k]:.3f}")


if __name__ == "__main__":
    main()
