#!/usr/bin/env python
"""Faithfulness and localisation of Grad-CAM / Grad-CAM++ explanations over a dataset split.

For each image the top-k predicted findings are explained; deletion / insertion AUCs are
compared against a random-saliency control. With ``--bbox-csv`` (NIH ``BBox_List_2017.csv``)
the pointing game and the energy inside expert boxes are reported for annotated findings.

Example:
    python tools/faithfulness.py --checkpoint runs/nih_densenet121/best.pth --split test \\
        --limit 200 --bbox-csv data/nih/BBox_List_2017.csv --out results/faithfulness.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxrtrust.data.datasets import build_datasets  # noqa: E402
from cxrtrust.faithfulness import deletion_insertion, energy_in_boxes, load_nih_boxes, pointing_game  # noqa: E402
from cxrtrust.gradcam import GradCAM  # noqa: E402
from cxrtrust.models import build_model  # noqa: E402

# NIH box labels differ slightly from the classification label names
BOX_ALIASES = {"Infiltrate": "Infiltration"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--baseline", choices=["blur", "mean", "zero"], default="blur")
    p.add_argument("--bbox-csv", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg, labels = ckpt["config"], ckpt["labels"]
    device = torch.device(args.device)
    model = build_model({**cfg, "model": {**cfg.get("model", {}), "pretrained": False}}, len(labels))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    ds = build_datasets(cfg)[args.split]
    size = cfg["data"]["image_size"]
    boxes = load_nih_boxes(args.bbox_csv, image_size=size) if args.bbox_csv else {}
    if boxes:  # explain the annotated images and findings
        indices = [i for i, path in enumerate(ds.paths) if Path(path).name in boxes][: args.limit]
    else:
        indices = list(range(min(args.limit, len(ds))))

    results = defaultdict(list)
    rng = torch.Generator().manual_seed(0)
    cams = {"gradcam": GradCAM(model, model.target_layer), "gradcam++": GradCAM(model, model.target_layer, True)}
    for n, i in enumerate(indices):
        item = ds[i]
        x = item["image"][None].to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].cpu().numpy()
        name = Path(item["path"]).name
        annotated = {BOX_ALIASES.get(lab, lab): [] for lab, _ in boxes.get(name, [])}
        for lab, box in boxes.get(name, []):
            annotated[BOX_ALIASES.get(lab, lab)].append(box)
        targets = [labels.index(lab) for lab in annotated if lab in labels] or list(np.argsort(-probs)[: args.topk])
        for k in targets:
            maps = {m: cam(x, int(k))[0].detach() for m, cam in cams.items()}
            maps["random"] = torch.rand(x.shape[-2:], generator=rng)
            for method, sal in maps.items():
                r = deletion_insertion(model, x, sal, int(k), steps=args.steps, baseline=args.baseline)
                results[f"{method}/deletion_auc"].append(r["deletion_auc"])
                results[f"{method}/insertion_auc"].append(r["insertion_auc"])
                if labels[k] in annotated:
                    s = sal.cpu().numpy()
                    results[f"{method}/pointing_game"].append(float(pointing_game(s, annotated[labels[k]])))
                    results[f"{method}/box_energy"].append(energy_in_boxes(s, annotated[labels[k]]))
        print(f"\r{n + 1}/{len(indices)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    for cam in cams.values():
        cam.remove()

    summary = {k: float(np.mean(v)) for k, v in sorted(results.items())}
    summary["num_explanations"] = len(results.get("gradcam/deletion_auc", []))
    for k, v in summary.items():
        print(f"{k:<32} {v:.4f}" if isinstance(v, float) else f"{k:<32} {v}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
