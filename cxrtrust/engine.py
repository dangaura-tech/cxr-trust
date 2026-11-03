"""Training loop with EMA weights, AMP and AUROC-based model selection."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import multilabel_report
from .utils import get_logger


class ModelEMA:
    """Exponential moving average of model weights (often +0.5-1 AUROC point for CXR models)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (logits, targets) as NumPy arrays."""
    model.eval()
    logits, targets = [], []
    for batch in loader:
        logits.append(model(batch["image"].to(device, non_blocking=True)).float().cpu())
        targets.append(batch["target"])
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, labels: Sequence[str]) -> Dict[str, float]:
    logits, targets = predict(model, loader, device)
    probs = 1 / (1 + np.exp(-logits))
    return multilabel_report(probs, targets, labels)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loaders: Dict[str, DataLoader],
        labels: Sequence[str],
        cfg: Dict,
        work_dir: str,
        device: torch.device,
    ) -> None:
        self.t = cfg["train"]
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.loaders = loaders
        self.labels = list(labels)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(str(self.work_dir / "train.log"))

        head = [p for n, p in model.named_parameters() if n.startswith("head")]
        body = [p for n, p in model.named_parameters() if not n.startswith("head")]
        lr = self.t["lr"]
        self.optimizer = torch.optim.AdamW(
            [{"params": body, "lr": lr}, {"params": head, "lr": lr * self.t.get("head_lr_mult", 10.0)}],
            weight_decay=self.t.get("weight_decay", 1e-5),
        )
        steps = self.t["epochs"] * len(loaders["train"])
        warmup = int(self.t.get("warmup_epochs", 1) * len(loaders["train"]))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: (
                (s + 1) / max(warmup, 1)
                if s < warmup
                else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1)))
            ),
        )
        self.use_amp = self.t.get("amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = ModelEMA(self.model, self.t["ema_decay"]) if self.t.get("ema_decay") else None
        (self.work_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    def _eval_model(self) -> nn.Module:
        return self.ema.ema if self.ema is not None else self.model

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(self.loaders["train"], 1):
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            m = batch["mask"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(x)
            loss = self.criterion(logits.float(), y, m)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.t.get("grad_clip", 5.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)
            total += loss.item() * x.size(0)
            n += x.size(0)
            if step % self.t.get("log_interval", 100) == 0:
                self.logger.info(f"epoch {epoch} [{step}/{len(self.loaders['train'])}] loss {total / n:.4f}")
        return total / max(n, 1)

    def fit(self) -> float:
        best = -1.0
        for epoch in range(1, self.t["epochs"] + 1):
            tic = time.time()
            loss = self.train_epoch(epoch)
            metrics = evaluate(self._eval_model(), self.loaders["val"], self.device, self.labels)
            score = metrics["auroc_macro"]
            if score > best:
                best = score
                torch.save(
                    {
                        "model": self._eval_model().state_dict(),
                        "config": self.cfg,
                        "labels": self.labels,
                        "epoch": epoch,
                        "val_auroc": score,
                    },
                    self.work_dir / "best.pth",
                )
            row = {
                "epoch": epoch,
                "loss": loss,
                "val_auroc": score,
                "val_auprc": metrics["auprc_macro"],
                "time_s": round(time.time() - tic, 1),
            }
            with (self.work_dir / "history.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.logger.info(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        self.logger.info(f"best val macro-AUROC {best:.4f}")
        return best
