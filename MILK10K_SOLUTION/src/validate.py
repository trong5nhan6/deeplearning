"""
validate.py — Validation loop for MILK10k training.
Returns val_loss, macro_f1, per_class_f1 for a single epoch.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_metrics, concat_outputs, sigmoid
from src.utils import AverageMeter


@torch.no_grad()
def validate_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    threshold: float = 0.5,
    use_amp:   bool  = True,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Run one validation pass.

    Returns
    -------
    val_loss   : average BCE loss over validation set
    macro_f1   : macro F1 with threshold=0.5 (official metric)
    per_cls_f1 : dict of per-class F1 scores
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")

    all_logits: list = []
    all_labels: list = []

    pbar = tqdm(loader, desc="  Val", leave=False, dynamic_ncols=True)

    for batch in pbar:
        labels = batch.get("labels")
        if labels is None:
            continue

        labels = labels.to(device, non_blocking=True)

        # ── Forward ──────────────────────────────────────────────────────────
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = _forward(model, batch, device)
            loss   = criterion(logits, labels)

        loss_meter.update(loss.item(), n=labels.size(0))
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    if not all_logits:
        return 0.0, 0.0, {}

    logits_np, labels_np = concat_outputs(all_logits, all_labels)
    metrics = compute_metrics(logits_np, labels_np, threshold=threshold)

    return loss_meter.avg, metrics["macro_f1"], metrics["per_class_f1"], metrics["accuracy"]


@torch.no_grad()
def collect_outputs(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool = True,
):
    """Collect all logits and labels from a dataloader (no loss computed)."""
    model.eval()
    all_logits: list = []
    all_labels: list = []
    for batch in loader:
        labels = batch.get("labels")
        if labels is None:
            continue
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = _forward(model, batch, device)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    return all_logits, all_labels


def _forward(model: nn.Module, batch: Dict, device: torch.device) -> torch.Tensor:
    """Route batch to model depending on mode (single vs dual image)."""
    if "clinical_image" in batch and "derm_image" in batch:
        # Dual-branch
        clin = batch["clinical_image"].to(device, non_blocking=True)
        derm = batch["derm_image"].to(device, non_blocking=True)
        meta = batch.get("metadata")
        if meta is not None:
            meta = meta.to(device, non_blocking=True)
        return model(clin, derm, meta)

    else:
        # Single image
        img  = batch["image"].to(device, non_blocking=True)
        meta = batch.get("metadata")
        if meta is not None:
            meta = meta.to(device, non_blocking=True)
        return model(img, meta)
