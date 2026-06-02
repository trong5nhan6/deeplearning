"""
utils.py — Shared utility functions for MILK10k training pipeline.
"""

from __future__ import annotations

import csv
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


# ── Config loading ───────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config and return as dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_cli_args(cfg: Dict, args) -> Dict:
    """Override config keys with non-None CLI arguments."""
    for k, v in vars(args).items():
        if v is not None and k != "config":
            cfg[k] = v
    return cfg


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / filename
    torch.save(state, str(path))
    return str(path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """Load checkpoint; returns the full state dict."""
    ckpt = torch.load(path, map_location=device)

    # Handle DataParallel / DDP saved models
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # Strip 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False  # prevent duplicate output in Jupyter/Colab

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

    return logger


class CSVLogger:
    """Appends rows to a CSV training log file."""

    def __init__(self, path: str, fieldnames: list):
        self.path       = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: Dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# ── AverageMeter ─────────────────────────────────────────────────────────────

class AverageMeter:
    """Keeps a running average for a single metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# ── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",   # 'max' for F1, 'min' for loss
        min_delta: float = 1e-5,
    ):
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return self.stop


# ── Scheduler helpers ─────────────────────────────────────────────────────────

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int = 1,
    warmup_epochs: int = 2,
    **kwargs,
):
    """
    Returns a scheduler.

    scheduler_name: 'cosine' | 'plateau' | 'step' | 'onecycle'
    warmup_epochs : number of warm-up epochs (linear ramp) prepended to cosine
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=kwargs.get("eta_min", 1e-7),
        )
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[warmup_epochs],
            )

    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
        )

    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.5),
        )

    elif scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 1e-3),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )

    else:
        raise ValueError(f"Unknown scheduler: '{scheduler_name}'")

    return scheduler


# ── Optimizer helper ──────────────────────────────────────────────────────────

def get_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float = 1e-4,
    layer_lr_decay: float = 1.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """Factory for optimizers. When layer_lr_decay < 1, backbone gets lr*decay, head gets lr."""
    optimizer_name = optimizer_name.lower()

    if layer_lr_decay < 1.0:
        # Collect backbone parameters, handling both SwinModel (.backbone) and
        # DualBranchModel (.clinical_branch / .derm_branch, may share weights).
        if hasattr(model, "backbone"):
            trunk_params = list(model.backbone.parameters())
        elif hasattr(model, "clinical_branch"):
            seen: set = set()
            trunk_params = []
            for branch in (model.clinical_branch, model.derm_branch):
                for p in branch.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        trunk_params.append(p)
        else:
            trunk_params = []

        if trunk_params:
            backbone_ids = {id(p) for p in trunk_params}
            params = [
                {"params": [p for p in model.parameters() if id(p) in backbone_ids],
                 "lr": lr * layer_lr_decay},
                {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
                 "lr": lr},
            ]
        else:
            params = model.parameters()  # type: ignore[assignment]
    else:
        params = model.parameters()  # type: ignore[assignment]

    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: '{optimizer_name}'")


# ── Device selection ──────────────────────────────────────────────────────────

def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev


# ── Timing utility ─────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> str:
        secs = int(time.time() - self.start)
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
