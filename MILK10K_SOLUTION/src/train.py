from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dataset import MILK10kDataset, MetadataProcessor, build_meta_processor
from src.losses import compute_pos_weight, get_loss
from src.metrics import LABEL_COLS, concat_outputs, compute_metrics
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    AverageMeter, CSVLogger, EarlyStopping, Timer,
    get_device, get_optimizer, get_scheduler,
    save_checkpoint, set_seed, setup_logger,
)
from src.validate import _forward, validate_epoch


def train_epoch(model, loader, criterion, optimizer, scaler, device, clip_grad=1.0, use_amp=True):
    model.train()
    loss_meter = AverageMeter("train_loss")
    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        labels = batch.get("labels")
        if labels is None:
            continue
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = _forward(model, batch, device)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item(), n=labels.size(0))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
    return loss_meter.avg


def _build_datasets(cfg):
    train_csv = cfg["train_csv"]
    val_csv   = cfg["val_csv"]
    image_dir = cfg.get("image_dir")
    mode      = cfg.get("mode", "single_image")
    img_type  = cfg.get("image_type", "dermoscopy")
    use_meta  = cfg.get("use_metadata", False)
    img_size  = cfg.get("image_size", 224)

    train_transform = get_train_transforms(img_size)
    val_transform   = get_val_transforms(img_size)

    meta_processor = None
    if use_meta:
        train_df       = pd.read_csv(train_csv)
        meta_processor = build_meta_processor(train_df)

    train_ds = MILK10kDataset(
        csv_path=train_csv, image_dir=image_dir, transform=train_transform,
        mode=mode, image_type=img_type, is_test=False,
        meta_processor=meta_processor, cfg=cfg,
    )
    val_ds = MILK10kDataset(
        csv_path=val_csv, image_dir=image_dir, transform=val_transform,
        mode=mode, image_type=img_type, is_test=False,
        meta_processor=meta_processor, cfg=cfg,
    )
    return train_ds, val_ds, meta_processor


def _build_sampler(train_ds, cfg):
    if not cfg.get("use_weighted_sampler", False):
        return None
    df     = train_ds.df
    labels = df[LABEL_COLS].values if all(c in df.columns for c in LABEL_COLS) else None
    if labels is None:
        return None
    main_class = np.argmax(labels, axis=1)
    counts     = np.bincount(main_class, minlength=11).clip(min=1)
    sample_w   = 1.0 / counts[main_class]
    sample_w   = sample_w / sample_w.sum()
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )


def train(cfg, model):
    set_seed(cfg.get("seed", 42))
    device = get_device()
    model  = model.to(device)

    model_name = cfg.get("model_name", "model")
    ckpt_dir   = Path(cfg.get("checkpoint_dir", "outputs/checkpoints")) / model_name
    log_dir    = Path(cfg.get("output_dir",    "outputs/logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(model_name, log_file=str(log_dir / f"{model_name}_train.log"))

    logger.info("Building datasets ...")
    train_ds, val_ds, meta_processor = _build_datasets(cfg)
    logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    sampler     = _build_sampler(train_ds, cfg)
    num_workers = cfg.get("num_workers", 4)
    batch_size  = cfg.get("batch_size",  32)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    pos_weight = None
    if cfg.get("use_pos_weight", True) and cfg.get("loss_name", "bce").lower() == "bce":
        if all(c in train_ds.df.columns for c in LABEL_COLS):
            labels_np  = train_ds.df[LABEL_COLS].values.astype(float)
            pos_weight = compute_pos_weight(labels_np).to(device)
            logger.info(f"  pos_weight: {[round(x,2) for x in pos_weight.tolist()]}")

    criterion = get_loss(cfg.get("loss_name", "bce"), pos_weight=pos_weight, **cfg).to(device)

    optimizer = get_optimizer(
        model, cfg.get("optimizer", "adamw"),
        lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = get_scheduler(
        optimizer, cfg.get("scheduler", "cosine"),
        epochs=cfg.get("epochs", 30),
        steps_per_epoch=len(train_loader),
        warmup_epochs=cfg.get("warmup_epochs", 2),
        eta_min=cfg.get("eta_min", 1e-7),
    )

    use_amp = cfg.get("use_amp", True) and torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    early_stop = EarlyStopping(patience=cfg.get("early_stopping_patience", 10), mode="max")
    csv_log = CSVLogger(
        path=str(log_dir / f"{model_name}_train_log.csv"),
        fieldnames=["epoch","train_loss","val_loss","macro_f1","lr"]
                   + [f"f1_{c}" for c in LABEL_COLS],
    )

    best_f1   = -1.0
    timer     = Timer()
    epochs    = cfg.get("epochs", 30)
    clip_grad = cfg.get("clip_grad", 1.0)
    threshold = cfg.get("threshold", 0.5)

    logger.info(f"Starting training for {epochs} epochs ...")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, clip_grad, use_amp,
        )
        val_loss, mf1, pc_f1 = validate_epoch(
            model, val_loader, criterion, device,
            threshold=threshold, use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        sched_name = cfg.get("scheduler", "cosine").lower()
        if sched_name == "plateau":
            scheduler.step(mf1)
        else:
            scheduler.step()

        per_cls_str = " | ".join(f"{k}={v:.3f}" for k, v in pc_f1.items())
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"macro_f1={mf1:.4f} | lr={current_lr:.2e} | {timer.elapsed()}"
        )
        logger.info(f"  {per_cls_str}")

        csv_log.log({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "macro_f1":   round(mf1,        6),
            "lr":         round(current_lr, 10),
            **{f"f1_{c}": round(pc_f1.get(c, 0.0), 6) for c in LABEL_COLS},
        })

        if mf1 > best_f1:
            best_f1 = mf1
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "best_macro_f1": best_f1, "cfg": cfg},
                checkpoint_dir=str(ckpt_dir), filename="best.pth",
            )
            logger.info(f"  New best macro_f1={best_f1:.4f} -> saved best.pth")

        save_checkpoint(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "macro_f1": mf1, "cfg": cfg},
            checkpoint_dir=str(ckpt_dir), filename="last.pth",
        )

        if early_stop(mf1):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Done. Best macro_f1={best_f1:.4f} | Time: {timer.elapsed()}")
    return best_f1