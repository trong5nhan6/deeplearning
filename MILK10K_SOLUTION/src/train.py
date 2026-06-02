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
from src.validate import _forward, collect_outputs, validate_epoch


def _log_final_summary(model, val_loader, device, threshold, use_amp, logger, cfg, ckpt_dir):
    """Load best weights, run final val pass, log comprehensive metrics and model info."""
    from src.metrics import compute_full_summary
    from src.utils import load_checkpoint

    best_path = Path(ckpt_dir) / "best.pth"
    if best_path.exists():
        load_checkpoint(str(best_path), model, device=str(device))
        logger.info(f"Loaded best weights: {best_path}")
    else:
        logger.warning(f"best.pth not found at {best_path} — summary uses last-epoch weights")

    all_logits, all_labels = collect_outputs(model, val_loader, device, use_amp=use_amp)
    logits_np, labels_np   = concat_outputs(all_logits, all_labels)
    s = compute_full_summary(logits_np, labels_np, threshold=threshold)

    logger.info("=" * 50)
    logger.info("===== FINAL VALIDATION SUMMARY (best checkpoint) =====")
    logger.info(f"  Acc@1      : {s['acc1']:.4f}")
    logger.info(f"  Acc@5      : {s['acc5']:.4f}")
    logger.info(f"  Precision  : {s['precision']:.4f}")
    logger.info(f"  Recall     : {s['recall']:.4f}")
    logger.info(f"  Macro F1   : {s['macro_f1']:.4f}")
    logger.info(f"  ROC-AUC    : {s['roc_auc']:.4f}")
    logger.info("===== MODEL INFO =====")
    logger.info(f"  Model      : {cfg.get('model_name', '?')}")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    img_size = cfg.get("image_size", 224)
    gflops   = None
    try:
        from torchinfo import summary as _ti_summary
        ti = _ti_summary(
            model, input_size=(1, 3, img_size, img_size),
            verbose=0, device=str(device),
        )
        gflops = ti.total_mult_adds / 1e9
    except Exception:
        pass

    if gflops is not None:
        logger.info(f"  GFLOPs     : {gflops:.2f}")
    logger.info(f"  Params (M) : {params_m:.2f}")
    logger.info("=" * 50)


def _auto_submit(model, cfg, device, use_amp, ckpt_dir, meta_processor, logger):
    """Run inference with best.pth and last.pth, save two submission CSVs."""
    from src.infer import run_inference
    from src.submission import build_submission
    from src.utils import load_checkpoint

    test_csv     = cfg.get("test_csv")
    test_img_dir = cfg.get("test_image_dir") or cfg.get("image_dir")
    sub_dir      = Path(cfg.get("submission_dir", "outputs/submissions"))
    model_name   = cfg.get("model_name", "model")

    if not test_csv or not Path(test_csv).exists():
        logger.info("test_csv not found — skipping auto submission")
        return

    sub_dir.mkdir(parents=True, exist_ok=True)
    test_ds = MILK10kDataset(
        csv_path=test_csv, image_dir=test_img_dir,
        transform=get_val_transforms(cfg.get("image_size", 224)),
        mode=cfg.get("mode", "single_image"),
        image_type=cfg.get("image_type", "dermoscopy"),
        is_test=True, meta_processor=meta_processor, cfg=cfg,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.get("batch_size", 32) * 2,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    logger.info("===== AUTO SUBMISSION =====")
    for tag in ("best", "last"):
        ckpt_path = Path(ckpt_dir) / f"{tag}.pth"
        if not ckpt_path.exists():
            logger.info(f"  {tag}.pth not found — skipped")
            continue
        load_checkpoint(str(ckpt_path), model, device=str(device))
        lesion_ids, probs = run_inference(model, test_loader, device, use_amp=use_amp)
        out_path = str(sub_dir / f"submission_{model_name}_{tag}.csv")
        build_submission(lesion_ids, probs, out_path)
        logger.info(f"  [{tag}] saved: {out_path}")


def _apply_mixup(batch, labels, alpha, device):
    """Mix images within a batch for MixUp augmentation."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(labels.size(0), device=device)
    labels_b = labels[idx]
    for key in ("image", "derm_image", "clinical_image"):
        if key in batch:
            img = batch[key].to(device, non_blocking=True)
            batch[key] = lam * img + (1 - lam) * img[idx]
    return batch, labels_b, lam


def train_epoch(model, loader, criterion, optimizer, scaler, device,
                clip_grad=1.0, use_amp=True, threshold=0.5, mixup_alpha=0.0):
    model.train()
    loss_meter = AverageMeter("train_loss")
    all_logits: list = []
    all_labels: list = []
    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        labels = batch.get("labels")
        if labels is None:
            continue
        labels = labels.to(device, non_blocking=True)

        labels_b, lam = None, 1.0
        if mixup_alpha > 0:
            batch, labels_b, lam = _apply_mixup(batch, labels, mixup_alpha, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = _forward(model, batch, device)
            if labels_b is not None:
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        loss_meter.update(loss.item(), n=labels.size(0))
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    logits_np, labels_np = concat_outputs(all_logits, all_labels)
    metrics   = compute_metrics(logits_np, labels_np, threshold=threshold)
    train_acc = float(metrics.get("accuracy", 0.0))
    train_f1  = float(metrics.get("macro_f1", 0.0))
    return loss_meter.avg, train_acc, train_f1


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
    log_dir    = Path(cfg.get("output_dir",    "outputs/logs")) / model_name
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

    loss_kwargs = {k: v for k, v in cfg.items() if k != "loss_name"}
    criterion = get_loss(cfg.get("loss_name", "bce"), pos_weight=pos_weight, **loss_kwargs).to(device)

    optimizer = get_optimizer(
        model, cfg.get("optimizer", "adamw"),
        lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4),
        layer_lr_decay=cfg.get("layer_lr_decay", 1.0),
    )
    scheduler = get_scheduler(
        optimizer, cfg.get("scheduler", "cosine"),
        epochs=cfg.get("epochs", 30),
        steps_per_epoch=len(train_loader),
        warmup_epochs=cfg.get("warmup_epochs", 2),
        eta_min=cfg.get("eta_min", 1e-7),
    )

    use_amp = cfg.get("use_amp", True) and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    early_stop = EarlyStopping(patience=cfg.get("early_stopping_patience", 10), mode="max")
    csv_log = CSVLogger(
        path=str(log_dir / f"{model_name}_train_log.csv"),
        fieldnames=["epoch","train_loss","train_acc","train_f1","val_loss","val_acc","macro_f1","lr"]
                   + [f"f1_{c}" for c in LABEL_COLS],
    )

    best_f1   = -1.0
    timer     = Timer()
    epochs    = cfg.get("epochs", 30)
    clip_grad = cfg.get("clip_grad", 1.0)
    threshold = cfg.get("threshold", 0.5)

    logger.info(f"Training {model_name}  epochs={epochs}  bs={cfg.get('batch_size',32)}  lr={cfg.get('lr',1e-4):.1e}")
    logger.info(f"{'Epoch':>10}  {'Loss tr/val':>14}  {'Acc tr/val':>12}  {'F1 tr/val':>12}  LR")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, clip_grad, use_amp, threshold,
            mixup_alpha=cfg.get("mixup_alpha", 0.0),
        )
        val_loss, mf1, pc_f1, val_acc = validate_epoch(
            model, val_loader, criterion, device,
            threshold=threshold, use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[-1]["lr"]  # head LR (last group)
        sched_name = cfg.get("scheduler", "cosine").lower()
        if sched_name == "plateau":
            scheduler.step(mf1)
        else:
            scheduler.step()

        is_best = mf1 > best_f1
        best_tag = " [BEST]" if is_best else ""

        # Compact per-class F1 — two chars abbrev
        cls_str = "  ".join(f"{k[:4]}={v:.2f}" for k, v in pc_f1.items())

        logger.info(
            f"[{epoch:03d}/{epochs}]"
            f"  Loss {train_loss:.4f}/{val_loss:.4f}"
            f"  Acc {train_acc:.4f}/{val_acc:.4f}"
            f"  F1 {train_f1:.4f}/{mf1:.4f}"
            f"  lr={current_lr:.1e}"
            f"  {timer.elapsed()}"
            f"{best_tag}"
        )
        logger.info(f"         {cls_str}")

        csv_log.log({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "train_f1":   round(train_f1,   6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "macro_f1":   round(mf1,        6),
            "lr":         round(current_lr, 10),
            **{f"f1_{c}": round(pc_f1.get(c, 0.0), 6) for c in LABEL_COLS},
        })

        if is_best:
            best_f1 = mf1
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "best_macro_f1": best_f1, "cfg": cfg},
                checkpoint_dir=str(ckpt_dir), filename="best.pth",
            )

        save_checkpoint(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "macro_f1": mf1, "cfg": cfg},
            checkpoint_dir=str(ckpt_dir), filename="last.pth",
        )

        if early_stop(mf1):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    _log_final_summary(model, val_loader, device, threshold, use_amp, logger, cfg, str(ckpt_dir))
    _auto_submit(model, cfg, device, use_amp, str(ckpt_dir), meta_processor, logger)
    logger.info(f"Done  best_val_f1={best_f1:.4f}  total_time={timer.elapsed()}")
    return best_f1