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
from src.losses import (
    compute_class_weight, compute_class_freq,
    compute_pos_weight, get_loss,
)
from src.metrics import LABEL_COLS, concat_outputs, compute_metrics
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    AverageMeter, CSVLogger, EarlyStopping, Timer,
    get_device, get_optimizer, get_scheduler,
    save_checkpoint, set_seed, setup_logger,
)
from src.validate import _forward, collect_outputs, validate_epoch
from src.visualize import (
    log_class_weights_table,
    log_confusion_matrix,
    visualize_embeddings,
)


def _log_final_summary(model, val_loader, device, threshold, use_amp, logger, cfg, ckpt_dir,
                        log_dir=None):
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

    # ── Confusion matrix ─────────────────────────────────────────────────────
    log_confusion_matrix(logits_np, labels_np, logger)


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


# ── cRT (Classifier Re-Training) helpers ─────────────────────────────────────

def _get_head_names(model: nn.Module) -> list:
    """
    Return the attribute names of the classifier head submodules.
    - HyCNN-Trans models: ['head_norm', 'head']
    - Single-branch models: ['classifier']
    """
    candidates = ["head_norm", "head", "classifier", "fc"]
    return [n for n in candidates if hasattr(model, n)
            and isinstance(getattr(model, n), nn.Module)]


def _freeze_backbone(model: nn.Module, head_names: list) -> int:
    """
    Freeze all parameters except those belonging to head_names modules.
    Returns number of frozen parameters (for logging).
    """
    head_param_ids = set()
    for name in head_names:
        for p in getattr(model, name).parameters():
            head_param_ids.add(id(p))

    n_frozen = 0
    for p in model.parameters():
        if id(p) not in head_param_ids:
            p.requires_grad_(False)
            n_frozen += p.numel()
    return n_frozen


def _unfreeze_all(model: nn.Module):
    """Re-enable gradients for all parameters."""
    for p in model.parameters():
        p.requires_grad_(True)


def _build_balanced_sampler(train_ds) -> WeightedRandomSampler:
    """
    Strictly class-balanced sampler: weight_i = 1 / n_{c_i}.
    Each class has equal expected representation per batch.
    """
    labels    = train_ds.df[LABEL_COLS].values
    main_cls  = np.argmax(labels, axis=1)
    counts    = np.bincount(main_cls, minlength=len(LABEL_COLS)).clip(min=1)
    sample_w  = 1.0 / counts[main_cls]
    sample_w  = sample_w / sample_w.sum()
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )


def _run_crt_stage2(model, train_ds, val_loader, cfg, device,
                    ckpt_dir: str, log_dir: Path, logger) -> float:
    """
    cRT Stage 2 — Kang et al. ICLR 2020.

    1. Load best stage-1 checkpoint.
    2. Freeze backbone; only classifier head is trainable.
    3. Train with class-balanced sampler + plain CrossEntropy for crt_epochs.
    4. Save best_crt.pth and last_crt.pth.

    WHY plain CE (not LA loss):
    - LA loss adjusts logits based on the original long-tail prior (BCC=48%, MAL_OTH=0.17%)
    - Balanced sampler already equalises class distribution (~9% each)
    - Using LA loss on top of balanced sampling double-corrects → overcorrects in wrong direction
    - Plain CE + balanced sampler is what the original cRT paper uses

    Returns best val macro-F1 achieved in stage 2.
    """
    from src.losses import SoftmaxCrossEntropyLoss
    from src.utils import load_checkpoint

    # ── load stage-1 best weights ────────────────────────────────────────────
    best_s1 = Path(ckpt_dir) / "best.pth"
    if best_s1.exists():
        load_checkpoint(str(best_s1), model, device=str(device))
        logger.info(f"[cRT] Loaded stage-1 best: {best_s1}")
    else:
        logger.warning("[cRT] best.pth not found — using current weights for stage 2")

    # ── freeze backbone ──────────────────────────────────────────────────────
    head_names  = _get_head_names(model)
    n_frozen    = _freeze_backbone(model, head_names)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[cRT] Frozen {n_frozen/1e6:.2f}M params | "
                f"Trainable {n_trainable/1e6:.2f}M params | head={head_names}")

    # ── cRT criterion: plain CrossEntropy (balanced sampler handles class dist)
    # Không dùng LA loss / focal ở đây — balanced sampler đã equalize rồi,
    # thêm LA loss sẽ double-correct và gây loss scale explosion (loss ~3.5 vs 0.9)
    crt_label_smoothing = cfg.get("crt_label_smoothing", cfg.get("label_smoothing", 0.1))
    crt_criterion = SoftmaxCrossEntropyLoss(
        label_smoothing=crt_label_smoothing,
        weight=None,   # không dùng class weight, balanced sampler đã lo
    ).to(device)
    logger.info(f"[cRT] criterion=CrossEntropy  label_smoothing={crt_label_smoothing}")

    # ── class-balanced dataloader ────────────────────────────────────────────
    crt_bs     = cfg.get("crt_batch_size", cfg.get("batch_size", 32))
    bal_loader = DataLoader(
        train_ds,
        batch_size=crt_bs,
        sampler=_build_balanced_sampler(train_ds),
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # ── optimizer on head params only ────────────────────────────────────────
    crt_lr      = cfg.get("crt_lr", 1e-5)
    head_params = []
    for name in head_names:
        head_params += list(getattr(model, name).parameters())
    optimizer = torch.optim.AdamW(
        head_params, lr=crt_lr,
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    crt_epochs = cfg.get("crt_epochs", 10)
    scheduler  = get_scheduler(
        optimizer, "cosine",
        epochs=crt_epochs,
        steps_per_epoch=len(bal_loader),
        warmup_epochs=max(1, crt_epochs // 10),
        eta_min=cfg.get("eta_min", 1e-7),
    )

    use_amp    = cfg.get("use_amp", True) and torch.cuda.is_available()
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp)
    clip_grad  = cfg.get("clip_grad", 1.0)
    threshold  = cfg.get("threshold", 0.5)
    model_name = cfg.get("model_name", "model")

    csv_log = CSVLogger(
        path=str(log_dir / f"{model_name}_crt_log.csv"),
        fieldnames=["epoch", "train_loss", "train_acc", "train_f1",
                    "val_loss", "val_acc", "macro_f1", "lr"]
                   + [f"f1_{c}" for c in LABEL_COLS],
    )

    best_f1 = -1.0
    timer   = Timer()
    logger.info(f"[cRT] Stage 2 | epochs={crt_epochs}  lr={crt_lr:.1e}  bs={crt_bs}")

    for epoch in range(1, crt_epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, bal_loader, crt_criterion, optimizer,
            scaler, device, clip_grad, use_amp, threshold,
        )
        val_loss, mf1, pc_f1, val_acc = validate_epoch(
            model, val_loader, crt_criterion, device,
            threshold=threshold, use_amp=use_amp,
        )
        scheduler.step()

        is_best  = mf1 > best_f1
        best_tag = " [BEST]" if is_best else ""
        cls_str  = "  ".join(f"{k[:4]}={v:.2f}" for k, v in pc_f1.items())
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[cRT {epoch:02d}/{crt_epochs}]"
            f"  Loss {train_loss:.4f}/{val_loss:.4f}"
            f"  Acc {train_acc:.4f}/{val_acc:.4f}"
            f"  F1 {train_f1:.4f}/{mf1:.4f}"
            f"  lr={current_lr:.1e}"
            f"  {timer.elapsed()}{best_tag}"
        )
        logger.info(f"         {cls_str}")

        csv_log.log({
            "epoch":      f"crt_{epoch}",
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
                {"epoch": f"crt_{epoch}", "model_state_dict": model.state_dict(),
                 "best_macro_f1": best_f1, "cfg": cfg},
                checkpoint_dir=ckpt_dir, filename="best_crt.pth",
            )
        save_checkpoint(
            {"epoch": f"crt_{epoch}", "model_state_dict": model.state_dict(),
             "macro_f1": mf1, "cfg": cfg},
            checkpoint_dir=ckpt_dir, filename="last_crt.pth",
        )

    _unfreeze_all(model)
    logger.info(f"[cRT] Stage 2 done  best_crt_f1={best_f1:.4f}")
    return best_f1


# ── BCL (Balanced Contrastive Learning) helpers ───────────────────────────────

def _setup_bcl(model: nn.Module, cfg: dict, train_ds, device, criterion):
    """
    Attach a ProjectionHead to the model via a forward hook on the classifier layer.

    Strategy:
    - HyCNN-Trans models: hook on model.head (Linear after head_norm) → input = feature [B, attn_dim]
    - Single-branch models: hook on first Linear inside model.classifier → input = backbone feature

    The hook stores features in feat_store["feat"] every forward pass,
    which train_epoch_bcl reads to build projected embeddings.

    Returns:
        proj_head    : ProjectionHead module (on device)
        bcl_criterion: BCLLoss module (on device)
        hook_handle  : RemovableHandle — call .remove() when done
        feat_store   : dict mutated in-place by the hook
    """
    from src.bcl import ProjectionHead, BCLLoss

    feat_store: dict = {}

    # ── find the linear layer to hook ────────────────────────────────────────
    hook_target = None
    # Priority: model.head (HyCNN) → model.classifier → model.fc
    for attr in ("head", "classifier", "fc"):
        if not hasattr(model, attr):
            continue
        mod = getattr(model, attr)
        if isinstance(mod, nn.Linear):
            hook_target = mod
            break
        elif isinstance(mod, nn.Sequential):
            # Hook the first Linear sub-module (feature enters before it)
            for sub in mod.modules():
                if isinstance(sub, nn.Linear):
                    hook_target = sub
                    break
            if hook_target is not None:
                break

    if hook_target is None:
        raise RuntimeError(
            "[BCL] Cannot find a Linear head to hook for feature extraction. "
            "Expected model.head, model.classifier, or model.fc."
        )

    def _hook_fn(module, inp, out):
        # inp is a tuple; inp[0] is the tensor entering the Linear layer
        feat_store["feat"] = inp[0]

    hook_handle = hook_target.register_forward_hook(_hook_fn)

    # ── detect feature dim via probe forward ──────────────────────────────────
    feat_in_dim = cfg.get("bcl_feat_in_dim") or cfg.get("attn_dim") or cfg.get("embed_dim")
    if feat_in_dim is None:
        from torch.utils.data import DataLoader as _DL
        probe_loader = _DL(train_ds, batch_size=2, shuffle=True,
                           num_workers=0, drop_last=False)
        model.eval()
        with torch.no_grad():
            probe_batch = next(iter(probe_loader))
            _forward(model, probe_batch, device)
        feat_in_dim = feat_store["feat"].shape[-1]
        model.train()

    proj_hidden = cfg.get("bcl_proj_hidden", 256)
    proj_dim    = cfg.get("bcl_proj_dim",    128)

    proj_head = ProjectionHead(
        in_dim=feat_in_dim, hidden_dim=proj_hidden, out_dim=proj_dim
    ).to(device)

    # Compute class_freq for WeightedSupConLoss
    bcl_class_freq = None
    if cfg.get("bcl_weighted_supcon", False):
        from src.losses import compute_class_freq
        from src.metrics import LABEL_COLS as _LC
        if all(c in train_ds.df.columns for c in _LC):
            _lbl = train_ds.df[_LC].values.astype(float)
            bcl_class_freq = compute_class_freq(_lbl).to(device)

    bcl_criterion = BCLLoss(
        num_classes=cfg.get("num_classes", 11),
        feat_dim=proj_dim,
        temperature=cfg.get("bcl_temperature",       0.07),
        lambda_sup=cfg.get("bcl_lambda_sup",         0.1),
        lambda_proto=cfg.get("bcl_lambda_proto",     0.1),
        proto_momentum=cfg.get("bcl_proto_momentum", 0.9),
        ce_criterion=criterion,
        use_weighted_supcon=cfg.get("bcl_weighted_supcon",       False),
        class_freq=bcl_class_freq,
        supcon_weight_clamp=cfg.get("bcl_supcon_weight_clamp",  20.0),
    ).to(device)

    return proj_head, bcl_criterion, hook_handle, feat_store


def train_epoch_bcl(
    model, proj_head, loader, bcl_criterion, feat_store,
    optimizer, proj_optimizer, scaler, device,
    clip_grad=1.0, use_amp=True, threshold=0.5,
):
    """
    Training epoch for BCL.

    Each step:
      1. model forward (hook captures backbone features into feat_store)
      2. proj_head(feat) → L2-normalized projected embeddings
      3. BCLLoss(logits, proj_feat, labels) = L_CE + λ₁·L_SupCon + λ₂·L_proto
      4. Backward through both model and proj_head

    Two separate optimizers share one AMP scaler; scaler.update() is called
    once per step after both .step() calls.
    """
    model.train()
    proj_head.train()

    loss_meter = AverageMeter("train_loss")
    all_logits: list = []
    all_labels: list = []

    pbar = tqdm(loader, desc="  BCL Train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        labels = batch.get("labels")
        if labels is None:
            continue
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        proj_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits    = _forward(model, batch, device)          # populates feat_store
            feat      = feat_store.get("feat")
            if feat is None:
                raise RuntimeError("[BCL] Forward hook did not capture features. "
                                   "Check that _setup_bcl found the correct layer.")
            proj_feat = proj_head(feat)                          # [B, proj_dim] normalized
            loss, loss_dict = bcl_criterion(logits, proj_feat, labels)

        scaler.scale(loss).backward()

        if clip_grad > 0:
            scaler.unscale_(optimizer)
            scaler.unscale_(proj_optimizer)
            all_p = list(model.parameters()) + list(proj_head.parameters())
            nn.utils.clip_grad_norm_(all_p, clip_grad)

        scaler.step(optimizer)
        scaler.step(proj_optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=labels.size(0))
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        pbar.set_postfix({
            "loss":    f"{loss_meter.avg:.4f}",
            "l_ce":    f"{loss_dict['l_ce']:.3f}",
            "l_sup":   f"{loss_dict['l_sup']:.3f}",
            "l_proto": f"{loss_dict['l_proto']:.3f}",
        })

    logits_np, labels_np = concat_outputs(all_logits, all_labels)
    metrics   = compute_metrics(logits_np, labels_np, threshold=threshold)
    train_acc = float(metrics.get("accuracy", 0.0))
    train_f1  = float(metrics.get("macro_f1", 0.0))
    return loss_meter.avg, train_acc, train_f1


# ── Training loop ─────────────────────────────────────────────────────────────

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
    image_dir = cfg.get("image_dir") or cfg.get("train_image_dir")
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

    pos_weight   = None
    class_weight = None
    class_freq   = None
    class_counts = None
    loss_name    = cfg.get("loss_name", "bce").lower()

    if all(c in train_ds.df.columns for c in LABEL_COLS):
        labels_np = train_ds.df[LABEL_COLS].values.astype(float)

        if cfg.get("use_pos_weight", True) and loss_name == "bce":
            pos_weight = compute_pos_weight(labels_np).to(device)
            logger.info(f"  pos_weight: {[round(x,2) for x in pos_weight.tolist()]}")

        # class_weight: dùng cho focal_softmax, ldam (DRW stage 2)
        _cw_losses = ("softmax_ce", "ce", "cross_entropy",
                      "focal_softmax", "softmax_focal", "ce_focal",
                      "ldam", "ldam_drw")
        if loss_name in _cw_losses:
            cw_mode = cfg.get("class_weight_mode", "effective")
            class_weight = compute_class_weight(
                labels_np,
                clip=cfg.get("class_weight_clip", 50.0),
                beta=cfg.get("class_weight_beta", 0.9999),
                mode=cw_mode,
            ).to(device)
            logger.info(f"  class_weight ({cw_mode}): "
                        f"{[round(x,2) for x in class_weight.tolist()]}")

        # class_freq: dùng cho logit_adjustment
        if loss_name in ("logit_adjustment", "la", "la_loss"):
            class_freq = compute_class_freq(labels_np).to(device)
            logger.info(f"  class_freq: {[round(x,4) for x in class_freq.tolist()]}")

        # class_counts: dùng cho ldam
        if loss_name in ("ldam", "ldam_drw"):
            counts_np    = labels_np.sum(axis=0).clip(min=1)
            class_counts = torch.tensor(counts_np, dtype=torch.float32).to(device)
            logger.info(f"  class_counts: {[int(x) for x in counts_np.tolist()]}")

    # ── Log class weight table before training ───────────────────────────────
    if all(c in train_ds.df.columns for c in LABEL_COLS):
        log_class_weights_table(
            labels_np=train_ds.df[LABEL_COLS].values.astype(float),
            class_weight=class_weight,
            pos_weight=pos_weight,
            class_freq=class_freq,
            weight_clamp=cfg.get("weight_clamp", 0.0),
            logger=logger,
        )

    loss_kwargs = {k: v for k, v in cfg.items() if k != "loss_name"}
    criterion = get_loss(
        loss_name,
        pos_weight=pos_weight,
        class_weight=class_weight,
        class_freq=class_freq,
        class_counts=class_counts,
        **loss_kwargs,
    ).to(device)

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

    # ── BCL setup (optional) ─────────────────────────────────────────────────
    use_bcl        = cfg.get("use_bcl", False)
    proj_head      = None
    bcl_criterion  = None
    bcl_hook       = None
    feat_store     = None
    proj_optimizer = None

    if use_bcl:
        proj_head, bcl_criterion, bcl_hook, feat_store = _setup_bcl(
            model, cfg, train_ds, device, criterion
        )
        proj_lr        = cfg.get("bcl_proj_lr", cfg.get("lr", 1e-4))
        proj_optimizer = torch.optim.AdamW(
            proj_head.parameters(),
            lr=proj_lr,
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        proj_dim = cfg.get("bcl_proj_dim", 128)
        logger.info(
            f"[BCL] enabled  proj_dim={proj_dim}"
            f"  λ_sup={cfg.get('bcl_lambda_sup',0.1)}"
            f"  λ_proto={cfg.get('bcl_lambda_proto',0.1)}"
            f"  τ={cfg.get('bcl_temperature',0.07)}"
            f"  proj_lr={proj_lr:.1e}"
        )

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
        if use_bcl:
            train_loss, train_acc, train_f1 = train_epoch_bcl(
                model, proj_head, train_loader, bcl_criterion, feat_store,
                optimizer, proj_optimizer, scaler, device,
                clip_grad, use_amp, threshold,
            )
        else:
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

    # ── BCL cleanup ──────────────────────────────────────────────────────────
    if use_bcl and bcl_hook is not None:
        bcl_hook.remove()
        logger.info("[BCL] Feature hook removed after stage 1.")

    # ── Stage 2: cRT (optional) ──────────────────────────────────────────────
    if cfg.get("use_crt", False):
        import shutil
        crt_f1 = _run_crt_stage2(
            model, train_ds, val_loader, cfg, device,
            str(ckpt_dir), log_dir, logger,
        )
        # Nếu cRT cải thiện F1, dùng best_crt.pth làm best.pth chính thức
        crt_best_path = Path(ckpt_dir) / "best_crt.pth"
        if crt_best_path.exists() and crt_f1 > best_f1:
            shutil.copy2(str(crt_best_path), str(Path(ckpt_dir) / "best.pth"))
            logger.info(
                f"[cRT] best_crt.pth (f1={crt_f1:.4f}) > stage-1 best (f1={best_f1:.4f})"
                " → replaced best.pth"
            )
            best_f1 = crt_f1
        else:
            logger.info(
                f"[cRT] cRT f1={crt_f1:.4f} did not improve stage-1 f1={best_f1:.4f}"
                " — keeping original best.pth"
            )

    _log_final_summary(model, val_loader, device, threshold, use_amp, logger, cfg,
                       str(ckpt_dir), log_dir=log_dir)

    # ── Embedding visualization ──────────────────────────────────────────────
    try:
        # Prototypes from BCL (if available)
        _prototypes = None
        _proto_init = None
        if use_bcl and bcl_criterion is not None:
            _prototypes = bcl_criterion.proto.prototypes
            _proto_init = bcl_criterion.proto.initialized

        viz_title = (
            "BCL Embedding Space — contrastive projection (val set)"
            if use_bcl else
            "Backbone Embedding Space (val set)"
        )
        visualize_embeddings(
            model=model,
            val_loader=val_loader,
            device=device,
            save_path=str(log_dir / f"{model_name}_embeddings.png"),
            proj_head=proj_head,       # None if not BCL
            prototypes=_prototypes,
            proto_init=_proto_init,
            use_amp=use_amp,
            title=viz_title,
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"[VIZ] Embedding visualization failed: {e}")

    _auto_submit(model, cfg, device, use_amp, str(ckpt_dir), meta_processor, logger)
    logger.info(f"Done  best_val_f1={best_f1:.4f}  total_time={timer.elapsed()}")
    return best_f1
