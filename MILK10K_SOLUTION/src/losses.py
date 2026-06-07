"""
losses.py — Loss functions for skin lesion classification.

Includes:
  - WeightedBCEWithLogitsLoss  (pos_weight for class imbalance)
  - FocalLoss                  (sigmoid-based, multi-label)
  - SoftmaxFocalLoss           (softmax-based, multi-class single-label)
  - AsymmetricLoss (ASL)       (multi-label imbalance)
  - LogitAdjustmentLoss        (Menon et al. 2021, ICLR) — long-tail
  - LDAMLoss                   (Cao et al. 2019, NeurIPS) — margin-based long-tail
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. BCE with optional pos_weight ─────────────────────────────────────────

class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Standard BCEWithLogitsLoss with optional per-class positive weights
    to counteract class imbalance.

    Usage:
        # Compute pos_weight from training labels
        pos_weight = compute_pos_weight(train_labels)   # shape (11,)
        criterion  = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
    """

    def __init__(
        self,
        pos_weight:  Optional[torch.Tensor] = None,
        reduction:   str   = "mean",
        weight_clamp: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.reduction    = reduction
        self.weight_clamp = weight_clamp

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = (
            torch.clamp(self.pos_weight, max=self.weight_clamp)
            if (self.pos_weight is not None and self.weight_clamp > 0)
            else self.pos_weight
        )
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight,
            reduction=self.reduction,
        )


def compute_pos_weight(labels: np.ndarray, clip: float = 10.0) -> torch.Tensor:
    """
    Compute per-class pos_weight = (N - n_pos) / n_pos.
    labels : (N, 11) binary numpy array
    clip   : maximum value (prevent extreme weights for very rare classes)
    """
    n_pos = labels.sum(axis=0).clip(min=1)
    n_neg = len(labels) - n_pos
    w     = (n_neg / n_pos).clip(max=clip)
    return torch.tensor(w, dtype=torch.float32)


# ── 2. Focal Loss ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for multi-label classification.
    Reduces loss contribution from easy examples.

    alpha : balance factor between positive/negative
    gamma : focusing parameter (0 = BCE, 2 is common)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p       = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = p * targets + (1 - p) * (1 - targets)
        loss    = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss    = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ── 3. Softmax CrossEntropy ──────────────────────────────────────────────────

class SoftmaxCrossEntropyLoss(nn.Module):
    """
    Standard CrossEntropy for single-label multi-class classification.
    Accepts targets as one-hot (N, C) or class indices (N,).

    label_smoothing : label smoothing factor (0.0 = off)
    weight          : optional per-class weight tensor of shape (C,)
    loss_clamp      : if > 0, clamp per-sample loss to this max before averaging.
                      Prevents extreme-imbalance outliers from dominating a step.
                      Sensible value: log(C) * 2  (e.g. 4.0 for C=11 → log(11)≈2.4)
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        weight:          Optional[torch.Tensor] = None,
        loss_clamp:      float = 0.0,
        weight_clamp:    float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss_clamp      = loss_clamp
        self.weight_clamp    = weight_clamp
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        weight = (
            torch.clamp(self.weight, max=self.weight_clamp)
            if (self.weight is not None and self.weight_clamp > 0)
            else self.weight
        )
        loss = F.cross_entropy(
            logits, targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="none" if self.loss_clamp > 0 else "mean",
        )
        if self.loss_clamp > 0:
            loss = loss.clamp(max=self.loss_clamp).mean()
        return loss


# ── 4. Softmax Focal Loss ────────────────────────────────────────────────────

class SoftmaxFocalLoss(nn.Module):
    """
    Focal Loss with Softmax for single-label multi-class classification.

    Uses softmax + CrossEntropy instead of sigmoid + BCE, so class probabilities
    compete with each other — correct for datasets where each sample has exactly
    one ground-truth label.

    gamma           : focusing strength (0 = standard CE, 2 is common default)
    label_smoothing : label smoothing factor (0.0 = off)
    weight          : optional per-class weight tensor of shape (C,) to handle
                      class imbalance (computed via compute_class_weight)
    """

    def __init__(
        self,
        gamma:           float = 2.0,
        label_smoothing: float = 0.0,
        weight:          Optional[torch.Tensor] = None,
        weight_clamp:    float = 0.0,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.weight_clamp    = weight_clamp
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (N, C) one-hot float → (N,) class indices
        if targets.dim() == 2:
            targets_idx = targets.argmax(dim=1)
        else:
            targets_idx = targets.long()

        weight = (
            torch.clamp(self.weight, max=self.weight_clamp)
            if (self.weight is not None and self.weight_clamp > 0)
            else self.weight
        )

        # Per-sample CE loss (no reduction yet)
        ce = F.cross_entropy(
            logits, targets_idx,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Focal weight: down-weight easy examples
        probs = F.softmax(logits, dim=1)
        p_t   = probs.gather(1, targets_idx.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - p_t) ** self.gamma

        return (focal_w * ce).mean()


def compute_class_weight(
    labels: np.ndarray,
    clip:   float = 50.0,
    beta:   float = 0.9999,
    mode:   str   = "effective",
) -> torch.Tensor:
    """
    Per-class weight tensor for SoftmaxFocalLoss / CE.

    Parameters
    ----------
    labels : (N, C) one-hot numpy array
    clip   : max weight (raised from 10 → 50 to not crush MAL_OTH signal)
    beta   : hyperparameter for effective number (default 0.9999, Cui et al. 2019)
    mode   : "effective" — Effective Number of Samples (Cui et al. 2019)
                           w_k = (1 - β) / (1 - β^n_k)
             "inv_sqrt"  — w_k = 1 / sqrt(n_k)  (softer than raw inv-freq)
             "inv_freq"  — w_k = 1 / n_k  (original behaviour)

    Returns tensor of shape (C,), normalised so mean weight = 1.
    """
    counts = labels.sum(axis=0).clip(min=1)

    if mode == "effective":
        # Cui et al. 2019 — Class-Balanced Loss
        # Effective number = (1 - β^n) / (1 - β)
        eff_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / eff_num
    elif mode == "inv_sqrt":
        w = 1.0 / np.sqrt(counts)
    else:  # inv_freq
        w = 1.0 / counts

    w = w / w.mean()        # normalise: mean weight = 1
    w = w.clip(max=clip)
    return torch.tensor(w, dtype=torch.float32)


def compute_class_freq(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class prior π_k = n_k / N for Logit Adjustment.
    labels : (N, C) one-hot numpy array
    Returns tensor of shape (C,) summing to 1.
    """
    counts = labels.sum(axis=0).clip(min=1).astype(float)
    freq   = counts / counts.sum()
    return torch.tensor(freq, dtype=torch.float32)


# ── 5. Asymmetric Loss (ASL) ─────────────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (Ben-Baruch et al., 2021).
    Addresses positive-negative imbalance by applying asymmetric focusing.

    gamma_neg   : focusing strength for negatives (default 4)
    gamma_pos   : focusing strength for positives (default 0 = no focusing for pos)
    clip        : probability margin to shift negatives (shifts logits by this amount)
    eps         : numerical stability

    Reference: https://arxiv.org/abs/2009.14119
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Probability
        x_sigmoid = torch.sigmoid(logits)
        xs_pos    = x_sigmoid
        xs_neg    = 1 - x_sigmoid

        # Asymmetric clip (shift negative probabilities)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic BCE
        los_pos = targets       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss    = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)    # pt = p if target = 0 else 1 - p
            pt  = pt0 + pt1
            one_sided_gamma = (
                self.gamma_pos * targets
                + self.gamma_neg * (1 - targets)
            )
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.sum() / logits.shape[0]


# ── 6. Logit Adjustment Loss ─────────────────────────────────────────────────

class LogitAdjustmentLoss(nn.Module):
    """
    Long-tail learning via Logit Adjustment (Menon et al., ICLR 2021).

    Key idea: subtract τ·log(π_k) from logit of class k before softmax,
    which is equivalent to adjusting the decision boundary so that minority
    classes need less probability mass to "win".

    Formally:
        adjusted_logit_k = z_k - τ · log(π_k)

    At test time the same adjustment is applied (or the raw logit is used
    — both are valid; adjusting at train+test is most common).

    Parameters
    ----------
    class_freq    : π_k — class prior, tensor of shape (C,)
                    Computed via compute_class_freq(labels_np)
    tau           : temperature for adjustment strength (default 1.0)
                    τ=0 → standard CE; τ=1 → theoretically optimal for macro accuracy
    label_smoothing : smoothing factor (default 0.0)

    References
    ----------
    Menon et al. "Long-tail learning via logit adjustment." ICLR 2021.
    https://arxiv.org/abs/2007.07314
    """

    def __init__(
        self,
        class_freq:      torch.Tensor,
        tau:             float = 1.0,
        label_smoothing: float = 0.0,
        loss_clamp:      float = 0.0,
    ):
        super().__init__()
        self.tau             = tau
        self.label_smoothing = label_smoothing
        self.loss_clamp      = loss_clamp
        # log(π_k): shape (C,) — registered as buffer so .to(device) works
        self.register_buffer("log_prior", torch.log(class_freq.clamp(min=1e-9)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, C]
        targets : [B, C] one-hot float  OR  [B,] class indices (long)
        """
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)   # [B,]

        # Adjust logits: z_k + τ·log(π_k)
        # log_prior is negative for rare classes → subtracting negative = adding positive
        # → rare classes get boosted logit → easier to predict
        adjusted = logits + self.tau * self.log_prior   # broadcast (C,) over batch

        loss = F.cross_entropy(
            adjusted, targets,
            label_smoothing=self.label_smoothing,
            reduction="none" if self.loss_clamp > 0 else "mean",
        )
        if self.loss_clamp > 0:
            loss = loss.clamp(max=self.loss_clamp).mean()
        return loss


# ── 7. LDAM Loss ─────────────────────────────────────────────────────────────

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (Cao et al., NeurIPS 2019).

    Key idea: enforce a class-dependent margin Δ_k that is larger for minority
    classes. Derived from Rademacher complexity theory:

        Δ_k = C / n_k^(1/4)

    where C is a scaling constant and n_k is the number of training samples
    in class k.

    Modified logit for class k when it is the ground-truth:
        z_k → z_k - Δ_k

    Can be combined with DRW (Deferred Re-Weighting): train first with uniform
    class weights, then switch to class-weighted loss at epoch T (e.g. 2/3 of
    total epochs). Pass class_weight=None for stage 1, class_weight=w for stage 2.

    Parameters
    ----------
    class_counts  : n_k for each class, tensor of shape (C,)
                    Computed via labels_np.sum(axis=0)
    C             : margin scale (default 0.5 — tune if needed)
    class_weight  : optional per-class weight for DRW reweighting (C,)
    label_smoothing : smoothing factor

    References
    ----------
    Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware
    Margin Loss." NeurIPS 2019. https://arxiv.org/abs/1906.07413
    """

    def __init__(
        self,
        class_counts:    torch.Tensor,
        C:               float = 0.5,
        class_weight:    Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        weight_clamp:    float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.C            = C
        self.weight_clamp = weight_clamp

        # Δ_k = C / n_k^(1/4) — larger margin for minority classes
        margins = C / (class_counts.float().clamp(min=1) ** 0.25)
        self.register_buffer("margins", margins)              # (C,)
        self.register_buffer(
            "class_weight",
            class_weight if class_weight is not None
            else torch.ones(class_counts.shape[0]),
        )

    def set_weight(self, class_weight: Optional[torch.Tensor]):
        """Update class_weight at runtime (for DRW scheduling)."""
        if class_weight is not None:
            self.class_weight = class_weight.to(self.margins.device)
        else:
            self.class_weight = torch.ones_like(self.margins)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, C]
        targets : [B, C] one-hot float  OR  [B,] long
        """
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)   # [B,]

        B, C = logits.shape

        # Build margin mask: for each sample, subtract Δ_k only from GT class logit
        # Shape: [B, C] — 0 everywhere except GT class column where it's Δ_k
        margin_mask = torch.zeros_like(logits)                         # [B, C]
        margin_mask.scatter_(1, targets.unsqueeze(1), 1.0)             # one-hot
        margin_mask = margin_mask * self.margins.unsqueeze(0)          # [B, C]

        # Adjusted logits: penalise GT class → harder to be confident → larger margin
        adjusted = logits - margin_mask                                 # [B, C]

        class_weight = (
            torch.clamp(self.class_weight, max=self.weight_clamp)
            if self.weight_clamp > 0
            else self.class_weight
        )
        return F.cross_entropy(
            adjusted, targets,
            weight=class_weight,
            label_smoothing=self.label_smoothing,
        )


# ── Factory ──────────────────────────────────────────────────────────────────

def get_loss(
    loss_name:    str,
    pos_weight:   Optional[torch.Tensor] = None,
    class_weight: Optional[torch.Tensor] = None,
    class_freq:   Optional[torch.Tensor] = None,
    class_counts: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory for loss functions.

    loss_name    : 'bce' | 'focal' | 'focal_softmax' | 'asl'
                   'logit_adjustment' | 'ldam'
    pos_weight   : per-class weight tensor (C,) — for BCE
    class_weight : per-class weight tensor (C,) — for focal_softmax / ldam DRW
    class_freq   : class prior π_k tensor  (C,) — for logit_adjustment
    class_counts : raw class counts        (C,) — for ldam
    """
    loss_name = loss_name.lower()

    weight_clamp = kwargs.get("weight_clamp", 0.0)

    if loss_name in ("bce", "bce_with_logits"):
        return WeightedBCEWithLogitsLoss(pos_weight=pos_weight,
                                         weight_clamp=weight_clamp)

    elif loss_name == "focal":
        alpha = kwargs.get("focal_alpha", 0.25)
        gamma = kwargs.get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name in ("softmax_ce", "ce", "cross_entropy"):
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        loss_clamp      = kwargs.get("loss_clamp", 0.0)
        return SoftmaxCrossEntropyLoss(label_smoothing=label_smoothing,
                                       weight=class_weight,
                                       loss_clamp=loss_clamp,
                                       weight_clamp=weight_clamp)

    elif loss_name in ("focal_softmax", "softmax_focal", "ce_focal"):
        gamma           = kwargs.get("focal_gamma", 2.0)
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        return SoftmaxFocalLoss(gamma=gamma, label_smoothing=label_smoothing,
                                weight=class_weight, weight_clamp=weight_clamp)

    elif loss_name in ("asl", "asymmetric"):
        gamma_neg = kwargs.get("asl_gamma_neg", 4.0)
        gamma_pos = kwargs.get("asl_gamma_pos", 0.0)
        clip      = kwargs.get("asl_clip", 0.05)
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    elif loss_name in ("logit_adjustment", "la", "la_loss"):
        assert class_freq is not None, \
            "logit_adjustment requires class_freq — computed via compute_class_freq(labels_np)"
        tau             = kwargs.get("la_tau", 1.0)
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        loss_clamp      = kwargs.get("loss_clamp", 0.0)
        return LogitAdjustmentLoss(
            class_freq=class_freq,
            tau=tau,
            label_smoothing=label_smoothing,
            loss_clamp=loss_clamp,
        )

    elif loss_name in ("ldam", "ldam_drw"):
        assert class_counts is not None, \
            "ldam requires class_counts — computed via labels_np.sum(axis=0)"
        C               = kwargs.get("ldam_c", 0.5)
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        return LDAMLoss(
            class_counts=class_counts,
            C=C,
            class_weight=class_weight,   # None in stage 1, weighted in stage 2 (DRW)
            label_smoothing=label_smoothing,
            weight_clamp=weight_clamp,
        )

    else:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. "
            "Choose from: bce, focal, softmax_ce, focal_softmax, asl, "
            "logit_adjustment, ldam"
        )
