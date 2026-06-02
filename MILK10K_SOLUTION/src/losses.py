"""
losses.py — Loss functions for multi-label classification.

Includes:
  - WeightedBCEWithLogitsLoss  (pos_weight for class imbalance)
  - FocalLoss
  - AsymmetricLoss (ASL) — best for multi-label imbalance
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

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
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


# ── 3. Asymmetric Loss (ASL) ─────────────────────────────────────────────────

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


# ── Factory ──────────────────────────────────────────────────────────────────

def get_loss(
    loss_name: str,
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory for loss functions.

    loss_name : 'bce' | 'focal' | 'asl' | 'asymmetric'
    pos_weight: optional tensor of shape (11,) for BCE
    """
    loss_name = loss_name.lower()

    if loss_name in ("bce", "bce_with_logits"):
        return WeightedBCEWithLogitsLoss(pos_weight=pos_weight)

    elif loss_name == "focal":
        alpha = kwargs.get("focal_alpha", 0.25)
        gamma = kwargs.get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name in ("asl", "asymmetric"):
        gamma_neg = kwargs.get("asl_gamma_neg", 4.0)
        gamma_pos = kwargs.get("asl_gamma_pos", 0.0)
        clip      = kwargs.get("asl_clip", 0.05)
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    else:
        raise ValueError(f"Unknown loss: '{loss_name}'. Choose from: bce, focal, asl")
