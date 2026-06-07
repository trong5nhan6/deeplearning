"""
BCL -- Balanced Contrastive Learning (Zhu et al. CVPR 2022)
+ Queue-based contrastive (BPaCo-style, MICCAI 2024)
+ Feature-space augmentation for minority classes
"""
from __future__ import annotations
import random
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """2-layer MLP -> BN -> ReLU -> Linear -> L2-norm. Output on unit hypersphere."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# ---------------------------------------------------------------------------
# 2. SupConLoss -- extended with optional extra_feats pool (aug + queue)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. NeurIPS 2020).

    Extended: accepts optional extra_feats/extra_labels that extend the
    positive/negative pool beyond the current batch.
    Only `features` are used as anchors (get gradients).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features:     torch.Tensor,
        labels:       torch.Tensor,
        extra_feats:  Optional[torch.Tensor] = None,
        extra_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, device = features.shape[0], features.device

        if extra_feats is not None and extra_feats.size(0) > 0:
            pool_f = torch.cat([features, extra_feats.to(device)], dim=0)
            pool_l = torch.cat([labels,   extra_labels.to(device)], dim=0)
        else:
            pool_f, pool_l = features, labels
        N = pool_f.shape[0]

        sim = torch.matmul(features, pool_f.T) / self.temperature  # [B, N]

        mask_self = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_self[:, :B] = torch.eye(B, dtype=torch.bool, device=device)
        mask_pos = (labels.unsqueeze(1) == pool_l.unsqueeze(0)) & ~mask_self

        if mask_pos.sum() == 0:
            return features.sum() * 0.0

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim) * (~mask_self).float()
        log_sum = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        log_prob = sim - log_sum

        n_pos = mask_pos.float().sum(dim=1).clamp(min=1)
        loss_per_anchor = -(mask_pos.float() * log_prob).sum(dim=1) / n_pos
        has_pos = mask_pos.any(dim=1)
        return loss_per_anchor[has_pos].mean()


# ---------------------------------------------------------------------------
# 3. WeightedSupConLoss -- minority-amplified, extended with extra_feats pool
# ---------------------------------------------------------------------------

class WeightedSupConLoss(nn.Module):
    """
    Weighted SupConLoss. Each anchor weight = 1/pi_k (normalised, clamped).
    Extended with same extra_feats/extra_labels pool as SupConLoss.
    """

    def __init__(self, temperature: float = 0.07, weight_clamp: float = 3.0):
        super().__init__()
        self.temperature  = temperature
        self.weight_clamp = weight_clamp

    def forward(
        self,
        features:     torch.Tensor,
        labels:       torch.Tensor,
        class_freq:   Optional[torch.Tensor] = None,
        extra_feats:  Optional[torch.Tensor] = None,
        extra_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, device = features.shape[0], features.device

        if extra_feats is not None and extra_feats.size(0) > 0:
            pool_f = torch.cat([features, extra_feats.to(device)], dim=0)
            pool_l = torch.cat([labels,   extra_labels.to(device)], dim=0)
        else:
            pool_f, pool_l = features, labels
        N = pool_f.shape[0]

        sim = torch.matmul(features, pool_f.T) / self.temperature  # [B, N]

        mask_self = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_self[:, :B] = torch.eye(B, dtype=torch.bool, device=device)
        mask_pos = (labels.unsqueeze(1) == pool_l.unsqueeze(0)) & ~mask_self

        if mask_pos.sum() == 0:
            return features.sum() * 0.0

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim) * (~mask_self).float()
        log_sum = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        log_prob = sim - log_sum

        n_pos = mask_pos.float().sum(dim=1).clamp(min=1)
        loss_per_anchor = -(mask_pos.float() * log_prob).sum(dim=1) / n_pos
        has_pos = mask_pos.any(dim=1)

        if has_pos.sum() == 0:
            return features.sum() * 0.0

        if class_freq is None:
            return loss_per_anchor[has_pos].mean()

        weights = 1.0 / class_freq[labels].clamp(min=1e-9)
        weights = weights / weights.mean()
        weights = torch.clamp(weights, max=self.weight_clamp)
        w    = weights[has_pos]
        loss = loss_per_anchor[has_pos]
        return (w * loss).sum() / w.sum()


# ---------------------------------------------------------------------------
# 4. PrototypeLoss (unchanged from original BCL)
# ---------------------------------------------------------------------------

class PrototypeLoss(nn.Module):
    """EMA prototype per class. Loss = CE(sim(z, protos)/tau, y)."""

    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum    = momentum
        self.num_classes = num_classes
        self.register_buffer("prototypes",  torch.zeros(num_classes, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            feat_c = F.normalize(features[mask].mean(dim=0), dim=0)
            if not self.initialized[c]:
                self.prototypes[c]  = feat_c
                self.initialized[c] = True
            else:
                proto = self.momentum * self.prototypes[c] + (1.0 - self.momentum) * feat_c
                self.prototypes[c]  = F.normalize(proto, dim=0)

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
        valid = self.initialized
        if valid.sum() < 2:
            return features.sum() * 0.0

        protos        = F.normalize(self.prototypes[valid], dim=1)
        valid_classes = torch.where(valid)[0]

        class_to_idx = torch.full(
            (self.num_classes,), -1, dtype=torch.long, device=features.device
        )
        for new_i, orig_c in enumerate(valid_classes):
            class_to_idx[orig_c] = new_i

        has_proto = valid[labels]
        if has_proto.sum() == 0:
            return features.sum() * 0.0

        feat_v = features[has_proto]
        lbl_v  = class_to_idx[labels[has_proto]]
        sim    = torch.matmul(feat_v, protos.T) / temperature
        return F.cross_entropy(sim, lbl_v)


# ---------------------------------------------------------------------------
# 5. BCLLoss -- combined loss with Queue + Feature Augmentation
# ---------------------------------------------------------------------------

class BCLLoss(nn.Module):
    """
    Balanced Contrastive Learning Loss + Queue + Feature-space augmentation.

    L_total = L_CE + lambda_sup * L_SupCon + lambda_proto * L_proto

    Queue (BPaCo-style):
      MoCo-style FIFO buffer that stores encoded features from recent batches.
      SupCon denominator pool = current batch + queue -> minority classes
      always participate even when absent from current batch.

    Feature augmentation:
      After feat_aug_warmup epochs, generates virtual features for minority
      classes: z_aug = alpha*z_real + (1-alpha)*proto + noise.
      Injected into SupCon pool only (not CE or PrototypeLoss).
    """

    def __init__(
        self,
        num_classes:              int,
        feat_dim:                 int   = 128,
        temperature:              float = 0.07,
        lambda_sup:               float = 0.1,
        lambda_proto:             float = 0.1,
        proto_momentum:           float = 0.9,
        ce_criterion:             Optional[nn.Module] = None,
        use_weighted_supcon:      bool  = False,
        class_freq:               Optional[torch.Tensor] = None,
        supcon_weight_clamp:      float = 3.0,
        queue_size:               int   = 0,
        feat_aug:                 bool  = False,
        feat_aug_warmup:          int   = 5,
        feat_aug_minority_thresh: int   = 100,
        feat_aug_per_sample:      int   = 4,
        feat_aug_alpha_min:       float = 0.3,
        feat_aug_alpha_max:       float = 0.7,
        feat_aug_noise_std:       float = 0.05,
        class_counts:             Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_sup          = lambda_sup
        self.lambda_proto        = lambda_proto
        self.temperature         = temperature
        self.use_weighted_supcon = use_weighted_supcon

        if use_weighted_supcon:
            self.supcon = WeightedSupConLoss(
                temperature=temperature, weight_clamp=supcon_weight_clamp,
            )
        else:
            self.supcon = SupConLoss(temperature=temperature)

        self.proto = PrototypeLoss(
            num_classes=num_classes, feat_dim=feat_dim, momentum=proto_momentum,
        )
        self.ce = ce_criterion

        if class_freq is not None:
            self.register_buffer("class_freq", class_freq)
        else:
            self.class_freq = None

        # Queue
        self.queue_size = queue_size
        if queue_size > 0:
            self.register_buffer(
                "_qf", F.normalize(torch.randn(queue_size, feat_dim), dim=1),
            )
            self.register_buffer(
                "_ql", torch.full((queue_size,), -1, dtype=torch.long),
            )
            self.register_buffer("_qptr", torch.zeros(1, dtype=torch.long))
        self._qvalid = 0  # not a buffer; resets on reload (queue refills fast)

        # Feature augmentation
        self.feat_aug                 = feat_aug
        self.feat_aug_warmup          = feat_aug_warmup
        self.feat_aug_minority_thresh = feat_aug_minority_thresh
        self.feat_aug_per_sample      = feat_aug_per_sample
        self.feat_aug_alpha_min       = feat_aug_alpha_min
        self.feat_aug_alpha_max       = feat_aug_alpha_max
        self.feat_aug_noise_std       = feat_aug_noise_std

        if class_counts is not None:
            self.register_buffer("class_counts", class_counts.float())
        else:
            self.class_counts = None

    # -- Queue helpers --

    @torch.no_grad()
    def _enqueue(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        if self.queue_size == 0:
            return
        B   = feats.size(0)
        ptr = int(self._qptr)
        end = ptr + B
        if end <= self.queue_size:
            self._qf[ptr:end] = feats
            self._ql[ptr:end] = labels
        else:
            part1 = self.queue_size - ptr
            self._qf[ptr:]       = feats[:part1]
            self._ql[ptr:]       = labels[:part1]
            self._qf[:B - part1] = feats[part1:]
            self._ql[:B - part1] = labels[part1:]
        self._qptr[0] = (ptr + B) % self.queue_size
        self._qvalid  = min(self._qvalid + B, self.queue_size)

    def _get_queue(self):
        if self.queue_size == 0 or self._qvalid == 0:
            return None, None
        n = self._qvalid
        return self._qf[:n].clone(), self._ql[:n].clone()

    # -- Feature augmentation helper --

    @torch.no_grad()
    def _augment_features(self, feats_det: torch.Tensor, labels: torch.Tensor):
        if self.class_counts is None:
            return None, None
        aug_list, lbl_list = [], []
        device = feats_det.device
        for c in range(self.proto.num_classes):
            if not self.proto.initialized[c]:
                continue
            if self.class_counts[c] > self.feat_aug_minority_thresh:
                continue
            mask = labels == c
            if mask.sum() == 0:
                continue
            real_feats = feats_det[mask]
            proto_c    = F.normalize(self.proto.prototypes[c].detach(), dim=0)
            for _ in range(self.feat_aug_per_sample):
                for rf in real_feats:
                    alpha = random.uniform(self.feat_aug_alpha_min, self.feat_aug_alpha_max)
                    z = alpha * rf + (1.0 - alpha) * proto_c
                    noise = torch.randn_like(z) * self.feat_aug_noise_std
                    z = F.normalize(z + noise, dim=0)
                    aug_list.append(z)
                    lbl_list.append(c)
        if not aug_list:
            return None, None
        aug_t = torch.stack(aug_list).to(device)
        lbl_t = torch.tensor(lbl_list, dtype=torch.long, device=device)
        return aug_t, lbl_t

    # -- Main forward --

    def forward(
        self,
        logits:    torch.Tensor,
        proj_feat: torch.Tensor,
        targets:   torch.Tensor,
        epoch:     int = 0,
    ):
        """
        logits    : [B, C]  raw classification logits
        proj_feat : [B, D]  L2-normalized projected features
        targets   : [B, C]  one-hot  OR  [B,] long class indices
        epoch     : int     current epoch (gate for feat_aug warmup)
        Returns   : (total_loss, info_dict)
        """
        labels = targets.argmax(dim=1) if targets.dim() == 2 else targets.long()

        # 1. CE loss
        l_ce = (
            self.ce(logits, targets) if self.ce is not None
            else torch.tensor(0.0, device=logits.device)
        )

        # 2. Update prototypes first (aug uses them)
        self.proto.update(proj_feat.detach(), labels)

        # 3. Feature augmentation (minority, after warmup)
        aug_feats, aug_labels = None, None
        if (self.feat_aug and epoch >= self.feat_aug_warmup
                and self.proto.initialized.any()):
            aug_feats, aug_labels = self._augment_features(proj_feat.detach(), labels)

        # 4. Queue features
        q_feats, q_labels = self._get_queue()

        # 5. Build extra pool: aug + valid-queue entries
        parts_f, parts_l = [], []
        if aug_feats is not None:
            parts_f.append(aug_feats)
            parts_l.append(aug_labels)
        if q_feats is not None:
            valid_q = q_labels >= 0
            if valid_q.any():
                parts_f.append(q_feats[valid_q])
                parts_l.append(q_labels[valid_q])

        extra_f = torch.cat(parts_f, dim=0) if parts_f else None
        extra_l = torch.cat(parts_l, dim=0) if parts_l else None

        # 6. SupCon loss (with extended pool)
        if self.use_weighted_supcon:
            l_sup = self.supcon(
                proj_feat, labels,
                class_freq=self.class_freq,
                extra_feats=extra_f, extra_labels=extra_l,
            )
        else:
            l_sup = self.supcon(
                proj_feat, labels,
                extra_feats=extra_f, extra_labels=extra_l,
            )

        # 7. Proto loss (real features only)
        l_proto = self.proto(proj_feat, labels, temperature=self.temperature)

        # 8. Enqueue AFTER loss (MoCo convention)
        if self.queue_size > 0:
            self._enqueue(proj_feat.detach(), labels)

        # 9. Total
        total = l_ce + self.lambda_sup * l_sup + self.lambda_proto * l_proto

        info = {
            "l_ce":    l_ce.item(),
            "l_sup":   l_sup.item(),
            "l_proto": l_proto.item(),
        }
        if aug_feats is not None:
            info["n_aug"] = aug_feats.size(0)
        if q_feats is not None:
            info["q_size"] = self._qvalid

        return total, info
