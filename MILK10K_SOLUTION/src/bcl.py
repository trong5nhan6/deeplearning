"""
BCL -- Balanced Contrastive Learning (Zhu et al. CVPR 2022)
=============================================================
L_total = L_CE + lambda_sup * L_SupCon + lambda_proto * L_proto

Motivation for MILK10k (IR = 280x, MAL_OTH = 18 samples):
- L_CE alone cannot overcome extreme class imbalance
- L_SupCon: pulls same-class features together, pushes other classes apart
  -> better feature discriminability in embedding space
- L_proto: pulls each feature toward its class prototype (running EMA centroid)
  -> always provides gradient for minority classes via their accumulated prototype,
     even when only 1 minority sample appears in a batch

Supports two SupCon variants:
  use_weighted_supcon=False -> standard SupConLoss    (equal weight per anchor)
  use_weighted_supcon=True  -> WeightedSupConLoss     (anchor weight = 1/pi_k)
    - minority anchors (MAL_OTH) get up to supcon_weight_clamp x more weight
    - weight_clamp=20 prevents a single bad sample from exploding the gradient
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP that projects backbone features into a contrastive embedding space.
    Output is L2-normalized -> lives on the unit hypersphere.

    Architecture:
        Linear(in_dim -> hidden_dim) -> BN -> ReLU -> Linear(hidden_dim -> out_dim) -> L2-norm

    BatchNorm stabilises training; L2-norm ensures cosine similarity == dot product.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)  # [B, out_dim] on unit sphere


# ---------------------------------------------------------------------------
# 2. SupConLoss (standard, equal weight)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss -- Khosla et al. NeurIPS 2020.

    For each anchor i:
      L_i = -1/|P(i)| * sum_{p in P(i)} log [
                exp(z_i . z_p / tau) / sum_{a != i} exp(z_i . z_a / tau)
             ]
    where P(i) = set of same-class samples in the batch (excluding i itself).

    Notes:
    - If a sample has no positive pair (only one sample of its class in batch),
      it contributes 0 loss (skipped gracefully).
    - Temperature tau < 0.1 -> sharper boundaries; tau=0.07 is the original default.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : [B, D] -- L2-normalized projected features
        labels   : [B,]   -- class indices (long tensor)
        Returns  : scalar loss
        """
        B, _ = features.shape
        device = features.device

        sim = torch.matmul(features, features.T) / self.temperature   # [B, B]

        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        labels_col = labels.unsqueeze(1)
        mask_pos   = (labels_col == labels_col.T) & ~mask_self         # [B, B]

        if mask_pos.sum() == 0:
            return features.sum() * 0.0

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()                                    # stability

        exp_sim = torch.exp(sim) * (~mask_self).float()
        log_sum = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        log_prob = sim - log_sum                                        # [B, B]

        n_pos = mask_pos.float().sum(dim=1).clamp(min=1)
        loss_per_anchor = -(mask_pos.float() * log_prob).sum(dim=1) / n_pos  # [B]

        has_pos = mask_pos.any(dim=1)
        return loss_per_anchor[has_pos].mean()


# ---------------------------------------------------------------------------
# 3. WeightedSupConLoss (minority-amplified)
# ---------------------------------------------------------------------------

class WeightedSupConLoss(nn.Module):
    """
    Weighted Supervised Contrastive Loss.

    Differs from standard SupConLoss in that each anchor gets a weight
    proportional to the inverse class frequency:

        w_i = 1 / pi_{c_i}     (pi_k = n_k / N)

    Example (MILK10k):
        MAL_OTH (pi=0.0017) -> raw w ~ 588
        BCC     (pi=0.48)   -> raw w ~ 2.1

    Weights are normalised so mean(w) = 1 over the batch, then clamped:
        w_i = clamp(w_i / mean(w), max=weight_clamp)

    With weight_clamp=20.0:
        MAL_OTH anchor contributes at most 20x more than a BCC anchor.

    Loss:
        L = sum_i(w_i * L_i) / sum_i(w_i)   (only anchors with >= 1 positive)

    If class_freq=None -> falls back to standard (unweighted) SupConLoss.

    Parameters
    ----------
    temperature  : tau for cosine similarity scaling (default 0.07)
    weight_clamp : max normalised anchor weight (default 20.0).
                   Set to 0 to disable clamping.
                   Prevents a bad minority outlier from dominating the step.
    """

    def __init__(self, temperature: float = 0.07, weight_clamp: float = 3.0):
        super().__init__()
        self.temperature  = temperature
        self.weight_clamp = weight_clamp

    def forward(
        self,
        features:   torch.Tensor,
        labels:     torch.Tensor,
        class_freq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        features   : [B, D]  L2-normalized projected features
        labels     : [B,]    long class indices
        class_freq : [C,]    pi_k = n_k/N per class (optional)
        """
        B, _ = features.shape
        device = features.device

        # Pairwise cosine similarity
        sim = torch.matmul(features, features.T) / self.temperature   # [B, B]

        mask_self  = torch.eye(B, dtype=torch.bool, device=device)
        labels_col = labels.unsqueeze(1)
        mask_pos   = (labels_col == labels_col.T) & ~mask_self         # [B, B]

        if mask_pos.sum() == 0:
            return features.sum() * 0.0

        # Per-anchor SupCon loss (same maths as SupConLoss)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim  = torch.exp(sim) * (~mask_self).float()
        log_sum  = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        log_prob = sim - log_sum                                        # [B, B]

        n_pos           = mask_pos.float().sum(dim=1).clamp(min=1)
        loss_per_anchor = -(mask_pos.float() * log_prob).sum(dim=1) / n_pos  # [B]

        has_pos = mask_pos.any(dim=1)
        if has_pos.sum() == 0:
            return features.sum() * 0.0

        # Unweighted fallback
        if class_freq is None:
            return loss_per_anchor[has_pos].mean()

        # Inverse-frequency weights, normalised to mean=1
        weights = 1.0 / class_freq[labels].clamp(min=1e-9)            # [B]
        weights = weights / weights.mean()                             # mean -> 1
        weights = torch.clamp(weights, max=self.weight_clamp)         # cap extremes

        w    = weights[has_pos]
        loss = loss_per_anchor[has_pos]
        return (w * loss).sum() / w.sum()


# ---------------------------------------------------------------------------
# 4. PrototypeLoss
# ---------------------------------------------------------------------------

class PrototypeLoss(nn.Module):
    """
    Prototype Loss for BCL.

    Maintains an EMA prototype (running centroid) per class.
    Loss = CE( sim(z, prototypes) / tau,  y )

    Key advantage over SupConLoss: once a prototype for MAL_OTH is initialised
    from early batches, every subsequent sample gets gradient pulling it toward
    the MAL_OTH centroid -- even if only 1 MAL_OTH sample appears in the batch.

    Parameters
    ----------
    num_classes : number of classes (C)
    feat_dim    : projection dimension (D)
    momentum    : EMA coefficient for prototype update (default 0.9)
    """

    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum    = momentum
        self.num_classes = num_classes
        self.register_buffer("prototypes",  torch.zeros(num_classes, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """EMA update. Call with detached features after each forward pass."""
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

    def forward(
        self,
        features:    torch.Tensor,
        labels:      torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Pull each feature toward its class prototype via CE on similarities.
        Only classes with an initialised prototype contribute.
        """
        valid = self.initialized
        if valid.sum() < 2:
            return features.sum() * 0.0

        protos        = F.normalize(self.prototypes[valid], dim=1)    # [C', D]
        valid_classes = torch.where(valid)[0]                          # [C']

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

        sim = torch.matmul(feat_v, protos.T) / temperature
        return F.cross_entropy(sim, lbl_v)


# ---------------------------------------------------------------------------
# 5. BCLLoss -- combined loss
# ---------------------------------------------------------------------------

class BCLLoss(nn.Module):
    """
    Balanced Contrastive Learning Loss (Zhu et al. CVPR 2022).

    L_total = L_CE + lambda_sup * L_SupCon + lambda_proto * L_proto

    Usage
    -----
        bcl = BCLLoss(num_classes=11, feat_dim=128, ce_criterion=my_ce_loss)
        loss, info = bcl(logits, proj_feat, targets_onehot)

    Parameters
    ----------
    num_classes          : number of output classes
    feat_dim             : projection head output dimension
    temperature          : tau for contrastive similarity
    lambda_sup           : weight for SupCon component
    lambda_proto         : weight for prototype component
    proto_momentum       : EMA momentum for prototype update
    ce_criterion         : external CE/LA/focal loss; None -> l_ce = 0
    use_weighted_supcon  : if True, use WeightedSupConLoss (w_i = 1/pi_k)
    class_freq           : [C,] pi_k tensor required when use_weighted_supcon=True
    supcon_weight_clamp  : max anchor weight in WeightedSupConLoss (default 20.0)
    """

    def __init__(
        self,
        num_classes:         int,
        feat_dim:            int   = 128,
        temperature:         float = 0.07,
        lambda_sup:          float = 0.1,
        lambda_proto:        float = 0.1,
        proto_momentum:      float = 0.9,
        ce_criterion:        nn.Module | None = None,
        use_weighted_supcon: bool  = False,
        class_freq:          torch.Tensor | None = None,
        supcon_weight_clamp: float = 20.0,
    ):
        super().__init__()
        self.lambda_sup          = lambda_sup
        self.lambda_proto        = lambda_proto
        self.temperature         = temperature
        self.use_weighted_supcon = use_weighted_supcon

        if use_weighted_supcon:
            self.supcon = WeightedSupConLoss(
                temperature=temperature,
                weight_clamp=supcon_weight_clamp,
            )
        else:
            self.supcon = SupConLoss(temperature=temperature)

        self.proto = PrototypeLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=proto_momentum,
        )
        self.ce = ce_criterion

        if class_freq is not None:
            self.register_buffer("class_freq", class_freq)
        else:
            self.class_freq = None

    def forward(
        self,
        logits:    torch.Tensor,
        proj_feat: torch.Tensor,
        targets:   torch.Tensor,
    ):
        """
        logits    : [B, C]  raw classification logits
        proj_feat : [B, D]  L2-normalized projected features
        targets   : [B, C]  one-hot  OR  [B,] long class indices

        Returns
        -------
        total_loss : scalar
        info       : dict with per-component loss values (for logging)
        """
        labels = targets.argmax(dim=1) if targets.dim() == 2 else targets.long()

        # Classification loss
        l_ce = (
            self.ce(logits, targets)
            if self.ce is not None
            else torch.tensor(0.0, device=logits.device)
        )

        # Supervised contrastive loss
        if self.use_weighted_supcon:
            l_sup = self.supcon(proj_feat, labels, class_freq=self.class_freq)
        else:
            l_sup = self.supcon(proj_feat, labels)

        # Prototype loss + EMA update
        self.proto.update(proj_feat.detach(), labels)
        l_proto = self.proto(proj_feat, labels, temperature=self.temperature)

        total = l_ce + self.lambda_sup * l_sup + self.lambda_proto * l_proto

        return total, {
            "l_ce":    l_ce.item(),
            "l_sup":   l_sup.item(),
            "l_proto": l_proto.item(),
        }
