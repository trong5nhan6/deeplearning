"""
dual_branch.py — Dual-branch model for MILK10k.

Architecture:
  clinical branch   → backbone (timm) → feature vector
  dermoscopy branch → backbone (timm) → feature vector
  concat(clinical_feat, derm_feat [, metadata_feat])
  → MLP head → 11 logits

When shared_weights=True, both branches share the same backbone weights.
When shared_weights=False, each branch has its own independent backbone.

Forward signature:
    model(clinical_image, derm_image)
    model(clinical_image, derm_image, metadata)
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn


class DualBranchModel(nn.Module):
    """
    Dual-branch image model with optional metadata fusion.

    Parameters
    ----------
    backbone_name   : timm model name used for both branches
    num_classes     : number of output logits (11 for MILK10k)
    pretrained      : load ImageNet pretrained weights
    meta_dim        : metadata feature dimension (0 = no metadata)
    drop_rate       : dropout rate
    drop_path_rate  : stochastic depth rate
    shared_weights  : if True, both branches share the same backbone
    """

    def __init__(
        self,
        backbone_name:  str   = "swin_base_patch4_window7_224",
        num_classes:    int   = 11,
        pretrained:     bool  = True,
        meta_dim:       int   = 0,
        drop_rate:      float = 0.0,
        drop_path_rate: float = 0.1,
        shared_weights: bool  = True,
    ):
        super().__init__()
        self.use_metadata   = meta_dim > 0
        self.shared_weights = shared_weights

        # ── Backbones ─────────────────────────────────────────────────────────
        self.clinical_branch = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.clinical_branch.num_features

        if shared_weights:
            # Share weights: dermoscopy branch IS the clinical branch
            self.derm_branch = self.clinical_branch
        else:
            # Independent backbone for dermoscopy
            self.derm_branch = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )

        # ── Metadata MLP ──────────────────────────────────────────────────────
        if self.use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 128),
                nn.GELU(),
            )
            meta_out_dim = 128
        else:
            self.meta_mlp = None
            meta_out_dim  = 0

        # ── Fusion & Classifier ───────────────────────────────────────────────
        # After concat: clinical_feat + derm_feat [+ meta_feat]
        fusion_dim = feat_dim * 2 + meta_out_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(drop_rate if drop_rate > 0 else 0.2),
        )

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(
        self,
        clinical_image: torch.Tensor,
        derm_image:     torch.Tensor,
        metadata:       Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Extract features from each branch
        clin_feat = self.clinical_branch(clinical_image)   # (B, feat_dim)
        derm_feat = self.derm_branch(derm_image)           # (B, feat_dim)

        # Concat image features
        parts = [clin_feat, derm_feat]

        # Optionally add metadata
        if self.use_metadata and metadata is not None:
            meta_feat = self.meta_mlp(metadata)            # (B, 128)
            parts.append(meta_feat)

        fused  = torch.cat(parts, dim=1)                  # (B, fusion_dim)
        fused  = self.fusion(fused)                        # (B, feat_dim)
        logits = self.classifier(fused)                    # (B, num_classes)
        return logits
