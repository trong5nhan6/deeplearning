"""
swin.py — Swin Transformer wrapper for MILK10k.

Default backbone: swin_base_patch4_window7_224
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn


class SwinModel(nn.Module):
    """
    Swin Transformer single-branch model.
    Optionally fuses metadata via a small MLP before the final classifier.

    Forward signatures:
        single image only : model(image)
        with metadata     : model(image, metadata)
    """

    def __init__(
        self,
        model_name:    str   = "swin_base_patch4_window7_224",
        num_classes:   int   = 11,
        pretrained:    bool  = True,
        meta_dim:      int   = 0,
        drop_rate:     float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.use_metadata = meta_dim > 0

        # Load backbone — output features (no head)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove original head
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features

        # Optional metadata MLP
        if self.use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 128),
                nn.GELU(),
            )
            classifier_in = feat_dim + 128
        else:
            self.meta_mlp    = None
            classifier_in    = feat_dim

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Dropout(drop_rate if drop_rate > 0 else 0.2),
            nn.Linear(classifier_in, num_classes),
        )

    def forward(
        self,
        image:    torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.backbone(image)             # (B, feat_dim)

        if self.use_metadata and metadata is not None:
            meta_feat = self.meta_mlp(metadata) # (B, 128)
            feat      = torch.cat([feat, meta_feat], dim=1)

        return self.classifier(feat)            # (B, num_classes)
