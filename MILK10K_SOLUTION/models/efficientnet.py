"""
efficientnet.py — EfficientNet wrapper for MILK10k.

Default backbone: efficientnet_b3
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn


class EfficientNetModel(nn.Module):
    """
    EfficientNet single-branch model with optional metadata fusion.

    Forward signatures:
        model(image)
        model(image, metadata)
    """

    def __init__(
        self,
        model_name:     str   = "efficientnet_b3",
        num_classes:    int   = 11,
        pretrained:     bool  = True,
        meta_dim:       int   = 0,
        drop_rate:      float = 0.0,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.use_metadata = meta_dim > 0

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            # EfficientNet uses drop_path_rate differently; use global_pool
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        if self.use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 128),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 128),
                nn.SiLU(),
            )
            classifier_in = feat_dim + 128
        else:
            self.meta_mlp    = None
            classifier_in    = feat_dim

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_in),
            nn.Dropout(drop_rate if drop_rate > 0 else 0.3),
            nn.Linear(classifier_in, num_classes),
        )

    def forward(
        self,
        image:    torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.backbone(image)

        if self.use_metadata and metadata is not None:
            meta_feat = self.meta_mlp(metadata)
            feat      = torch.cat([feat, meta_feat], dim=1)

        return self.classifier(feat)
