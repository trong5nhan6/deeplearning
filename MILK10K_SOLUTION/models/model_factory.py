"""
model_factory.py — Instantiate any MILK10k model from config.

Supported modes:
  single_image : one backbone + optional metadata MLP
  dual_image   : two branches + optional metadata MLP

Supported backbones (via timm):
  swin_base_patch4_window7_224
  convnext_base
  efficientnet_b3
  maxvit_tiny_tf_224
  vit_base_patch16_224
  hycnn_trans_xattnres        (HyCNN-Trans-XAttnRes dual-encoder)
  ... and any other timm model name
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.dual_branch import DualBranchModel
from models.efficientnet import EfficientNetModel
from models.convnext import ConvNextModel
from models.hycnn_trans_xattnres import HyCNNTransXAttnRes
from models.swin import SwinModel
from models.maxvit import MaxViTModel
from models.vit import ViTModel


# Map backbone name prefixes to wrapper classes
_BACKBONE_MAP = {
    "swin":          SwinModel,
    "convnext":      ConvNextModel,
    "efficientnet":  EfficientNetModel,
    "maxvit":        MaxViTModel,
    "vit":           ViTModel,
}

NUM_CLASSES = 11  # AKIEC BCC BEN_OTH BKL DF INF MAL_OTH MEL NV SCCKA VASC


def _get_single_model_class(model_name: str):
    """Return the appropriate wrapper class for a given timm model name."""
    name_lower = model_name.lower()
    for prefix, cls in _BACKBONE_MAP.items():
        if name_lower.startswith(prefix):
            return cls
    # Fallback: generic wrapper that works for any timm model
    return SwinModel  # SwinModel is the most generic wrapper


def build_model(
    cfg: Dict,
    meta_dim: int = 0,
) -> nn.Module:
    """
    Build a model from config dict.

    Parameters
    ----------
    cfg      : config dictionary (from YAML)
    meta_dim : dimension of metadata feature vector (0 = no metadata)

    Returns
    -------
    nn.Module ready for training or inference
    """
    model_name   = cfg.get("model_name", "swin_base_patch4_window7_224")
    mode         = cfg.get("mode", "single_image")
    pretrained   = cfg.get("pretrained", True)
    num_classes  = cfg.get("num_classes", NUM_CLASSES)
    use_metadata = cfg.get("use_metadata", False) and meta_dim > 0
    drop_rate    = cfg.get("drop_rate", 0.0)
    drop_path_rate = cfg.get("drop_path_rate", 0.1)

    effective_meta_dim = meta_dim if use_metadata else 0

    # ── HyCNN-Trans-XAttnRes ──────────────────────────────────────────────────
    if model_name == "hycnn_trans_xattnres":
        return HyCNNTransXAttnRes(
            num_classes    = num_classes,
            pretrained     = pretrained,
            attn_dim       = cfg.get("attn_dim", 768),
            num_heads      = cfg.get("num_heads", 8),
            window_size    = cfg.get("window_size", 8),
            mode           = mode,
            image_type     = cfg.get("image_type", "dermoscopy"),
            use_metadata   = use_metadata,
            meta_dim       = effective_meta_dim,
            meta_hidden    = cfg.get("meta_hidden", 256),
            dropout        = drop_rate,
            drop_path_rate = drop_path_rate,
        )

    # ── Dual-branch generic models ────────────────────────────────────────────
    if mode == "dual_image":
        model = DualBranchModel(
            backbone_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            meta_dim=effective_meta_dim,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            shared_weights=cfg.get("dual_shared_weights", True),
        )
    else:
        model_cls = _get_single_model_class(model_name)
        model     = model_cls(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            meta_dim=effective_meta_dim,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    return model
