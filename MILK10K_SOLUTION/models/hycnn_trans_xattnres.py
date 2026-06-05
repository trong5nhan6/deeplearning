"""
hycnn_trans_xattnres.py
=======================
HyCNN-Trans-XAttnRes:
  Hybrid CNN-Transformer with Cross-Stream Cross-Stage Attention Residuals.

Architecture flow:
  1. Dual encoder   : ConvNeXt-S (derm) + SwinV2-S (clinical) — independent forward
  2. Cross-attention: Bidirectional per stage (window for stage 0-1, full for stage 2-3)
  3. Aligner        : Pool + Conv1x1 → all stages to [B, 8, 8, 768]
  4. AttnRes        : Softmax over 4 stage candidates, query = tabular MLP or fixed w
  5. Classifier     : GAP → LayerNorm → Linear → 11 logits

References:
  - AttnRes      : Kimi Team (arXiv 2603.15031)
  - XAttnRes Seg : arXiv 2604.03297
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ═══════════════════════════════════════════════════════════════════
# 1. Primitives
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class MultiHeadCrossAttention(nn.Module):
    """Standard multi-head cross-attention (Q from one stream, K/V from other)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(dim, dim, bias=False)
        self.k_proj  = nn.Linear(dim, dim, bias=False)
        self.v_proj  = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,   # [B, N_q, C]
        k: torch.Tensor,   # [B, N_k, C]
        v: torch.Tensor,   # [B, N_k, C]
    ) -> torch.Tensor:
        B, N_q, C = q.shape
        N_k = k.shape[1]
        H   = self.num_heads
        D   = self.head_dim

        Q = self.q_proj(q).reshape(B, N_q, H, D).transpose(1, 2)  # [B, H, N_q, D]
        K = self.k_proj(k).reshape(B, N_k, H, D).transpose(1, 2)
        V = self.v_proj(v).reshape(B, N_k, H, D).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale              # [B, H, N_q, N_k]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N_q, C)        # [B, N_q, C]
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════
# 2. Window utilities
# ═══════════════════════════════════════════════════════════════════

def window_partition(x: torch.Tensor, ws: int) -> Tuple[torch.Tensor, int, int]:
    """
    Partition spatial map into non-overlapping windows.

    Args:
        x  : [B, H, W, C]
        ws : window size

    Returns:
        windows : [B * nH * nW, ws*ws, C]
        nH, nW  : number of windows along H, W
    """
    B, H, W, C = x.shape
    nH, nW = H // ws, W // ws
    x = x.reshape(B, nH, ws, nW, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # [B, nH, nW, ws, ws, C]
    x = x.reshape(B * nH * nW, ws * ws, C)
    return x, nH, nW


def window_unpartition(
    windows: torch.Tensor,
    ws: int,
    nH: int,
    nW: int,
    B: int,
) -> torch.Tensor:
    """
    Reverse of window_partition.

    Args:
        windows : [B * nH * nW, ws*ws, C]

    Returns:
        x : [B, H, W, C]
    """
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # [B, nH, ws, nW, ws, C]
    x = x.reshape(B, nH * ws, nW * ws, C)
    return x


# ═══════════════════════════════════════════════════════════════════
# 3. Cross-Attention Modules
# ═══════════════════════════════════════════════════════════════════

class WindowCrossAttention(nn.Module):
    """
    Window-based bidirectional cross-attention.
    Used for stage 0, 1 where N = 4096 / 1024 (large).

    Complexity: O(N * ws²) instead of O(N²) — linear in N.
    Motivation: at coarse stages, CNN and Swin features need to align
    local texture/color → local window interaction is sufficient.
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.ws = window_size

        self.norm_e = RMSNorm(dim)
        self.norm_t = RMSNorm(dim)

        # CNN queries Swin
        self.cross_e = MultiHeadCrossAttention(dim, num_heads, dropout)
        # Swin queries CNN
        self.cross_t = MultiHeadCrossAttention(dim, num_heads, dropout)

    def forward(self, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        e, t : [B, N, C]  (N = H*W, same spatial layout)
        returns f : [B, N, C]
        """
        B, N, C = e.shape
        H = W = int(N ** 0.5)
        ws = self.ws

        # Normalize before cross-attention (align CNN/Swin distributions)
        e_n = self.norm_e(e).reshape(B, H, W, C)
        t_n = self.norm_t(t).reshape(B, H, W, C)

        # Partition into windows
        e_win, nH, nW = window_partition(e_n, ws)   # [B*nW, ws², C]
        t_win, _,  _  = window_partition(t_n, ws)

        # Bidirectional cross-attention within each window
        e_out = e_win + self.cross_e(e_win, t_win, t_win)   # CNN ← Swin
        t_out = t_win + self.cross_t(t_win, e_win, e_win)   # Swin ← CNN

        # Unpartition and flatten
        e_out = window_unpartition(e_out, ws, nH, nW, B).reshape(B, N, C)
        t_out = window_unpartition(t_out, ws, nH, nW, B).reshape(B, N, C)

        # Merge: element-wise sum of two attended streams
        return e_out + t_out   # [B, N, C]


class FullCrossAttention(nn.Module):
    """
    Full bidirectional cross-attention.
    Used for stage 2, 3 where N = 256 / 64 (small).

    Motivation: at deep stages, tokens encode global semantics →
    long-range cross-modal interaction is needed.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_e = RMSNorm(dim)
        self.norm_t = RMSNorm(dim)
        self.cross_e = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.cross_t = MultiHeadCrossAttention(dim, num_heads, dropout)

    def forward(self, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        e, t : [B, N, C]
        returns f : [B, N, C]
        """
        e_n = self.norm_e(e)
        t_n = self.norm_t(t)

        e_out = e + self.cross_e(e_n, t_n, t_n)
        t_out = t + self.cross_t(t_n, e_n, e_n)

        return e_out + t_out


# ═══════════════════════════════════════════════════════════════════
# 4. Stage Aligner
# ═══════════════════════════════════════════════════════════════════

class StageAligner(nn.Module):
    """
    Align fused stage feature f_i to a common spatial size and channel dim.

    f_i [B, N_i, C_i]  →  [B, target_size, target_size, out_channels]

    Uses AdaptiveAvgPool2d (spatial) + Conv1x1 (channel) + BatchNorm.
    """

    def __init__(self, in_channels: int, out_channels: int, target_size: int = 8):
        super().__init__()
        self.target_size = target_size
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, N, C]   N = H * W
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        # [B, N, C] → [B, C, H, W]
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Spatial align
        if H != self.target_size:
            x = F.adaptive_avg_pool2d(x, self.target_size)   # [B, C, 8, 8]

        # Channel projection + norm
        x = self.norm(self.proj(x))                          # [B, out_ch, 8, 8]

        # [B, out_ch, 8, 8] → [B, 8, 8, out_ch]
        return x.permute(0, 2, 3, 1).contiguous()


# ═══════════════════════════════════════════════════════════════════
# 5. Tabular MLP (pseudo-query)
# ═══════════════════════════════════════════════════════════════════

class TabularMLP(nn.Module):
    """
    Project tabular metadata to a patient-specific pseudo-query vector.

    tabular [B, in_dim]  →  query [B, out_dim]

    The query conditions the AttnRes stage-selection on patient demographics
    and MONET features, making the aggregation personalized.
    """

    def __init__(
        self,
        in_dim:     int,
        out_dim:    int,
        hidden_dim: int   = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)   # [B, out_dim]


# ═══════════════════════════════════════════════════════════════════
# 6. Cross-Stage AttnRes
# ═══════════════════════════════════════════════════════════════════

class CrossStageAttnRes(nn.Module):
    """
    Attention Residuals applied over K aligned stage features.

    Replaces fixed weighted sum with softmax-attention over stage candidates,
    conditioned on a pseudo-query (fixed or patient-specific).

    Mechanism:
        V      = stack([f0_a, f1_a, f2_a, f3_a])   [K, B, H, W, d]
        logits = q · RMSNorm(V)                      [K, B, H, W]
        α      = softmax(logits, dim=0)              [K, B, H, W]
        out    = Σ_k  α_k · V_k                     [B, H, W, d]

    Each spatial location independently selects which stage's fused
    representation is most relevant for the given patient.
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.norm = RMSNorm(dim)

    def forward(
        self,
        stages: List[torch.Tensor],   # K × [B, H, W, d]
        query:  torch.Tensor,         # [B, d]
    ) -> torch.Tensor:

        # Stack candidates: [K, B, H, W, d]
        V = torch.stack(stages, dim=0)

        # Normalize value tensor
        V_norm = self.norm(V)   # [K, B, H, W, d]

        # Query: [B, d] → [1, B, 1, 1, d] for spatial broadcasting
        q = query.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # Logit per stage per spatial location: dot product
        logits = (V_norm * q).sum(dim=-1)       # [K, B, H, W]

        # Softmax over stage dimension (dim=0)
        alpha = F.softmax(logits, dim=0)        # [K, B, H, W]

        # Weighted sum
        alpha = alpha.unsqueeze(-1)             # [K, B, H, W, 1]
        out   = (alpha * V).sum(dim=0)          # [B, H, W, d]

        return out


# ═══════════════════════════════════════════════════════════════════
# 7. Main Model
# ═══════════════════════════════════════════════════════════════════

class HyCNNTransXAttnRes(nn.Module):
    """
    HyCNN-Trans-XAttnRes

    Parameters
    ----------
    num_classes     : output logits (11 for MILK10k)
    pretrained      : use pretrained backbone weights
    attn_dim        : unified dim for AttnRes and classifier (= last stage channels)
    num_heads       : number of attention heads in cross-attention
    window_size     : window size for window cross-attention (stage 0, 1)
    mode            : "single_image" | "dual_image"
                      single_image → one image feeds both branches
                      dual_image   → derm → ConvNeXt, clinical → SwinV2
    image_type      : "dermoscopy" | "clinical"  (only used when mode=single_image)
    use_metadata    : if True, tabular metadata conditions the pseudo-query in AttnRes
                      meta_dim must be provided (obtained from MetadataProcessor.meta_dim)
    meta_dim        : number of metadata features — computed by MetadataProcessor.fit(),
                      NOT hardcoded in config. Typical value: 28 (3 + 11 sites + 14 MONET)
    meta_hidden     : hidden dim of metadata MLP (architecture choice, can be in config)
    dropout         : dropout rate
    drop_path_rate  : stochastic depth for ConvNeXt backbone

    Forward signature (single_image)
    ---------------------------------
    model(image)                            # no metadata
    model(image, metadata=meta_tensor)      # with metadata

    Forward signature (dual_image)
    --------------------------------
    model(derm_image, clin_image)                          # no metadata
    model(derm_image, clin_image, metadata=meta_tensor)    # with metadata
    """

    # Channels at each stage for ConvNeXt-S / SwinV2-S (they match)
    _STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes:    int   = 11,
        pretrained:     bool  = True,
        attn_dim:       int   = 768,
        num_heads:      int   = 8,
        window_size:    int   = 8,
        mode:           str   = "dual_image",
        image_type:     str   = "dermoscopy",
        use_metadata:   bool  = False,
        meta_dim:       int   = 28,
        meta_hidden:    int   = 256,
        dropout:        float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        assert mode in ("single_image", "dual_image"), \
            f"mode must be 'single_image' or 'dual_image', got '{mode}'"
        assert image_type in ("dermoscopy", "clinical"), \
            f"image_type must be 'dermoscopy' or 'clinical', got '{image_type}'"
        if use_metadata:
            assert meta_dim > 0, "meta_dim must be set when use_metadata=True"

        self.mode         = mode
        self.image_type   = image_type
        self.use_metadata = use_metadata
        self.attn_dim     = attn_dim

        # ── Dual backbone (independent weights) ───────────────────────────────
        self.derm_encoder = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
            out_indices=(0, 1, 2, 3),
        )
        self.clin_encoder = timm.create_model(
            "swinv2_small_window8_256",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # ── Cross-attention per stage ──────────────────────────────────────────
        # Stage 0, 1 → Window (large N: 4096, 1024)
        # Stage 2, 3 → Full   (small N: 256, 64)
        self.cross_attns = nn.ModuleList([
            WindowCrossAttention(self._STAGE_CHANNELS[0], num_heads, window_size, dropout),
            WindowCrossAttention(self._STAGE_CHANNELS[1], num_heads, window_size, dropout),
            FullCrossAttention  (self._STAGE_CHANNELS[2], num_heads, dropout),
            FullCrossAttention  (self._STAGE_CHANNELS[3], num_heads, dropout),
        ])

        # ── Stage aligners ────────────────────────────────────────────────────
        self.aligners = nn.ModuleList([
            StageAligner(self._STAGE_CHANNELS[i], attn_dim, target_size=8)
            for i in range(4)
        ])

        # ── Pseudo-query: metadata MLP or fixed learnable vector ─────────────
        if use_metadata:
            # meta_dim đến từ MetadataProcessor.meta_dim (tính lúc runtime, không hardcode)
            self.meta_mlp = TabularMLP(meta_dim, attn_dim, meta_hidden, dropout)
        else:
            # Fixed pseudo-query (như paper gốc XAttnRes)
            self.pseudo_query = nn.Parameter(torch.randn(attn_dim) * 0.02)

        # ── AttnRes ───────────────────────────────────────────────────────────
        self.attn_res = CrossStageAttnRes(attn_dim)

        # ── Classifier head ───────────────────────────────────────────────────
        self.head_norm = nn.LayerNorm(attn_dim)
        self.head      = nn.Linear(attn_dim, num_classes)

        self._init_weights()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if not self.use_metadata:
            nn.init.trunc_normal_(self.pseudo_query, std=0.02)

    # ── Feature format normalisation ──────────────────────────────────────────

    @staticmethod
    def _to_bnc(feat: torch.Tensor) -> torch.Tensor:
        """
        Convert any backbone stage output to [B, N, C].

        ConvNeXt → BCHW  : permute + flatten
        SwinV2   → BHWC  : flatten HW
        """
        if feat.dim() == 4:
            if feat.shape[1] < feat.shape[-1]:
                # BHWC (SwinV2): channels last, C > H
                B, H, W, C = feat.shape
                return feat.reshape(B, H * W, C)
            else:
                # BCHW (ConvNeXt): channels first
                B, C, H, W = feat.shape
                return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        image_or_derm: torch.Tensor,
        clin_image:    Optional[torch.Tensor] = None,
        metadata:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        single_image mode:
            image_or_derm : [B, 3, H, W]  single image (derm or clinical)
            clin_image    : None
            metadata      : [B, meta_dim]  từ MetadataProcessor.transform() — optional

        dual_image mode:
            image_or_derm : [B, 3, H, W]  dermoscopy image  → ConvNeXt
            clin_image    : [B, 3, H, W]  clinical image    → SwinV2
            metadata      : [B, meta_dim]  từ MetadataProcessor.transform() — optional

        Returns
        -------
        logits : [B, num_classes]
        """
        B = image_or_derm.shape[0]

        # ── Resolve inputs theo mode ──────────────────────────────────────────
        if self.mode == "single_image":
            # Một ảnh đi vào cả hai branch
            derm_input = image_or_derm
            clin_input = image_or_derm
        else:
            # dual_image: derm → ConvNeXt, clinical → SwinV2
            assert clin_image is not None, \
                "dual_image mode requires both derm and clinical images"
            derm_input = image_or_derm
            clin_input = clin_image

        # ── Stage 1: Dual encoder forward (completely independent) ────────────
        derm_feats = self.derm_encoder(derm_input)   # list[4] of BCHW
        clin_feats = self.clin_encoder(clin_input)   # list[4] of BHWC

        # ── Stage 2: RMSNorm + Cross-attention per corresponding stage ────────
        fused = []
        for i in range(4):
            e = self._to_bnc(derm_feats[i])    # [B, N_i, C_i]
            t = self._to_bnc(clin_feats[i])    # [B, N_i, C_i]
            f_i = self.cross_attns[i](e, t)    # [B, N_i, C_i]
            fused.append(f_i)

        # ── Stage 3: Align all stages to [B, 8, 8, attn_dim] ─────────────────
        aligned = [self.aligners[i](fused[i]) for i in range(4)]

        # ── Stage 4: Pseudo-query ─────────────────────────────────────────────
        if self.use_metadata and metadata is not None:
            query = self.meta_mlp(metadata)                         # [B, attn_dim]
        else:
            query = self.pseudo_query.unsqueeze(0).expand(B, -1)    # [B, attn_dim]

        # ── Stage 5: Cross-stage AttnRes ──────────────────────────────────────
        out = self.attn_res(aligned, query)    # [B, 8, 8, attn_dim]

        # ── Stage 6: Classifier ───────────────────────────────────────────────
        out = out.mean(dim=(1, 2))             # GAP → [B, attn_dim]
        out = self.head_norm(out)
        logits = self.head(out)                # [B, num_classes]

        return logits


# ═══════════════════════════════════════════════════════════════════
# 8. Factory helper (dùng với model_factory.py)
# ═══════════════════════════════════════════════════════════════════

def build_hycnn_trans_xattnres(cfg, meta_processor=None) -> HyCNNTransXAttnRes:
    """
    Build model from a config dict / namespace / argparse.Namespace.

    Expected config keys (nhất quán với các model khác):
        num_classes, pretrained, attn_dim, num_heads, window_size,
        mode, image_type,
        use_metadata,   ← dùng use_metadata (KHÔNG phải use_tabular)
        meta_hidden,    ← optional, default 256
        dropout, drop_path_rate

    KHÔNG có tabular_dim trong config — được tính tự động từ meta_processor.meta_dim.

    Parameters
    ----------
    cfg            : config dict hoặc namespace
    meta_processor : MetadataProcessor instance (sau khi .fit() trên train set)
                     Nếu use_metadata=True thì bắt buộc phải có.
    """
    def _get(key, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    use_metadata = _get("use_metadata", False)

    # meta_dim lấy từ processor, không từ config
    meta_dim = 0
    if use_metadata:
        assert meta_processor is not None, \
            "use_metadata=True nhưng meta_processor chưa được truyền vào build_hycnn_trans_xattnres()"
        meta_dim = meta_processor.meta_dim

    return HyCNNTransXAttnRes(
        num_classes    = _get("num_classes",    11),
        pretrained     = _get("pretrained",     True),
        attn_dim       = _get("attn_dim",       768),
        num_heads      = _get("num_heads",      8),
        window_size    = _get("window_size",    8),
        mode           = _get("mode",           "dual_image"),
        image_type     = _get("image_type",     "dermoscopy"),
        use_metadata   = use_metadata,
        meta_dim       = meta_dim,
        meta_hidden    = _get("meta_hidden",    256),
        dropout        = _get("drop_rate",      0.1),
        drop_path_rate = _get("drop_path_rate", 0.1),
    )


# ═══════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2
    derm = torch.randn(B, 3, 256, 256).to(device)
    clin = torch.randn(B, 3, 256, 256).to(device)
    # meta_dim=28: 3 (age/sex/tone) + 11 (site one-hot) + 14 (7 MONET × 2 views)
    # — giá trị này đến từ MetadataProcessor.meta_dim sau khi .fit(), không hardcode
    META_DIM = 28
    meta = torch.randn(B, META_DIM).to(device)

    # ── Test 1: dual_image, no metadata ──────────────────────────────────────
    print("=" * 60)
    print("Test 1: mode=dual_image, use_metadata=False")
    m = HyCNNTransXAttnRes(num_classes=11, pretrained=False,
                           mode="dual_image", use_metadata=False).to(device)
    out = m(derm, clin)
    assert out.shape == (B, 11), f"Expected (2,11), got {out.shape}"
    print(f"  Output: {out.shape}  ✓")

    # ── Test 2: dual_image, with metadata ────────────────────────────────────
    print("Test 2: mode=dual_image, use_metadata=True")
    m = HyCNNTransXAttnRes(num_classes=11, pretrained=False,
                           mode="dual_image", use_metadata=True,
                           meta_dim=META_DIM).to(device)
    out = m(derm, clin, metadata=meta)
    assert out.shape == (B, 11), f"Expected (2,11), got {out.shape}"
    print(f"  Output: {out.shape}  ✓")

    # ── Test 3: single_image (dermoscopy), no metadata ────────────────────────
    print("Test 3: mode=single_image, image_type=dermoscopy, use_metadata=False")
    m = HyCNNTransXAttnRes(num_classes=11, pretrained=False,
                           mode="single_image", image_type="dermoscopy",
                           use_metadata=False).to(device)
    out = m(derm)
    assert out.shape == (B, 11), f"Expected (2,11), got {out.shape}"
    print(f"  Output: {out.shape}  ✓")

    # ── Test 4: single_image (clinical), with metadata ────────────────────────
    print("Test 4: mode=single_image, image_type=clinical, use_metadata=True")
    m = HyCNNTransXAttnRes(num_classes=11, pretrained=False,
                           mode="single_image", image_type="clinical",
                           use_metadata=True, meta_dim=META_DIM).to(device)
    out = m(clin, metadata=meta)
    assert out.shape == (B, 11), f"Expected (2,11), got {out.shape}"
    print(f"  Output: {out.shape}  ✓")

    # ── Param count ───────────────────────────────────────────────────────────
    print("=" * 60)
    total = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  Total params: {total:.1f}M")
    print("All tests passed!")
