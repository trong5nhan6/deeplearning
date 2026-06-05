"""
hycnn_trans_softmoe.py
======================
HyCNN-Trans-SoftMoE:
  Hybrid CNN-Transformer with Cross-Stream Cross-Attention + Soft Mixture-of-Experts.

Architecture flow:
  1. Dual encoder   : ConvNeXt-S (derm) + SwinV2-S (clinical) — independent forward
  2. Cross-attention: Bidirectional per stage, concat → Linear → f_i
                      Window for stage 0-1 (N=4096, 1024), Full for stage 2-3 (N=256, 64)
  3. GAP + Project  : f_i → GAP → Linear → [B, D] × 4 stage vectors
  4. SoftMoE        : [B, 5, D] sequence (4 stage tokens + 1 tabular token)
                      → n_experts expert FFNs via soft dispatch
                      → mean pool → [B, D]
  5. Classifier     : LayerNorm → Linear → 11 logits

Key differences from HyCNN-Trans-XAttnRes:
  - CrossAttn output: concat(e', t') → Linear  (vs. element-wise sum)
  - Stage aggregator: SoftMoE (vs. AttnRes weighted sum)
  - Tabular: appended as 5th token in sequence (vs. pseudo-query vector)
  - No StageAligner spatial pooling — each stage goes GAP immediately

References:
  - SoftMoE    : Puigcerver et al., "From Sparse to Soft Mixtures of Experts" (2023)
  - AttnRes    : Kimi Team (arXiv 2603.15031)
  - XAttnRes   : arXiv 2604.03297
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ═══════════════════════════════════════════════════════════════════
# 1. Primitives  (shared with hycnn_trans_xattnres)
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

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
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N_q, C = q.shape
        N_k = k.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(q).reshape(B, N_q, H, D).transpose(1, 2)
        K = self.k_proj(k).reshape(B, N_k, H, D).transpose(1, 2)
        V = self.v_proj(v).reshape(B, N_k, H, D).transpose(1, 2)

        attn = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ V).transpose(1, 2).reshape(B, N_q, C)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════
# 2. Window utilities
# ═══════════════════════════════════════════════════════════════════

def window_partition(x: torch.Tensor, ws: int) -> Tuple[torch.Tensor, int, int]:
    """[B, H, W, C] → [B*nH*nW, ws², C]"""
    B, H, W, C = x.shape
    nH, nW = H // ws, W // ws
    x = x.reshape(B, nH, ws, nW, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(B * nH * nW, ws * ws, C), nH, nW


def window_unpartition(windows: torch.Tensor, ws: int, nH: int, nW: int, B: int) -> torch.Tensor:
    """[B*nH*nW, ws², C] → [B, H, W, C]"""
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(B, nH * ws, nW * ws, C)


# ═══════════════════════════════════════════════════════════════════
# 3. Cross-Attention Modules  (concat→Linear fusion)
# ═══════════════════════════════════════════════════════════════════

class WindowCrossAttentionFuse(nn.Module):
    """
    Window-based bidirectional cross-attention.
    Output: concat(e', t') → Linear → f_i   [B, N, C]

    Stage 0 (N=4096), Stage 1 (N=1024) — window size 8×8 reduces complexity 64×.
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.ws = window_size

        self.norm_e  = RMSNorm(dim)
        self.norm_t  = RMSNorm(dim)
        self.cross_e = MultiHeadCrossAttention(dim, num_heads, dropout)   # CNN ← Swin
        self.cross_t = MultiHeadCrossAttention(dim, num_heads, dropout)   # Swin ← CNN

        # Fuse: concat [B, N, 2*C] → [B, N, C]
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim, bias=False),
            RMSNorm(dim),
        )

    def forward(self, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """e, t: [B, N, C]  →  f: [B, N, C]"""
        B, N, C = e.shape
        H = W = int(N ** 0.5)

        e_n = self.norm_e(e).reshape(B, H, W, C)
        t_n = self.norm_t(t).reshape(B, H, W, C)

        e_win, nH, nW = window_partition(e_n, self.ws)
        t_win, _,  _  = window_partition(t_n, self.ws)

        e_out = e_win + self.cross_e(e_win, t_win, t_win)   # [Bw, ws², C]
        t_out = t_win + self.cross_t(t_win, e_win, e_win)

        e_out = window_unpartition(e_out, self.ws, nH, nW, B).reshape(B, N, C)
        t_out = window_unpartition(t_out, self.ws, nH, nW, B).reshape(B, N, C)

        # Concat + project → single fused feature
        return self.fuse(torch.cat([e_out, t_out], dim=-1))   # [B, N, C]


class FullCrossAttentionFuse(nn.Module):
    """
    Full bidirectional cross-attention.
    Output: concat(e', t') → Linear → f_i   [B, N, C]

    Stage 2 (N=256), Stage 3 (N=64) — full attention feasible.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_e  = RMSNorm(dim)
        self.norm_t  = RMSNorm(dim)
        self.cross_e = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.cross_t = MultiHeadCrossAttention(dim, num_heads, dropout)

        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim, bias=False),
            RMSNorm(dim),
        )

    def forward(self, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """e, t: [B, N, C]  →  f: [B, N, C]"""
        e_n, t_n = self.norm_e(e), self.norm_t(t)
        e_out = e + self.cross_e(e_n, t_n, t_n)
        t_out = t + self.cross_t(t_n, e_n, e_n)
        return self.fuse(torch.cat([e_out, t_out], dim=-1))   # [B, N, C]


# ═══════════════════════════════════════════════════════════════════
# 4. GAP Stage Projector
# ═══════════════════════════════════════════════════════════════════

class StageProjector(nn.Module):
    """
    f_i [B, N_i, C_i]  →  Global Average Pool  →  Linear  →  [B, out_dim]

    Each fused stage feature is collapsed to a single vector before SoftMoE.
    This is appropriate for classification where spatial layout is secondary
    to the overall feature distribution of the fused representation.
    """

    def __init__(self, in_channels: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C]  →  [B, out_dim]"""
        x = x.mean(dim=1)        # GAP: [B, C]
        return self.proj(x)      # [B, out_dim]


# ═══════════════════════════════════════════════════════════════════
# 5. Tabular MLP  → token
# ═══════════════════════════════════════════════════════════════════

class TabularTokenMLP(nn.Module):
    """
    Project tabular metadata to a token of dim D,
    which is appended as the 5th token in the SoftMoE sequence.

    metadata [B, meta_dim]  →  token [B, D]
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)   # [B, out_dim]


# ═══════════════════════════════════════════════════════════════════
# 6. Soft Mixture-of-Experts
# ═══════════════════════════════════════════════════════════════════

class SoftMoE(nn.Module):
    """
    Soft Mixture-of-Experts — full paper formulation with slots.

    Paper: Puigcerver et al. (2023) "From Sparse to Soft Mixtures of Experts"

    Formulation (E experts, p slots each → E*p total slots):
      M      ∈ [D, E*p]                dispatch/combine matrix (learned)
      Φ(X)   = norm(X) @ M             logits          [B, S, E*p]
      D_mat  = softmax(Φ, dim=1)       dispatch        [B, S, E*p]  token→slot
      C_mat  = softmax(Φ, dim=2)       combine         [B, S, E*p]  slot→token
      X̃_j    = Σ_s D_mat[s,j] * X[s]  per-slot input  [B, E*p, D]
      Ỹ_j    = FFN_k(X̃_j)             expert k owns slots [k*p : (k+1)*p]
      Y[s]   = Σ_j C_mat[s,j] * Ỹ[j]  combine back    [B, S, D]
      out    = x + Y                   residual

    With p=1 (n_slots=1) this reduces to the simplified 1-slot-per-expert version.
    With p>1 each expert gets richer input — more expressive at slight extra cost.

    Parameters
    ----------
    dim       : token dimension D
    n_experts : number of expert FFNs   E  (default 4)
    n_slots   : slots per expert        p  (default 1, paper default)
    expand    : FFN hidden expansion ratio (default 4)
    dropout   : dropout inside expert FFN
    """

    def __init__(
        self,
        dim:      int,
        n_experts: int  = 4,
        n_slots:   int  = 1,
        expand:    int  = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_slots   = n_slots
        self.total_slots = n_experts * n_slots   # E * p

        # Dispatch / combine matrix M: [D, E*p]
        self.M = nn.Parameter(torch.empty(dim, self.total_slots))
        nn.init.trunc_normal_(self.M, std=0.02)

        # Expert FFNs — expert k processes slots [k*p : (k+1)*p]
        hidden = dim * expand
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, S, D]
        out : [B, S, D]
        """
        B, S, D = x.shape
        E, P, EP = self.n_experts, self.n_slots, self.total_slots

        x_norm = self.norm(x)                              # [B, S, D]

        # Dispatch logits over all slots: [B, S, E*p]
        logits = x_norm @ self.M                           # [B, S, EP]

        # Dispatch: softmax over tokens (dim=1) → how much each token contributes to each slot
        D_mat = F.softmax(logits, dim=1)                   # [B, S, EP]
        # Combine: softmax over slots (dim=2) → how much each slot contributes back to each token
        C_mat = F.softmax(logits, dim=2)                   # [B, S, EP]

        # Per-slot input: X̃[j] = Σ_s D_mat[s,j] * x[s]  →  [B, EP, D]
        X_slots = torch.einsum('bse,bsd->bed', D_mat, x_norm)   # [B, EP, D]

        # Apply expert FFNs: expert k owns slots [k*P : (k+1)*P]
        Y_slots = torch.zeros_like(X_slots)                # [B, EP, D]
        for k in range(E):
            slot_start = k * P
            slot_end   = slot_start + P
            # X_slots[:, slot_start:slot_end, :] → [B, P, D]
            # flatten → [B*P, D] for FFN, then reshape back
            x_k = X_slots[:, slot_start:slot_end, :].reshape(B * P, D)
            y_k = self.experts[k](x_k).reshape(B, P, D)
            Y_slots[:, slot_start:slot_end, :] = y_k

        # Combine back: Y[s] = Σ_j C_mat[s,j] * Ỹ[j]  →  [B, S, D]
        Y = torch.einsum('bse,bed->bsd', C_mat, Y_slots)   # [B, S, D]

        # Residual
        return x + Y


# ═══════════════════════════════════════════════════════════════════
# 7. Main Model
# ═══════════════════════════════════════════════════════════════════

class HyCNNTransSoftMoE(nn.Module):
    """
    HyCNN-Trans-SoftMoE

    Parameters
    ----------
    num_classes     : output logits (11 for MILK10k)
    pretrained      : use pretrained backbone weights
    embed_dim       : unified token dimension D (default 768)
    num_heads       : attention heads in cross-attention
    window_size     : window size for window cross-attention (stage 0, 1)
    n_experts       : number of SoftMoE experts (default 4)
    n_slots         : slots per expert — default 1 (paper default, p=1)
    moe_expand      : FFN expansion in each expert (default 4)
    mode            : "single_image" | "dual_image"
    image_type      : "dermoscopy" | "clinical"  (only for single_image mode)
    use_metadata    : True → tabular metadata appended as 5th SoftMoE token
    meta_dim        : metadata feature count — from MetadataProcessor.meta_dim
    meta_hidden     : hidden dim of metadata MLP
    dropout         : dropout rate
    drop_path_rate  : stochastic depth for ConvNeXt backbone

    Forward signature (dual_image mode)
    ------------------------------------
    model(clin_image, derm_image)                       # no metadata
    model(clin_image, derm_image, metadata=meta_tensor) # with metadata

    Forward signature (single_image mode)
    ---------------------------------------
    model(image)                               # no metadata
    model(image, metadata=meta_tensor)         # with metadata
    """

    _STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes:    int   = 11,
        pretrained:     bool  = True,
        embed_dim:      int   = 768,
        num_heads:      int   = 8,
        window_size:    int   = 8,
        n_experts:      int   = 4,
        n_slots:        int   = 1,
        moe_expand:     int   = 4,
        mode:           str   = "dual_image",
        image_type:     str   = "dermoscopy",
        use_metadata:   bool  = False,
        meta_dim:       int   = 28,
        meta_hidden:    int   = 256,
        dropout:        float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        assert mode in ("single_image", "dual_image")
        assert image_type in ("dermoscopy", "clinical")
        if use_metadata:
            assert meta_dim > 0, "meta_dim must be > 0 when use_metadata=True"

        self.mode         = mode
        self.image_type   = image_type
        self.use_metadata = use_metadata
        self.embed_dim    = embed_dim

        # ── Dual backbone ────────────────────────────────────────────────────────
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

        # ── Cross-attention per stage (bidirectional, concat→Linear fusion) ─────
        C = self._STAGE_CHANNELS
        self.cross_attns = nn.ModuleList([
            WindowCrossAttentionFuse(C[0], num_heads, window_size, dropout),
            WindowCrossAttentionFuse(C[1], num_heads, window_size, dropout),
            FullCrossAttentionFuse  (C[2], num_heads, dropout),
            FullCrossAttentionFuse  (C[3], num_heads, dropout),
        ])

        # ── GAP + Project: f_i → [B, embed_dim] ─────────────────────────────────
        self.stage_proj = nn.ModuleList([
            StageProjector(C[i], embed_dim, dropout)
            for i in range(4)
        ])

        # ── Tabular token (5th SoftMoE token) ───────────────────────────────────
        if use_metadata:
            self.tab_mlp = TabularTokenMLP(meta_dim, embed_dim, meta_hidden, dropout)
        else:
            # Learnable fixed token (like CLS token — no patient info)
            self.tab_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # ── SoftMoE ─────────────────────────────────────────────────────────────
        self.moe = SoftMoE(embed_dim, n_experts=n_experts, n_slots=n_slots,
                           expand=moe_expand, dropout=dropout)

        # ── Classifier head ──────────────────────────────────────────────────────
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if not self.use_metadata:
            nn.init.trunc_normal_(self.tab_token, std=0.02)

    # ── Feature format normalisation ─────────────────────────────────────────────

    @staticmethod
    def _to_bnc(feat: torch.Tensor) -> torch.Tensor:
        """
        ConvNeXt → BCHW  →  [B, N, C]
        SwinV2   → BHWC  →  [B, N, C]
        """
        if feat.dim() == 4:
            if feat.shape[1] < feat.shape[-1]:
                # BHWC (SwinV2)
                B, H, W, C = feat.shape
                return feat.reshape(B, H * W, C)
            else:
                # BCHW (ConvNeXt)
                B, C, H, W = feat.shape
                return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    # ── Forward ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        image_or_clin: torch.Tensor,
        derm_image:    Optional[torch.Tensor] = None,
        metadata:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        dual_image mode:
            image_or_clin : [B, 3, H, W]  clinical  → SwinV2-S
            derm_image    : [B, 3, H, W]  derm      → ConvNeXt-S
            metadata      : [B, meta_dim] optional

        single_image mode:
            image_or_clin : [B, 3, H, W]  single image → both encoders
            derm_image    : None
            metadata      : [B, meta_dim] optional

        Returns
        -------
        logits : [B, num_classes]
        """
        B = image_or_clin.shape[0]

        # ── Resolve inputs ────────────────────────────────────────────────────────
        if self.mode == "single_image":
            clin_input = image_or_clin
            derm_input = image_or_clin
        else:
            assert derm_image is not None, "dual_image mode requires derm_image"
            clin_input = image_or_clin
            derm_input = derm_image

        # ── Step 1: Dual encoder ─────────────────────────────────────────────────
        derm_feats = self.derm_encoder(derm_input)   # list[4] BCHW
        clin_feats = self.clin_encoder(clin_input)   # list[4] BHWC

        # ── Step 2: CrossAttn per stage  →  fused f_i [B, N_i, C_i] ────────────
        fused = []
        for i in range(4):
            e   = self._to_bnc(derm_feats[i])        # [B, N_i, C_i]
            t   = self._to_bnc(clin_feats[i])
            f_i = self.cross_attns[i](e, t)           # [B, N_i, C_i]
            fused.append(f_i)

        # ── Step 3: GAP + Project  →  stage tokens [B, embed_dim] × 4 ──────────
        stage_tokens = [self.stage_proj[i](fused[i]) for i in range(4)]
        # Stack: [B, 4, embed_dim]
        seq = torch.stack(stage_tokens, dim=1)

        # ── Step 4: Build tabular token and append ───────────────────────────────
        if self.use_metadata and metadata is not None:
            tab = self.tab_mlp(metadata).unsqueeze(1)    # [B, 1, embed_dim]
        else:
            tab = self.tab_token.expand(B, -1, -1)       # [B, 1, embed_dim]

        # Final sequence: [B, 5, embed_dim]  (4 stage tokens + 1 tabular token)
        seq = torch.cat([seq, tab], dim=1)

        # ── Step 5: SoftMoE ──────────────────────────────────────────────────────
        seq = self.moe(seq)                              # [B, 5, embed_dim]

        # ── Step 6: Mean pool + classify ─────────────────────────────────────────
        out    = seq.mean(dim=1)                         # [B, embed_dim]
        out    = self.head_norm(out)
        logits = self.head(out)                          # [B, num_classes]

        return logits


# ═══════════════════════════════════════════════════════════════════
# 8. Factory helper
# ═══════════════════════════════════════════════════════════════════

def build_hycnn_trans_softmoe(cfg, meta_processor=None) -> HyCNNTransSoftMoE:
    """
    Build HyCNNTransSoftMoE from config dict / namespace.

    Config keys:
        num_classes, pretrained, embed_dim, num_heads, window_size,
        n_experts, moe_expand,
        mode, image_type,
        use_metadata,
        meta_hidden,
        drop_rate, drop_path_rate

    meta_dim comes from meta_processor.meta_dim — NOT from config.
    """
    def _get(key, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    use_metadata = _get("use_metadata", False)
    meta_dim     = 0
    if use_metadata:
        assert meta_processor is not None, \
            "use_metadata=True but meta_processor was not passed to build_hycnn_trans_softmoe()"
        meta_dim = meta_processor.meta_dim

    return HyCNNTransSoftMoE(
        num_classes    = _get("num_classes",    11),
        pretrained     = _get("pretrained",     True),
        embed_dim      = _get("embed_dim",      768),
        num_heads      = _get("num_heads",      8),
        window_size    = _get("window_size",    8),
        n_experts      = _get("n_experts",      4),
        n_slots        = _get("n_slots",        1),
        moe_expand     = _get("moe_expand",     4),
        mode           = _get("mode",           "dual_image"),
        image_type     = _get("image_type",     "dermoscopy"),
        use_metadata   = use_metadata,
        meta_dim       = meta_dim,
        meta_hidden    = _get("meta_hidden",    256),
        dropout        = _get("drop_rate",      0.1),
        drop_path_rate = _get("drop_path_rate", 0.1),
    )


# ═══════════════════════════════════════════════════════════════════
# Quick sanity check (uncomment to run locally)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, yaml

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load config từ yaml ────────────────────────────────────────────────────
    _cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "hycnn_trans_softmoe.yaml"
    )
    with open(_cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Lấy các setting từ config
    N_EXPERTS = cfg.get("n_experts",  4)
    N_SLOTS   = cfg.get("n_slots",    1)
    EMBED_DIM = cfg.get("embed_dim",  768)
    MODE      = cfg.get("mode",       "dual_image")
    META_DIM  = 28   # từ MetadataProcessor.meta_dim — hardcode cho test
    B         = 2

    print(f"Config: n_experts={N_EXPERTS}, n_slots={N_SLOTS}, "
          f"embed_dim={EMBED_DIM}, mode={MODE}")
    print(f"  → total slots per forward = {N_EXPERTS * N_SLOTS}")
    print("=" * 60)

    derm = torch.randn(B, 3, 256, 256).to(device)
    clin = torch.randn(B, 3, 256, 256).to(device)
    meta = torch.randn(B, META_DIM).to(device)

    # ── Test 1: dual_image, no metadata ───────────────────────────────────────
    print("Test 1: dual_image, use_metadata=False")
    m = HyCNNTransSoftMoE(
        num_classes=11, pretrained=False,
        embed_dim=EMBED_DIM, n_experts=N_EXPERTS, n_slots=N_SLOTS,
        mode="dual_image", use_metadata=False,
    ).to(device)
    out = m(clin, derm)
    assert out.shape == (B, 11), f"Expected ({B},11), got {out.shape}"
    print(f"  Output: {out.shape}  ✓")

    # ── Test 2: dual_image, with metadata ─────────────────────────────────────
    print("Test 2: dual_image, use_metadata=True")
    m = HyCNNTransSoftMoE(
        num_classes=11, pretrained=False,
        embed_dim=EMBED_DIM, n_experts=N_EXPERTS, n_slots=N_SLOTS,
        mode="dual_image", use_metadata=True, meta_dim=META_DIM,
    ).to(device)
    out = m(clin, derm, metadata=meta)
    assert out.shape == (B, 11), f"Expected ({B},11), got {out.shape}"
    print(f"  Output: {out.shape}  ✓")

    # ── Test 3: single_image, no metadata ─────────────────────────────────────
    print("Test 3: single_image, use_metadata=False")
    m = HyCNNTransSoftMoE(
        num_classes=11, pretrained=False,
        embed_dim=EMBED_DIM, n_experts=N_EXPERTS, n_slots=N_SLOTS,
        mode="single_image", use_metadata=False,
    ).to(device)
    out = m(derm)
    assert out.shape == (B, 11), f"Expected ({B},11), got {out.shape}"
    print(f"  Output: {out.shape}  ✓")

    # ── Param count ───────────────────────────────────────────────────────────
    total = sum(p.numel() for p in m.parameters()) / 1e6
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f"num experts: {N_EXPERTS}")
    print(f"num slots: {N_SLOTS}")
    print("=" * 60)
    print(f"  Total params    : {total:.1f}M")
    print(f"  Trainable params: {trainable:.1f}M")
    print("All tests passed!")
