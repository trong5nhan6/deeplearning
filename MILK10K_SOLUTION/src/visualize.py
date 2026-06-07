"""
Post-training visualization utilities for MILK10k.

1. log_class_weights_table  — pretty-print per-class counts / freqs / weights
2. log_confusion_matrix     — ASCII confusion matrix in log file
3. collect_embeddings       — extract backbone/projected features via hook
4. visualize_embeddings     — 2D scatter (UMAP / t-SNE) saved as PNG
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.metrics import LABEL_COLS


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Class weight table
# ─────────────────────────────────────────────────────────────────────────────

def log_class_weights_table(
    labels_np:    np.ndarray,
    class_weight: Optional[torch.Tensor],
    pos_weight:   Optional[torch.Tensor],
    class_freq:   Optional[torch.Tensor],
    weight_clamp: float,
    logger,
):
    """
    Print a per-class summary table before training starts.

    Example:
    ═══════════════════════════════════════════════════════════════
      CLASS WEIGHTS / FREQUENCIES
    Class          |  Count |   Freq% |   Weight |  Clamped |   pi_k
    ───────────────────────────────────────────────────────────────
    BCC            |   2516 |  48.02% |    0.150 |    0.150 | 0.4802
    ...
    MAL_OTH        |     18 |   0.34% |   10.000 |    3.000 | 0.0034
    ═══════════════════════════════════════════════════════════════
    """
    counts = labels_np.sum(axis=0).clip(min=1).astype(int)
    total  = int(counts.sum())
    freqs  = counts / total

    cw_np    = class_weight.cpu().numpy() if class_weight is not None else None
    cw_clamp = np.clip(cw_np, None, weight_clamp) if (cw_np is not None and weight_clamp > 0) else cw_np
    pw_np    = pos_weight.cpu().numpy()   if pos_weight   is not None else None
    cf_np    = class_freq.cpu().numpy()   if class_freq   is not None else None

    # Build column list
    cols: list[tuple[str, int]] = [
        ("Class",   14),
        ("Count",    7),
        ("Freq%",    8),
    ]
    if cw_np    is not None: cols += [("Weight", 9), ("Clamped", 9)]
    if pw_np    is not None: cols += [("pos_w",  7)]
    if cf_np    is not None: cols += [("pi_k",   8)]

    sep   = "  "
    hdr   = sep.join(f"{name:>{width}}" for name, width in cols)
    divid = "─" * len(hdr)

    logger.info("═" * len(hdr))
    logger.info("  CLASS WEIGHTS / FREQUENCIES BEFORE TRAINING")
    logger.info(hdr)
    logger.info(divid)

    for i, cls in enumerate(LABEL_COLS):
        row_vals: list[str] = [
            f"{cls:<14}",
            f"{counts[i]:>7}",
            f"{freqs[i]*100:>7.2f}%",
        ]
        if cw_np    is not None:
            row_vals += [f"{cw_np[i]:>9.3f}", f"{cw_clamp[i]:>9.3f}"]
        if pw_np    is not None:
            row_vals += [f"{pw_np[i]:>7.3f}"]
        if cf_np    is not None:
            row_vals += [f"{cf_np[i]:>8.4f}"]
        logger.info(sep.join(row_vals))

    logger.info(divid)
    logger.info(f"  Total samples: {total}  |  IR = {counts.max()/counts.min():.0f}×"
                f"  |  weight_clamp = {weight_clamp if weight_clamp > 0 else 'off'}")
    logger.info("═" * len(hdr))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def log_confusion_matrix(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    logger,
) -> np.ndarray:
    """
    Log an ASCII confusion matrix.
    Rows = true class, Cols = predicted class.
    Returns the confusion matrix as ndarray.
    """
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        logger.warning("[VIZ] sklearn not available — skipping confusion matrix")
        return np.zeros((len(LABEL_COLS), len(LABEL_COLS)), dtype=int)

    y_true = labels_np.argmax(axis=1)
    y_pred = logits_np.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_COLS))))

    abbrevs  = [c[:6] for c in LABEL_COLS]
    col_w    = 7
    row_head = 8

    header = " " * row_head + "".join(f"{a:>{col_w}}" for a in abbrevs)
    divid  = "─" * len(header)

    logger.info("═" * len(header))
    logger.info("  CONFUSION MATRIX  (rows = true class, cols = predicted)")
    logger.info(header)
    logger.info(divid)
    for i, row_lbl in enumerate(abbrevs):
        row = f"{row_lbl:<{row_head}}" + "".join(f"{cm[i,j]:>{col_w}}" for j in range(len(LABEL_COLS)))
        logger.info(row)
    logger.info(divid)

    # Per-class recall from diagonal
    recalls = [
        cm[i, i] / max(cm[i].sum(), 1)
        for i in range(len(LABEL_COLS))
    ]
    rec_str = "  ".join(f"{LABEL_COLS[i][:4]}={recalls[i]:.2f}" for i in range(len(LABEL_COLS)))
    logger.info(f"  Recall: {rec_str}")
    logger.info("═" * len(header))
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature collection via forward hook
# ─────────────────────────────────────────────────────────────────────────────

def collect_embeddings(
    model:     nn.Module,
    loader,
    device,
    proj_head: Optional[nn.Module] = None,
    use_amp:   bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect backbone features (or projected features) + labels from *loader*.

    - If proj_head is not None → return L2-normalized projected features (contrastive space).
    - Otherwise                → return raw pre-head backbone features.

    Uses a forward hook on the first Linear layer of model.head / .classifier / .fc
    to capture features without modifying the model.

    Returns
    -------
    feats  : (N, D) float32 numpy
    labels : (N,)   int    numpy  (argmax of one-hot label)
    """
    from src.validate import _forward

    feat_store: dict = {}
    hook_handle      = None

    for attr in ("head", "classifier", "fc"):
        if not hasattr(model, attr):
            continue
        mod    = getattr(model, attr)
        target = None
        if isinstance(mod, nn.Linear):
            target = mod
        elif isinstance(mod, nn.Sequential):
            for sub in mod.modules():
                if isinstance(sub, nn.Linear):
                    target = sub
                    break
        if target is not None:
            def _hook(m, inp, out, _store=feat_store):
                _store["feat"] = inp[0].detach().cpu()
            hook_handle = target.register_forward_hook(_hook)
            break

    all_feats:  list = []
    all_labels: list = []

    model.eval()
    if proj_head is not None:
        proj_head.eval()

    with torch.no_grad():
        for batch in loader:
            lbl = batch.get("labels")
            if lbl is None:
                continue
            with torch.amp.autocast(
                "cuda",
                enabled=use_amp and torch.cuda.is_available()
            ):
                _forward(model, batch, device)

            feat = feat_store.get("feat")       # [B, D] CPU
            if feat is None:
                continue

            if proj_head is not None:
                feat = proj_head(feat.to(device)).detach().cpu()

            all_feats.append(feat.float().numpy())
            all_labels.append(
                lbl.argmax(dim=1).numpy() if lbl.dim() == 2
                else lbl.long().numpy()
            )

    if hook_handle is not None:
        hook_handle.remove()

    if not all_feats:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=int)

    return (
        np.concatenate(all_feats,  axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dimensionality reduction + plot
# ─────────────────────────────────────────────────────────────────────────────

# 11-class colour palette — dark-background friendly
_CLASS_COLORS = [
    "#4e9af1",   # BCC       — blue
    "#ff6b6b",   # MEL       — red-orange
    "#f8b400",   # NV        — yellow
    "#2ecc71",   # BKL       — green
    "#9b59b6",   # AKIEC     — purple
    "#e67e22",   # SCC       — orange
    "#1abc9c",   # DF        — teal
    "#e74c3c",   # VASC      — crimson
    "#74b9ff",   # SEB       — sky blue
    "#95a5a6",   # UNK       — grey
    "#ff9ff3",   # MAL_OTH   — pink  (minority — make it stand out)
]


def _reduce_2d(feats: np.ndarray) -> np.ndarray:
    """UMAP preferred → t-SNE fallback."""
    try:
        import umap as _umap
        reducer = _umap.UMAP(
            n_components=2, random_state=42,
            n_neighbors=min(15, len(feats) - 1),
            min_dist=0.1,
        )
        return reducer.fit_transform(feats)
    except Exception:
        pass

    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, len(feats) // 4))
    return TSNE(
        n_components=2, random_state=42,
        perplexity=perplexity, n_iter=1000, verbose=0,
    ).fit_transform(feats)


def visualize_embeddings(
    model:      nn.Module,
    val_loader,
    device,
    save_path:  str,
    proj_head:  Optional[nn.Module]    = None,
    prototypes: Optional[torch.Tensor] = None,  # [C, D] EMA prototypes
    proto_init: Optional[torch.Tensor] = None,  # [C,] bool — which prototypes are set
    use_amp:    bool = True,
    title:      str  = "BCL Embedding Space (val set)",
    logger      = None,
):
    """
    Collect val-set embeddings, reduce to 2D, and save a scatter plot PNG.

    Parameters
    ----------
    prototypes : optional [C, D] tensor — class prototypes from BCLLoss.proto
    proto_init : optional [C,] bool tensor — which classes have an init prototype
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if logger:
            logger.warning("[VIZ] matplotlib not installed — skipping embedding plot")
        return

    if logger:
        logger.info(f"[VIZ] Collecting {'projected' if proj_head else 'backbone'} embeddings ...")

    feats, labels = collect_embeddings(model, val_loader, device, proj_head, use_amp)

    if len(feats) == 0:
        if logger:
            logger.warning("[VIZ] No features collected — skipping embedding plot")
        return

    if logger:
        logger.info(f"[VIZ] Reducing {len(feats)} × {feats.shape[1]}D → 2D ...")

    # ── combine feats + prototypes for coherent 2D space ──────────────────────
    proto_2d   = None
    proto_mask = None

    if prototypes is not None and proto_init is not None:
        proto_np = prototypes.detach().cpu().float().numpy()
        init_np  = proto_init.detach().cpu().numpy().astype(bool)
        if init_np.any():
            combined    = np.concatenate([feats, proto_np[init_np]], axis=0)
            combined_2d = _reduce_2d(combined)
            xy          = combined_2d[:len(feats)]
            p_xy        = combined_2d[len(feats):]

            proto_2d              = np.zeros((len(LABEL_COLS), 2), dtype=float)
            proto_2d[init_np]     = p_xy
            proto_mask            = init_np
        else:
            xy = _reduce_2d(feats)
    else:
        xy = _reduce_2d(feats)

    # ── plot ──────────────────────────────────────────────────────────────────
    bg_dark  = "#12122a"
    ax_dark  = "#1a1a2e"

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(bg_dark)
    ax.set_facecolor(ax_dark)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

    # Sample scatter
    for c in range(len(LABEL_COLS)):
        mask = labels == c
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=_CLASS_COLORS[c % len(_CLASS_COLORS)],
            alpha=0.55, s=16, edgecolors="none",
            label=f"{LABEL_COLS[c]} ({mask.sum()})",
        )

    # Prototype stars
    if proto_2d is not None:
        for c in range(len(LABEL_COLS)):
            if not proto_mask[c]:
                continue
            px, py = proto_2d[c]
            ax.scatter(
                px, py,
                c=_CLASS_COLORS[c % len(_CLASS_COLORS)],
                marker="*", s=350,
                edgecolors="white", linewidths=0.7,
                zorder=10,
            )
            ax.annotate(
                f"{LABEL_COLS[c][:5]}\nproto",
                (px, py), textcoords="offset points",
                xytext=(6, 3), fontsize=6.5,
                color="white", alpha=0.9,
            )

    method = "UMAP" if _umap_available() else "t-SNE"
    ax.set_title(f"{title}  [{method}]", color="white", fontsize=12, pad=10)
    ax.legend(
        loc="upper right", fontsize=7, framealpha=0.35,
        facecolor=ax_dark, edgecolor="#333366", labelcolor="white",
        ncol=2, markerscale=1.8,
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    if logger:
        logger.info(f"[VIZ] Embedding plot saved → {save_path}")


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except Exception:
        return False
