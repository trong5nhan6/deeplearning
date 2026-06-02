"""
metrics.py — Evaluation metrics for MILK10k multi-label classification.

Official metric: Macro F1 / Dice with threshold = 0.5.
Optional: per-class threshold search on validation set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score


LABEL_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
              "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply threshold to probability array."""
    return (probs >= threshold).astype(int)


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Macro F1 score (as used in MILK10k evaluation).
    y_true : (N, 11) binary ground-truth
    y_pred : (N, 11) probability predictions [0,1]
    """
    y_bin = binarize(y_pred, threshold)
    # zero_division=0 avoids warnings for classes with no positive predictions
    return f1_score(y_true, y_bin, average="macro", zero_division=0)


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Returns per-class F1 scores as a dictionary."""
    y_bin   = binarize(y_pred, threshold)
    scores  = f1_score(y_true, y_bin, average=None, zero_division=0)
    names   = label_names or LABEL_COLS
    return {n: float(s) for n, s in zip(names, scores)}


def binary_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Element-wise binary accuracy across all samples and label columns."""
    y_bin = binarize(y_pred, threshold)
    return float((y_bin == y_true.astype(int)).mean())


def _top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of samples that have ≥1 true-positive label in the top-k predictions.
    Samples with no positive label are excluded from the denominator."""
    top_k_idx = np.argsort(logits, axis=1)[:, -k:]
    valid, correct = 0, 0
    for i in range(logits.shape[0]):
        true_pos = np.where(labels[i] == 1)[0]
        if len(true_pos) == 0:
            continue
        valid += 1
        if np.any(np.isin(true_pos, top_k_idx[i])):
            correct += 1
    return correct / valid if valid > 0 else 0.0


def compute_full_summary(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Extended metrics for end-of-training summary: Acc@k, Precision, Recall, F1, ROC-AUC."""
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    probs = sigmoid(logits)
    preds = binarize(probs, threshold)

    acc1      = _top_k_accuracy(logits, labels, k=1)
    acc5      = _top_k_accuracy(logits, labels, k=5)
    precision = float(precision_score(labels, preds, average="macro", zero_division=0))
    recall    = float(recall_score(labels, preds, average="macro", zero_division=0))
    f1        = float(f1_score(labels, preds, average="macro", zero_division=0))

    try:
        roc_auc = float(roc_auc_score(labels, probs, average="macro"))
    except ValueError:
        roc_auc = 0.0

    return {
        "acc1":      acc1,
        "acc5":      acc5,
        "precision": precision,
        "recall":    recall,
        "macro_f1":  f1,
        "roc_auc":   roc_auc,
    }


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Full metric computation from raw model logits.
    Applies sigmoid internally.
    Returns dict with macro_f1, per_class_f1, and accuracy.
    """
    probs = sigmoid(logits)
    m_f1  = macro_f1(labels, probs, threshold)
    pc_f1 = per_class_f1(labels, probs, threshold, label_names)
    acc   = binary_accuracy(labels, probs, threshold)
    return {
        "macro_f1":     m_f1,
        "per_class_f1": pc_f1,
        "accuracy":     acc,
        "threshold":    threshold,
    }


def compute_metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Same as compute_metrics but accepts probabilities directly (post-sigmoid)."""
    m_f1  = macro_f1(labels, probs, threshold)
    pc_f1 = per_class_f1(labels, probs, threshold, label_names)
    return {
        "macro_f1":     m_f1,
        "per_class_f1": pc_f1,
        "threshold":    threshold,
    }


# ── Optional: per-class threshold search ────────────────────────────────────

def search_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    candidates: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Search for the best per-class threshold that maximises macro F1.
    Returns (thresholds_array_shape_11, best_macro_f1).
    NOTE: This can overfit on validation if used carelessly.
           The official submission uses threshold=0.5.
    """
    if candidates is None:
        candidates = np.arange(0.1, 0.95, 0.05)

    n_classes = probs.shape[1]
    names     = label_names or LABEL_COLS
    best_thresholds = np.full(n_classes, 0.5, dtype=np.float32)

    for cls_idx in range(n_classes):
        best_thr   = 0.5
        best_f1    = -1.0
        y_true_cls = labels[:, cls_idx]
        y_prob_cls = probs[:, cls_idx]

        for thr in candidates:
            y_bin = (y_prob_cls >= thr).astype(int)
            f1    = f1_score(y_true_cls, y_bin, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = thr

        best_thresholds[cls_idx] = best_thr

    # Compute final macro f1 with tuned thresholds
    y_pred_tuned = np.stack(
        [(probs[:, i] >= best_thresholds[i]).astype(int)
         for i in range(n_classes)],
        axis=1,
    )
    final_f1 = float(f1_score(labels, y_pred_tuned, average="macro", zero_division=0))
    return best_thresholds, final_f1


def save_thresholds(
    thresholds: np.ndarray,
    path: str,
    label_names: Optional[List[str]] = None,
):
    names  = label_names or LABEL_COLS
    thr_dict = {n: float(t) for n, t in zip(names, thresholds)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(thr_dict, f, indent=2)
    print(f"Saved thresholds → {path}")


def load_thresholds(
    path: str,
    label_names: Optional[List[str]] = None,
) -> np.ndarray:
    names = label_names or LABEL_COLS
    with open(path) as f:
        thr_dict = json.load(f)
    return np.array([thr_dict.get(n, 0.5) for n in names], dtype=np.float32)


# ── Convenience: aggregate logits/probs from list of tensors ────────────────

def concat_outputs(
    all_logits: List[torch.Tensor],
    all_labels: Optional[List[torch.Tensor]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    logits = torch.cat(all_logits, dim=0).cpu().numpy()
    labels = None
    if all_labels:
        labels = torch.cat(all_labels, dim=0).cpu().numpy()
    return logits, labels
