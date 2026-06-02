"""
split_data.py — Train/validation split for MILK10k.

Supports:
  - Simple stratified train_test_split
  - StratifiedKFold (multi-fold cross-validation)

Multi-label stratification uses the argmax label (main_label column)
or MultilabelStratifiedKFold from iterstrat if available.

Usage:
    python src/split_data.py \
        --csv datasets/train/train_metadata.csv \
        --out_dir datasets/splits \
        --fold 0 \
        --val_size 0.2 \
        --seed 42 \
        --n_folds 5
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


LABEL_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
              "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]


def detect_label_cols(df: pd.DataFrame) -> List[str]:
    """Return which of the 11 standard label cols exist in df."""
    return [c for c in LABEL_COLS if c in df.columns]


def make_main_label(df: pd.DataFrame, label_cols: List[str]) -> pd.Series:
    """
    For stratification: derive a single integer label per sample.
    Uses argmax over label columns (most-probable class).
    If a row has all zeros, assigns label = -1 (unknown).
    """
    mat = df[label_cols].values.astype(float)
    # Argmax per row
    main = np.argmax(mat, axis=1)
    # Mark rows with no positive label
    row_sum = mat.sum(axis=1)
    main[row_sum == 0] = -1
    return pd.Series(main, index=df.index, name="main_label")


def try_multilabel_stratified(
    df: pd.DataFrame,
    label_cols: List[str],
    n_folds: int,
    fold: int,
    seed: int,
) -> Optional[tuple]:
    """
    Attempt MultilabelStratifiedKFold from iterstrat.
    Falls back to None if package is not installed.
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        X = np.zeros(len(df))
        y = df[label_cols].values
        splits = list(mskf.split(X, y))
        train_idx, val_idx = splits[fold]
        return df.iloc[train_idx], df.iloc[val_idx]
    except ImportError:
        warnings.warn(
            "iterstrat not installed. Falling back to single-label stratification. "
            "Install with: pip install iterative-stratification"
        )
        return None


def split_fold(
    csv_path: str,
    out_dir: str,
    fold: int = 0,
    n_folds: int = 10,
    val_size: float = 0.1,
    seed: int = 42,
    use_kfold: bool = True,
):
    """
    Main split function.

    Parameters
    ----------
    csv_path   : path to train_metadata.csv
    out_dir    : output directory for split CSVs
    fold       : which fold to use (0-indexed)
    n_folds    : total number of folds (used when use_kfold=True)
    val_size   : fraction for validation (used when use_kfold=False)
    seed       : random seed
    use_kfold  : if True, use StratifiedKFold; else simple split
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    label_cols = detect_label_cols(df)
    if not label_cols:
        raise ValueError(
            f"No label columns found in {csv_path}. "
            f"Expected: {LABEL_COLS}"
        )
    print(f"Found label columns: {label_cols}")

    # Try multi-label stratification first
    if use_kfold and len(label_cols) > 1:
        result = try_multilabel_stratified(df, label_cols, n_folds, fold, seed)
        if result is not None:
            train_df, val_df = result
            print(f"Used MultilabelStratifiedKFold: "
                  f"train={len(train_df)}, val={len(val_df)}")
            _save_splits(train_df, val_df, out_dir, fold)
            return

    # Derive main_label for single-label stratification
    main_label = make_main_label(df, label_cols)
    df["main_label"] = main_label

    if use_kfold:
        skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        # Filter out rows with main_label=-1 from stratification
        # but still include them in train/val
        strat_idx = df.index[df["main_label"] != -1]
        splits    = list(skf.split(strat_idx, df.loc[strat_idx, "main_label"]))

        if fold >= len(splits):
            raise ValueError(f"fold={fold} but only {len(splits)} folds available.")

        train_strat_idx, val_strat_idx = splits[fold]
        train_df = df.loc[strat_idx[train_strat_idx]]
        val_df   = df.loc[strat_idx[val_strat_idx]]

        # Append unknown-label rows to train
        unknown_df = df[df["main_label"] == -1]
        if len(unknown_df):
            print(f"  {len(unknown_df)} rows with no positive label → appended to train.")
            train_df = pd.concat([train_df, unknown_df], ignore_index=True)

        print(f"StratifiedKFold fold {fold}/{n_folds-1}: "
              f"train={len(train_df)}, val={len(val_df)}")

    else:
        strat_mask  = df["main_label"] != -1
        strat_df    = df[strat_mask]
        unknown_df  = df[~strat_mask]

        train_df, val_df = train_test_split(
            strat_df,
            test_size=val_size,
            random_state=seed,
            stratify=strat_df["main_label"],
        )
        if len(unknown_df):
            train_df = pd.concat([train_df, unknown_df], ignore_index=True)

        print(f"train_test_split: train={len(train_df)}, val={len(val_df)}")

    # Drop helper column before saving
    train_df = train_df.drop(columns=["main_label"], errors="ignore")
    val_df   = val_df.drop(columns=["main_label"],   errors="ignore")

    _save_splits(train_df, val_df, out_dir, fold)
    _print_class_distribution(train_df, val_df, label_cols)


def _save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path, fold: int):
    train_path = out_dir / f"train_fold{fold}.csv"
    val_path   = out_dir / f"val_fold{fold}.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")


def _print_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_cols: List[str],
):
    print("\n-- Class Distribution ----------------------------------")
    print(f"{'Class':<12} {'Train':>8} {'Val':>6} {'Train%':>8} {'Val%':>6}")
    print("-" * 46)
    for col in label_cols:
        tr_n = int(train_df[col].sum()) if col in train_df.columns else 0
        vl_n = int(val_df[col].sum())   if col in val_df.columns   else 0
        tr_p = 100 * tr_n / max(len(train_df), 1)
        vl_p = 100 * vl_n / max(len(val_df),   1)
        print(f"{col:<12} {tr_n:>8} {vl_n:>6} {tr_p:>7.1f}% {vl_p:>5.1f}%")
    print("-" * 46)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Split MILK10k train CSV into train/val folds.")
    p.add_argument("--csv",      required=True, help="Path to train_metadata.csv")
    p.add_argument("--out_dir",  required=True, help="Output directory for split CSVs")
    p.add_argument("--fold",     type=int,   default=0,    help="Fold index (0-indexed)")
    p.add_argument("--n_folds",  type=int,   default=10,   help="Total number of folds")
    p.add_argument("--val_size", type=float, default=0.1,  help="Val fraction (if no kfold)")
    p.add_argument("--seed",     type=int,   default=42,   help="Random seed")
    p.add_argument("--no_kfold", action="store_true",      help="Use simple split instead of KFold")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_fold(
        csv_path=args.csv,
        out_dir=args.out_dir,
        fold=args.fold,
        n_folds=args.n_folds,
        val_size=args.val_size,
        seed=args.seed,
        use_kfold=not args.no_kfold,
    )
