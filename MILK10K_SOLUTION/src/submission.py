"""
submission.py — Build and validate the ISIC MILK10k submission CSV.

Expected format:
    lesion, AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC
Values must be float in [0, 1].
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


LABEL_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
              "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]

SUBMISSION_COLS = ["lesion"] + LABEL_COLS


def build_submission(
    lesion_ids: List[str],
    probs: np.ndarray,
    out_path: str,
    label_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create and save a submission CSV.

    Parameters
    ----------
    lesion_ids : list of lesion ID strings, length N
    probs      : (N, 11) float array of probabilities in [0, 1]
    out_path   : path to save the CSV
    label_cols : column names (defaults to LABEL_COLS)
    """
    label_cols = label_cols or LABEL_COLS
    assert len(lesion_ids) == len(probs), (
        f"lesion_ids length {len(lesion_ids)} != probs rows {len(probs)}"
    )
    assert probs.shape[1] == len(label_cols), (
        f"probs has {probs.shape[1]} columns but {len(label_cols)} label cols"
    )

    # Clip to [0, 1] and fill NaN with 0
    probs = np.nan_to_num(probs, nan=0.0)
    probs = np.clip(probs, 0.0, 1.0)

    df = pd.DataFrame(probs, columns=label_cols)
    df.insert(0, "lesion", lesion_ids)

    # Ensure column order matches official format
    df = df[SUBMISSION_COLS]

    # Validate
    _validate_submission(df)

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Submission saved → {out_path}  ({len(df)} rows)")
    return df


def _validate_submission(df: pd.DataFrame):
    """Check that submission follows competition rules. Raises on violation."""
    # Column presence
    missing = [c for c in SUBMISSION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Submission missing columns: {missing}")

    # Value range
    for col in LABEL_COLS:
        bad = ((df[col] < 0) | (df[col] > 1)).sum()
        if bad > 0:
            raise ValueError(
                f"Column '{col}' has {bad} values outside [0, 1]."
            )

    # NaN check
    if df[LABEL_COLS].isna().any().any():
        raise ValueError("Submission contains NaN values.")

    # Duplicate lesion IDs
    dupes = df["lesion"].duplicated().sum()
    if dupes > 0:
        raise ValueError(f"Submission has {dupes} duplicate lesion IDs.")

    print(f"[OK] Submission validated: {len(df)} lesions, "
          f"no NaN, all values in [0, 1].")


def load_and_validate(path: str) -> pd.DataFrame:
    """Load an existing submission CSV and validate it."""
    df = pd.read_csv(path)
    _validate_submission(df)
    return df


def ensemble_submissions(
    paths: List[str],
    weights: Optional[List[float]] = None,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Average multiple submission CSVs (weighted ensemble).

    Parameters
    ----------
    paths   : list of paths to submission CSVs
    weights : optional list of weights (uniform if None)
    out_path: if given, saves the result
    """
    if weights is None:
        weights = [1.0] * len(paths)
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    dfs = [pd.read_csv(p) for p in paths]

    # Verify lesion order is consistent
    ref_lesions = dfs[0]["lesion"].values
    for i, d in enumerate(dfs[1:], start=1):
        if not np.array_equal(d["lesion"].values, ref_lesions):
            raise ValueError(
                f"Submission {paths[i]} has different lesion order than {paths[0]}."
                " Sort submissions before ensembling."
            )

    ensemble_probs = np.zeros((len(ref_lesions), len(LABEL_COLS)), dtype=np.float64)
    for df, w in zip(dfs, weights):
        ensemble_probs += w * df[LABEL_COLS].values

    result_df = pd.DataFrame(ensemble_probs, columns=LABEL_COLS)
    result_df.insert(0, "lesion", ref_lesions)
    result_df = result_df[SUBMISSION_COLS]
    _validate_submission(result_df)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Ensemble submission saved → {out_path}")

    return result_df
