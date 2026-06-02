"""
prepare_data.py — Preprocess ISIC MILK10k raw CSVs into combined per-lesion CSVs.

Input (raw download structure):
  datasets/MILK10k/train/
      MILK10k_Training_Metadata.csv     (10480 rows = 2 per lesion)
      MILK10k_Training_GroundTruth.csv  (5240 rows = 1 per lesion)
      MILK10k_Training_Input/
          <lesion_id>/
              <isic_id>.jpg   (clinical)
              <isic_id>.jpg   (dermoscopic)

  datasets/MILK10k/test/
      MILK10k_Test_Metadata.csv         (958 rows = 2 per lesion)
      MILK10k_Test_Input/
          <lesion_id>/
              <isic_id>.jpg
              <isic_id>.jpg

Output:
  datasets/MILK10k/train/train_combined.csv   (5240 rows, 1 per lesion)
  datasets/MILK10k/test/test_combined.csv     (479 rows, 1 per lesion)

Combined CSV schema (per lesion):
  lesion_id, clinical_path, derm_path,
  age_approx, sex, skin_tone_class, site,
  clin_MONET_ulceration_crust, clin_MONET_hair, ...,
  derm_MONET_ulceration_crust, derm_MONET_hair, ...,
  [AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC]  ← train only

Paths are stored RELATIVE to the image_dir, e.g.:
  clinical_path = "IL_0000652/ISIC_8149219.jpg"

Usage:
  python src/prepare_data.py
  python src/prepare_data.py --base_dir datasets/MILK10k
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABEL_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
              "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]

MONET_COLS = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

# Exact string values in image_type column
CLINICAL_TYPE = "clinical: close-up"
DERM_TYPE     = "dermoscopic"


def pivot_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from 2 rows per lesion → 1 row per lesion.
    Separates clinical and dermoscopic rows, then merges side by side.

    Returns DataFrame with columns:
      lesion_id, clinical_isic_id, derm_isic_id,
      age_approx, sex, skin_tone_class, site,
      clin_MONET_*, derm_MONET_*
    """
    # ── Split by image type ───────────────────────────────────────────────────
    clin = meta_df[meta_df["image_type"] == CLINICAL_TYPE].copy()
    derm = meta_df[meta_df["image_type"] == DERM_TYPE].copy()

    clin = clin.set_index("lesion_id")
    derm = derm.set_index("lesion_id")

    # ── Shared metadata (same for both rows; take from clinical) ─────────────
    shared_cols = ["age_approx", "sex", "skin_tone_class", "site"]
    shared = clin[shared_cols].copy()

    # ── ISIC IDs (= filenames without extension) ─────────────────────────────
    # Path = <lesion_id>/<isic_id>.jpg
    clin_ids = clin["isic_id"].rename("clinical_isic_id")
    derm_ids = derm["isic_id"].rename("derm_isic_id")

    # ── MONET scores — keep separate with prefix ──────────────────────────────
    # Only take MONET cols that actually exist in the dataframe
    existing_monet = [c for c in MONET_COLS if c in meta_df.columns]
    clin_monet = clin[existing_monet].add_prefix("clin_")
    derm_monet = derm[existing_monet].add_prefix("derm_")

    # ── Merge all ─────────────────────────────────────────────────────────────
    combined = (
        shared
        .join(clin_ids,  how="left")
        .join(derm_ids,  how="left")
        .join(clin_monet, how="left")
        .join(derm_monet, how="left")
        .reset_index()
        .rename(columns={"index": "lesion_id"})
    )

    # Ensure lesion_id column is correctly named after reset_index
    if "lesion_id" not in combined.columns:
        combined = combined.rename(columns={combined.columns[0]: "lesion_id"})

    # ── Build relative image paths ────────────────────────────────────────────
    # Format: "IL_XXXXXXX/ISIC_XXXXXXX.jpg"
    combined["clinical_path"] = (
        combined["lesion_id"] + "/" + combined["clinical_isic_id"] + ".jpg"
    )
    combined["derm_path"] = (
        combined["lesion_id"] + "/" + combined["derm_isic_id"] + ".jpg"
    )

    return combined


def build_train_combined(
    meta_path:  str,
    gt_path:    str,
    image_dir:  str,
    out_path:   str,
):
    """Build and save train_combined.csv."""
    print(f"Reading metadata: {meta_path}")
    meta_df = pd.read_csv(meta_path)

    print(f"Reading ground truth: {gt_path}")
    gt_df = pd.read_csv(gt_path)

    print(f"Metadata shape: {meta_df.shape}")
    print(f"Ground truth shape: {gt_df.shape}")

    # ── Pivot metadata ────────────────────────────────────────────────────────
    combined = pivot_metadata(meta_df)
    print(f"After pivot: {combined.shape}")

    # ── Merge with labels ─────────────────────────────────────────────────────
    combined = combined.merge(
        gt_df[["lesion_id"] + LABEL_COLS],
        on="lesion_id",
        how="left",
    )
    missing_labels = combined[LABEL_COLS].isna().any(axis=1).sum()
    if missing_labels > 0:
        print(f"  [WARN] {missing_labels} lesions have missing labels — filling with 0.")
        combined[LABEL_COLS] = combined[LABEL_COLS].fillna(0.0)

    # ── Verify images exist ───────────────────────────────────────────────────
    image_root = Path(image_dir)
    missing_clin = 0
    missing_derm = 0
    for _, row in combined.head(20).iterrows():  # spot check first 20
        if not (image_root / row["clinical_path"]).exists():
            missing_clin += 1
        if not (image_root / row["derm_path"]).exists():
            missing_derm += 1
    if missing_clin or missing_derm:
        print(f"  [WARN] Spot check: {missing_clin} clinical / {missing_derm} derm images NOT found "
              f"in {image_dir}. Check paths.")
    else:
        print(f"  [OK] Spot check: first 20 images found in {image_dir}")

    # ── Final column order ────────────────────────────────────────────────────
    monet_cols = [c for c in combined.columns if "MONET" in c]
    ordered_cols = (
        ["lesion_id", "clinical_path", "derm_path",
         "clinical_isic_id", "derm_isic_id",
         "age_approx", "sex", "skin_tone_class", "site"]
        + monet_cols
        + LABEL_COLS
    )
    ordered_cols = [c for c in ordered_cols if c in combined.columns]
    combined = combined[ordered_cols]

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(combined)} rows, {len(combined.columns)} cols)")

    # ── Label distribution ────────────────────────────────────────────────────
    print("\n── Label distribution ──────────────────────────────────")
    for col in LABEL_COLS:
        n = int(combined[col].sum())
        pct = 100 * n / len(combined)
        print(f"  {col:<10}: {n:>5}  ({pct:>5.1f}%)")

    return combined


def build_test_combined(
    meta_path: str,
    image_dir: str,
    out_path:  str,
):
    """Build and save test_combined.csv (no labels)."""
    print(f"\nReading test metadata: {meta_path}")
    meta_df = pd.read_csv(meta_path)
    print(f"Test metadata shape: {meta_df.shape}")

    combined = pivot_metadata(meta_df)
    print(f"After pivot: {combined.shape}")

    # ── Verify images ─────────────────────────────────────────────────────────
    image_root = Path(image_dir)
    sample = combined.head(5)
    for _, row in sample.iterrows():
        cp = image_root / row["clinical_path"]
        dp = image_root / row["derm_path"]
        clin_ok = "✓" if cp.exists() else "✗"
        derm_ok = "✓" if dp.exists() else "✗"
        print(f"  {row['lesion_id']}: clin={clin_ok} derm={derm_ok}")

    # ── Final column order ────────────────────────────────────────────────────
    monet_cols = [c for c in combined.columns if "MONET" in c]
    ordered_cols = (
        ["lesion_id", "clinical_path", "derm_path",
         "clinical_isic_id", "derm_isic_id",
         "age_approx", "sex", "skin_tone_class", "site"]
        + monet_cols
    )
    ordered_cols = [c for c in ordered_cols if c in combined.columns]
    combined = combined[ordered_cols]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(combined)} rows)")
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess MILK10k raw CSVs → per-lesion combined CSVs."
    )
    p.add_argument(
        "--base_dir",
        default="datasets/MILK10k",
        help="Root directory of the MILK10k dataset (default: datasets/MILK10k)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    base   = Path(args.base_dir)

    # ── Train ─────────────────────────────────────────────────────────────────
    build_train_combined(
        meta_path=str(base / "train" / "MILK10k_Training_Metadata.csv"),
        gt_path=str(  base / "train" / "MILK10k_Training_GroundTruth.csv"),
        image_dir=str(base / "train" / "MILK10k_Training_Input"),
        out_path=str( base / "train" / "train_combined.csv"),
    )

    # ── Test ──────────────────────────────────────────────────────────────────
    build_test_combined(
        meta_path=str(base / "test" / "MILK10k_Test_Metadata.csv"),
        image_dir=str(base / "test" / "MILK10k_Test_Input"),
        out_path=str( base / "test"  / "test_combined.csv"),
    )

    print("\nDone! Next steps:")
    print("  1. python src/split_data.py --csv datasets/MILK10k/train/train_combined.csv \\")
    print("                              --out_dir datasets/MILK10k/splits --fold 0")
    print("  2. python train.py --config configs/swin_base.yaml")
