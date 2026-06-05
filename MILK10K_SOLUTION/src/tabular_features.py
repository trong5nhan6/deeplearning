"""
Feature engineering for MILK10k tabular data.
Pivots 2 rows/lesion (dermoscopic + clinical) into one wide row
and adds ratios, log transforms, interactions, composites.
"""

import numpy as np
import pandas as pd

MONET_BASE = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]
SHORT = {c: c.replace("MONET_", "") for c in MONET_BASE}
EPS = 1e-6

# clinically meaningful ratio pairs (numerator, denominator)
RATIO_PAIRS = [
    ("MONET_pigmented",            "MONET_erythema"),
    ("MONET_pigmented",            "MONET_ulceration_crust"),
    ("MONET_ulceration_crust",     "MONET_pigmented"),
    ("MONET_ulceration_crust",     "MONET_erythema"),
    ("MONET_vasculature_vessels",  "MONET_erythema"),
    ("MONET_vasculature_vessels",  "MONET_pigmented"),
    ("MONET_erythema",             "MONET_pigmented"),
    ("MONET_erythema",             "MONET_ulceration_crust"),
    ("MONET_gel_water_drop_fluid_dermoscopy_liquid", "MONET_pigmented"),
    ("MONET_skin_markings_pen_ink_purple_pen",       "MONET_pigmented"),
    ("MONET_hair",                 "MONET_erythema"),
]


def build_features(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Input : raw metadata DataFrame (one row per image, 2 rows per lesion).
    Output: one row per lesion with all engineered features.
    """
    DERM = "dermoscopic"
    CLIN = "clinical: close-up"

    derm = meta[meta["image_type"] == DERM].set_index("lesion_id")
    clin = meta[meta["image_type"] == CLIN].set_index("lesion_id")

    # ── base MONET split by view ───────────────────────────────────────────────
    derm_m = derm[MONET_BASE].rename(columns={c: f"derm_{c}" for c in MONET_BASE})
    clin_m = clin[MONET_BASE].rename(columns={c: f"clin_{c}" for c in MONET_BASE})

    shared = derm[["age_approx", "sex", "skin_tone_class",
                   "site", "image_manipulation"]].copy()

    df = shared.join(derm_m, how="left").join(clin_m, how="left")

    # ── delta & mean across views ──────────────────────────────────────────────
    for c in MONET_BASE:
        df[f"delta_{c}"] = df[f"derm_{c}"] - df[f"clin_{c}"]
        df[f"mean_{c}"]  = (df[f"derm_{c}"] + df[f"clin_{c}"]) / 2.0
        df[f"abs_delta_{c}"] = df[f"delta_{c}"].abs()   # cross-view inconsistency

    # ── log1p transforms ──────────────────────────────────────────────────────
    for c in MONET_BASE:
        for prefix in ("derm_", "clin_", "mean_"):
            col = f"{prefix}{c}"
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # ── MONET ratios (mean view) ───────────────────────────────────────────────
    for num, den in RATIO_PAIRS:
        n = f"mean_{num}"
        d = f"mean_{den}"
        name = f"ratio_{SHORT[num]}_over_{SHORT[den]}"
        df[name] = df[n] / (df[d] + EPS)

    # ── composite clinical scores ──────────────────────────────────────────────
    df["composite_malignant"]  = (df["mean_MONET_ulceration_crust"] +
                                  df["mean_MONET_erythema"] +
                                  df["mean_MONET_vasculature_vessels"])
    df["composite_pigmented"]  = (df["mean_MONET_pigmented"] +
                                  df["mean_MONET_skin_markings_pen_ink_purple_pen"])
    df["composite_benign_sign"] = (df["mean_MONET_hair"] +
                                   df["mean_MONET_gel_water_drop_fluid_dermoscopy_liquid"])
    df["composite_asymmetry"]  = df[[f"abs_delta_{c}" for c in MONET_BASE]].sum(axis=1)

    # ── age features ──────────────────────────────────────────────────────────
    df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median())
    df["age_group"]  = pd.cut(df["age_approx"],
                               bins=[0, 30, 40, 50, 60, 70, 80, 200],
                               labels=[0, 1, 2, 3, 4, 5, 6]).astype(float)

    # ── interactions ──────────────────────────────────────────────────────────
    df["age_x_pigmented"]   = df["age_approx"] * df["mean_MONET_pigmented"]
    df["age_x_ulceration"]  = df["age_approx"] * df["mean_MONET_ulceration_crust"]
    df["age_x_erythema"]    = df["age_approx"] * df["mean_MONET_erythema"]
    df["tone_x_pigmented"]  = df["skin_tone_class"] * df["mean_MONET_pigmented"]
    df["tone_x_ulceration"] = df["skin_tone_class"] * df["mean_MONET_ulceration_crust"]

    # ── encode categoricals as integer codes ─────────────────────────────────
    df["site"] = df["site"].fillna("unknown")
    for col in ["sex", "site", "image_manipulation"]:
        df[col] = pd.Categorical(df[col]).codes.astype(float)

    return df.reset_index()   # lesion_id back as column


CAT_FEATURE_NAMES = ["sex", "site", "image_manipulation"]
