"""
Tabular ML pipeline for MILK10k.
Models: LightGBM, XGBoost, CatBoost  (multiclass softmax, 5-fold CV)
Outputs:
  outputs/tabular/oof/           OOF probability arrays (.npy) for DL ensemble
  outputs/tabular/submissions/   per-model and ensemble submission CSV
"""

import os, warnings, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

# ── dirs ───────────────────────────────────────────────────────────────────────
OOF_DIR = "outputs/tabular/oof"
SUB_DIR = "outputs/tabular/submissions"
os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

CLASS_COLS = ["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]
N_CLASSES  = len(CLASS_COLS)
N_FOLDS    = 5
SEED       = 42
MONET_COLS = [
    "MONET_ulceration_crust", "MONET_hair", "MONET_vasculature_vessels",
    "MONET_erythema", "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def build_features(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot 2 rows per lesion (dermoscopic / clinical) into one wide row.
    MONET features are split by image_type; categorical taken from either row.
    """
    DERM  = "dermoscopic"
    CLIN  = "clinical: close-up"

    derm  = meta[meta["image_type"] == DERM].set_index("lesion_id")
    clin  = meta[meta["image_type"] == CLIN].set_index("lesion_id")

    derm_monet = derm[MONET_COLS].rename(columns={c: f"derm_{c}" for c in MONET_COLS})
    clin_monet = clin[MONET_COLS].rename(columns={c: f"clin_{c}" for c in MONET_COLS})

    # shared clinical features — take from derm row (same per lesion)
    shared = derm[["age_approx","sex","skin_tone_class","site","image_manipulation"]].copy()

    df = shared.join(derm_monet, how="left").join(clin_monet, how="left")

    # delta MONET (dermoscopic − clinical)
    for c in MONET_COLS:
        df[f"delta_{c}"] = df[f"derm_{c}"] - df[f"clin_{c}"]

    # mean MONET across both views
    for c in MONET_COLS:
        df[f"mean_{c}"] = (df[f"derm_{c}"] + df[f"clin_{c}"]) / 2.0

    # encode categoricals
    for col in ["sex","site","image_manipulation"]:
        df[col] = df[col].astype("category")

    df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median())
    df["site"]       = df["site"].cat.add_categories("unknown").fillna("unknown")

    return df.reset_index()   # lesion_id back as column


print("Loading data...")
train_meta = pd.read_csv("datasets/MILK10k/train/MILK10k_Training_Metadata.csv")
test_meta  = pd.read_csv("datasets/MILK10k/test/MILK10k_Test_Metadata.csv")
gt         = pd.read_csv("datasets/MILK10k/train/MILK10k_Training_GroundTruth.csv")

train_feat = build_features(train_meta)
test_feat  = build_features(test_meta)

# merge labels
train_feat = train_feat.merge(gt, on="lesion_id", how="left")
y = train_feat[CLASS_COLS].values.argmax(axis=1)      # single-label index

FEATURE_COLS = [c for c in train_feat.columns
                if c not in ["lesion_id"] + CLASS_COLS]

X_train = train_feat[FEATURE_COLS].copy()
X_test  = test_feat[FEATURE_COLS].copy()
test_ids = test_feat["lesion_id"].values

print(f"Train: {X_train.shape}  Test: {X_test.shape}  Classes: {N_CLASSES}")
print(f"Features: {FEATURE_COLS}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════
CAT_FEATURES = ["sex","site","image_manipulation"]

LGB_PARAMS = dict(
    objective      = "multiclass",
    num_class      = N_CLASSES,
    metric         = "multi_logloss",
    learning_rate  = 0.05,
    num_leaves     = 63,
    max_depth      = -1,
    min_child_samples = 20,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    lambda_l1      = 0.1,
    lambda_l2      = 0.1,
    verbose        = -1,
    seed           = SEED,
    n_jobs         = -1,
)

XGB_PARAMS = dict(
    objective        = "multi:softprob",
    num_class        = N_CLASSES,
    eval_metric      = "mlogloss",
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    tree_method      = "hist",
    seed             = SEED,
    verbosity        = 0,
    n_jobs           = -1,
)

CAT_PARAMS = dict(
    loss_function       = "MultiClass",
    eval_metric         = "MultiClass",
    learning_rate       = 0.05,
    depth               = 6,
    l2_leaf_reg         = 3,
    iterations          = 2000,
    early_stopping_rounds = 100,
    random_seed         = SEED,
    verbose             = 0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def run_lgb(X_tr, y_tr, X_val, y_val, X_te):
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_FEATURES]
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_idx)
    dval= lgb.Dataset(X_val, label=y_val, categorical_feature=cat_idx, reference=dtr)
    model = lgb.train(
        LGB_PARAMS, dtr,
        num_boost_round    = 2000,
        valid_sets         = [dval],
        callbacks          = [lgb.early_stopping(100, verbose=False),
                               lgb.log_evaluation(-1)],
    )
    return model.predict(X_val), model.predict(X_te), model.best_iteration


def run_xgb(X_tr, y_tr, X_val, y_val, X_te):
    # encode categoricals for XGB
    le = {c: LabelEncoder() for c in CAT_FEATURES}
    X_tr  = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    for c, enc in le.items():
        X_tr[c]  = enc.fit_transform(X_tr[c].astype(str))
        X_val[c] = enc.transform(X_val[c].astype(str))
        X_te[c]  = enc.transform(X_te[c].astype(str))
    dtr  = xgb.DMatrix(X_tr,  label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dte  = xgb.DMatrix(X_te)
    model = xgb.train(
        XGB_PARAMS, dtr,
        num_boost_round   = 2000,
        evals             = [(dval, "val")],
        early_stopping_rounds = 100,
        verbose_eval      = False,
    )
    return model.predict(dval), model.predict(dte), model.best_iteration


def run_cat(X_tr, y_tr, X_val, y_val, X_te):
    cat_idx = [X_tr.columns.get_loc(c) for c in CAT_FEATURES]
    model = cb.CatBoostClassifier(cat_features=cat_idx, **CAT_PARAMS)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    return (model.predict_proba(X_val),
            model.predict_proba(X_te),
            model.get_best_iteration())


RUNNERS = {"lgb": run_lgb, "xgb": run_xgb, "cat": run_cat}

results = {}

for model_name, runner in RUNNERS.items():
    print(f"\n{'='*55}")
    print(f"  {model_name.upper()}  — {N_FOLDS}-Fold CV")
    print(f"{'='*55}")

    oof  = np.zeros((len(X_train), N_CLASSES))
    test_preds = np.zeros((len(X_test), N_CLASSES))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        val_pred, te_pred, best_iter = runner(X_tr, y_tr, X_val, y_val, X_test)

        oof[val_idx] = val_pred
        test_preds  += te_pred / N_FOLDS

        score = log_loss(y_val, val_pred)
        fold_scores.append(score)
        print(f"  Fold {fold+1}/{N_FOLDS}  logloss={score:.4f}  best_iter={best_iter}")

    overall = log_loss(y, oof)
    print(f"  OOF logloss: {overall:.4f}")

    results[model_name] = {
        "oof": oof,
        "test": test_preds,
        "cv_score": overall,
        "fold_scores": fold_scores,
    }

    # save OOF + test probs for later DL ensemble
    np.save(f"{OOF_DIR}/oof_{model_name}.npy",  oof)
    np.save(f"{OOF_DIR}/test_{model_name}.npy", test_preds)

    # save lesion order for alignment
    if model_name == "lgb":
        pd.DataFrame({"lesion_id": train_feat["lesion_id"]}).to_csv(
            f"{OOF_DIR}/train_lesion_order.csv", index=False)
        pd.DataFrame({"lesion_id": test_ids}).to_csv(
            f"{OOF_DIR}/test_lesion_order.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. GENERATE SUBMISSIONS
# ═══════════════════════════════════════════════════════════════════════════════
def make_submission(test_probs: np.ndarray, lesion_ids, fname: str):
    df = pd.DataFrame(test_probs, columns=CLASS_COLS)
    df.insert(0, "lesion_id", lesion_ids)
    df.to_csv(fname, index=False)
    print(f"Saved: {fname}")


print(f"\n{'='*55}")
print("  GENERATING SUBMISSIONS")
print(f"{'='*55}")

for model_name, res in results.items():
    make_submission(res["test"], test_ids,
                    f"{SUB_DIR}/submission_{model_name}_tabular.csv")

# ensemble: average of all 3
ensemble_test = np.mean([r["test"] for r in results.values()], axis=0)
make_submission(ensemble_test, test_ids,
                f"{SUB_DIR}/submission_ensemble_tabular.csv")

# best single model
best_model = min(results, key=lambda m: results[m]["cv_score"])
make_submission(results[best_model]["test"], test_ids,
                f"{SUB_DIR}/submission_best_tabular.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("  SUMMARY")
print(f"{'='*55}")
summary = []
for model_name, res in results.items():
    summary.append({
        "model": model_name,
        "oof_logloss": round(res["cv_score"], 5),
        **{f"fold{i+1}": round(s, 5) for i, s in enumerate(res["fold_scores"])},
    })
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(f"{SUB_DIR}/cv_summary.csv", index=False)
print(f"\nBest model: {best_model}  (logloss={results[best_model]['cv_score']:.4f})")
print(f"\nOOF arrays  -> {OOF_DIR}/")
print(f"Submissions -> {SUB_DIR}/")
print("\nDone.")
