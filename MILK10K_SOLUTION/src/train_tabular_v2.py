"""
Tabular ML pipeline v2 — MILK10k
- LGB, XGB, MLP : 5-fold CV + SMOTE + class weights
- CatBoost       : train 1 lần toàn bộ data (nặng, chỉ lấy test predictions)
- Ensemble       : weighted average by OOF macro F1 (CatBoost dùng train F1)
- Temperature scaling trên ensemble OOF
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from tabular_features import build_features, CAT_FEATURE_NAMES

# ── dirs ───────────────────────────────────────────────────────────────────────
OOF_DIR = "outputs/tabular/oof"
SUB_DIR = "outputs/tabular/submissions"
os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

CLASS_COLS = ["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]
N_CLASSES  = len(CLASS_COLS)
N_FOLDS    = 5
SEED       = 42
SMOTE_MIN  = 100

# ── load & engineer features ───────────────────────────────────────────────────
print("Loading & engineering features...")
train_meta = pd.read_csv("datasets/MILK10k/train/MILK10k_Training_Metadata.csv")
test_meta  = pd.read_csv("datasets/MILK10k/test/MILK10k_Test_Metadata.csv")
gt         = pd.read_csv("datasets/MILK10k/train/MILK10k_Training_GroundTruth.csv")

train_feat = build_features(train_meta)
test_feat  = build_features(test_meta)
train_feat = train_feat.merge(gt, on="lesion_id", how="left")

y        = train_feat[CLASS_COLS].values.argmax(axis=1)
test_ids = test_feat["lesion_id"].values

FEATURE_COLS = [c for c in train_feat.columns
                if c not in ["lesion_id"] + CLASS_COLS]
X_train = train_feat[FEATURE_COLS].copy()
X_test  = test_feat[FEATURE_COLS].copy()
print(f"Train: {X_train.shape}  Test: {X_test.shape}  Features: {len(FEATURE_COLS)}")

# ── class weights ──────────────────────────────────────────────────────────────
class_counts      = np.bincount(y, minlength=N_CLASSES).astype(float)
class_weights_arr = np.clip(class_counts.max() / (class_counts + 1e-6), 1.0, 10.0)
print("\nClass weights:", {c: round(w, 2) for c, w in zip(CLASS_COLS, class_weights_arr)})

def sample_weights(y_arr):
    return np.array([class_weights_arr[c] for c in y_arr])

def smote(X_df, y_arr, seed):
    counts   = np.bincount(y_arr, minlength=N_CLASSES)
    strategy = {i: max(counts[i], SMOTE_MIN) for i in range(N_CLASSES) if counts[i] > 0}
    try:
        X_res, y_res = SMOTE(sampling_strategy=strategy, k_neighbors=3,
                             random_state=seed, n_jobs=-1).fit_resample(
                             X_df.values.astype(float), y_arr)
        return pd.DataFrame(X_res, columns=X_df.columns), y_res
    except Exception:
        return X_df, y_arr

# ── model configs ──────────────────────────────────────────────────────────────
LGB_PARAMS = dict(
    objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
    learning_rate=0.05, num_leaves=127, max_depth=7,
    min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, lambda_l1=0.1, lambda_l2=0.1,
    verbose=-1, seed=SEED, n_jobs=-1,
)

XGB_PARAMS = dict(
    objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
    learning_rate=0.05, max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    tree_method="hist", seed=SEED, verbosity=0, n_jobs=-1,
)

CAT_PARAMS = dict(
    loss_function="MultiClass", eval_metric="MultiClass",
    learning_rate=0.05, depth=6, l2_leaf_reg=3,
    iterations=1000, random_seed=SEED, verbose=0,
    class_weights=list(class_weights_arr),
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5-FOLD CV — LGB, XGB, MLP
# ═══════════════════════════════════════════════════════════════════════════════
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results = {}

for model_name in ["lgb", "xgb", "mlp"]:
    print(f"\n{'='*55}")
    print(f"  {model_name.upper()} — {N_FOLDS}-Fold CV (SMOTE + class weights)")
    print(f"{'='*55}")

    oof        = np.zeros((len(X_train), N_CLASSES))
    test_preds = np.zeros((len(X_test),  N_CLASSES))
    fold_f1    = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        Xtr, Xval = X_train.iloc[tr_idx].copy(), X_train.iloc[val_idx].copy()
        ytr, yval = y[tr_idx], y[val_idx]

        # SMOTE trên fold train (tất cả models)
        Xtr, ytr = smote(Xtr, ytr, SEED + fold)

        Xtr_np  = Xtr.values.astype(float)
        Xval_np = Xval.values.astype(float)
        Xte_np  = X_test.values.astype(float)

        if model_name == "lgb":
            sw   = sample_weights(ytr)
            dtr  = lgb.Dataset(Xtr_np, label=ytr, weight=sw)
            dval = lgb.Dataset(Xval_np, label=yval, reference=dtr)
            m    = lgb.train(LGB_PARAMS, dtr, num_boost_round=1000,
                             valid_sets=[dval],
                             callbacks=[lgb.early_stopping(50, verbose=False),
                                        lgb.log_evaluation(-1)])
            vp = m.predict(Xval_np)
            tp = m.predict(Xte_np)

        elif model_name == "xgb":
            sw   = sample_weights(ytr)
            dtr  = xgb.DMatrix(Xtr_np, label=ytr, weight=sw)
            dval = xgb.DMatrix(Xval_np, label=yval)
            m    = xgb.train(XGB_PARAMS, dtr, num_boost_round=1000,
                             evals=[(dval, "val")],
                             early_stopping_rounds=50, verbose_eval=False)
            vp = m.predict(xgb.DMatrix(Xval_np))
            tp = m.predict(xgb.DMatrix(Xte_np))

        elif model_name == "mlp":
            scaler = StandardScaler()
            Xtr_s  = scaler.fit_transform(Xtr_np)
            Xval_s = scaler.transform(Xval_np)
            Xte_s  = scaler.transform(Xte_np)
            m = MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                              alpha=0.01, learning_rate_init=5e-4,
                              max_iter=300, early_stopping=True,
                              n_iter_no_change=20, random_state=SEED)
            m.fit(Xtr_s, ytr)
            vp = m.predict_proba(Xval_s)
            tp = m.predict_proba(Xte_s)

        oof[val_idx]  = vp
        test_preds   += tp / N_FOLDS
        f1 = f1_score(yval, vp.argmax(axis=1), average="macro")
        fold_f1.append(f1)
        print(f"  Fold {fold+1}/{N_FOLDS}  F1={f1:.4f}")

    oof_f1 = f1_score(y, oof.argmax(axis=1), average="macro")
    print(f"  OOF macro-F1: {oof_f1:.4f}")
    results[model_name] = {"oof": oof, "test": test_preds,
                           "oof_f1": oof_f1, "fold_f1": fold_f1}
    np.save(f"{OOF_DIR}/oof_{model_name}_v2.npy",  oof)
    np.save(f"{OOF_DIR}/test_{model_name}_v2.npy", test_preds)

# ═══════════════════════════════════════════════════════════════════════════════
# CATBOOST — train 1 time on full data
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  CATBOOST — Train 1 time (full data, no CV)")
print(f"{'='*55}")

X_all_np = X_train.values.astype(float)
X_te_np  = X_test.values.astype(float)
X_all_sm, y_all_sm = smote(X_train, y, SEED)

cat_model = cb.CatBoostClassifier(**CAT_PARAMS)
cat_model.fit(X_all_sm.values.astype(float), y_all_sm)

cat_train_pred = cat_model.predict_proba(X_all_np)
cat_test_pred  = cat_model.predict_proba(X_te_np)
cat_train_f1   = f1_score(y, cat_train_pred.argmax(axis=1), average="macro")
print(f"  Train macro-F1 (full): {cat_train_f1:.4f}  (not OOF - reference only)")

results["cat"] = {"oof": cat_train_pred, "test": cat_test_pred,
                  "oof_f1": cat_train_f1 * 0.85,  # discount vì không phải OOF
                  "fold_f1": []}
np.save(f"{OOF_DIR}/oof_cat_v2.npy",  cat_train_pred)
np.save(f"{OOF_DIR}/test_cat_v2.npy", cat_test_pred)

# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE + TEMPERATURE SCALING
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_ORDER = ["lgb", "xgb", "mlp", "cat"]
weights     = np.array([results[m]["oof_f1"] for m in MODEL_ORDER])
weights    /= weights.sum()
print(f"\nEnsemble weights:")
for m, w in zip(MODEL_ORDER, weights):
    print(f"  {m}: {w:.3f}  (F1={results[m]['oof_f1']:.4f})")

ens_oof  = sum(w * results[m]["oof"]  for w, m in zip(weights, MODEL_ORDER))
ens_test = sum(w * results[m]["test"] for w, m in zip(weights, MODEL_ORDER))

# temperature scaling trên OOF (chỉ dùng 3 model CV thực sự)
oof_for_T = sum(results[m]["oof"] * results[m]["oof_f1"]
                for m in ["lgb","xgb","mlp"])
oof_for_T /= sum(results[m]["oof_f1"] for m in ["lgb","xgb","mlp"])

best_T, best_f1 = 1.0, 0.0
for T in np.arange(0.3, 2.1, 0.05):
    scaled = np.exp(np.log(oof_for_T + 1e-9) / T)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    f1 = f1_score(y, scaled.argmax(axis=1), average="macro")
    if f1 > best_f1:
        best_f1, best_T = f1, T

print(f"\nTemperature scaling: T={best_T:.2f}  OOF F1={best_f1:.4f}")
ens_test_scaled = np.exp(np.log(ens_test + 1e-9) / best_T)
ens_test_scaled = ens_test_scaled / ens_test_scaled.sum(axis=1, keepdims=True)

np.save(f"{OOF_DIR}/oof_ensemble_v2.npy",  ens_oof)
np.save(f"{OOF_DIR}/test_ensemble_v2.npy", ens_test_scaled)
json.dump({"temperature": float(best_T),
           "weights": dict(zip(MODEL_ORDER, weights.tolist()))},
          open(f"{OOF_DIR}/ensemble_config.json","w"), indent=2)

# ── lesion order ───────────────────────────────────────────────────────────────
pd.DataFrame({"lesion_id": train_feat["lesion_id"]}).to_csv(
    f"{OOF_DIR}/train_lesion_order.csv", index=False)
pd.DataFrame({"lesion_id": test_ids}).to_csv(
    f"{OOF_DIR}/test_lesion_order.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SUBMISSIONS
# ═══════════════════════════════════════════════════════════════════════════════
def save_sub(probs, ids, path):
    df = pd.DataFrame(probs, columns=CLASS_COLS)
    df.insert(0, "lesion_id", ids)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

print(f"\n{'='*55}\n  SUBMISSIONS\n{'='*55}")
for m in MODEL_ORDER:
    save_sub(results[m]["test"], test_ids, f"{SUB_DIR}/submission_{m}_v2.csv")
save_sub(ens_test_scaled, test_ids, f"{SUB_DIR}/submission_ensemble_v2.csv")

best_cv_m = max(["lgb","xgb","mlp"], key=lambda m: results[m]["oof_f1"])
save_sub(results[best_cv_m]["test"], test_ids, f"{SUB_DIR}/submission_best_v2.csv")

# ── summary ────────────────────────────────────────────────────────────────────
rows = []
for m in MODEL_ORDER:
    r = results[m]
    row = {"model": m, "oof_f1": round(r["oof_f1"], 5),
           "cv": "5-fold" if m != "cat" else "1-shot"}
    if r["fold_f1"]:
        for i, v in enumerate(r["fold_f1"]):
            row[f"fold{i+1}"] = round(v, 5)
    rows.append(row)
rows.append({"model": "ensemble", "oof_f1": round(best_f1, 5), "cv": "—"})

summary = pd.DataFrame(rows)
summary.to_csv(f"{SUB_DIR}/cv_summary_v2.csv", index=False)

print(f"\n{'='*55}\n  SUMMARY\n{'='*55}")
print(summary[["model","cv","oof_f1"]].to_string(index=False))
print(f"\nBest CV model : {best_cv_m}  (F1={results[best_cv_m]['oof_f1']:.4f})")
print(f"Ensemble (T={best_T:.2f}): F1={best_f1:.4f}")
print(f"\nOOF  -> {OOF_DIR}/")
print(f"Subs -> {SUB_DIR}/")
print("\nDone.")
