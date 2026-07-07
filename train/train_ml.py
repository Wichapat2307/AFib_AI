"""
train_ml.py
===========
Trains RF, XGBoost, CatBoost on HRV features.
Train/Val: LTAF   Test: MITB
Run extract_features.py first.
"""

import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, fbeta_score, confusion_matrix
)
from sklearn.impute import SimpleImputer

# ============================================================
# LOAD
# ============================================================

print("Loading HRV features...")

X_train = np.load("data/ltaf/hrv_X_train.npy")
y_train = np.load("data/ltaf/y_train.npy")
X_val   = np.load("data/ltaf/hrv_X_val.npy")
y_val   = np.load("data/ltaf/y_val.npy")
X_test  = np.load("data/mitb/hrv_X_test.npy")
y_test  = np.load("data/mitb/y_test.npy")

print(f"Train: {X_train.shape}  AFib={np.sum(y_train==1)}  Normal={np.sum(y_train==0)}")
print(f"Val  : {X_val.shape}    AFib={np.sum(y_val==1)}    Normal={np.sum(y_val==0)}")
print(f"Test : {X_test.shape}   AFib={np.sum(y_test==1)}   Normal={np.sum(y_test==0)}")

# ============================================================
# CLEAN
# ============================================================

imp = SimpleImputer(strategy="median")
X_train = imp.fit_transform(X_train)
X_val   = imp.transform(X_val)
X_test  = imp.transform(X_test)
X_train = np.nan_to_num(X_train)
X_val   = np.nan_to_num(X_val)
X_test  = np.nan_to_num(X_test)



X_res, y_res = X_train, y_train
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ============================================================
# METRICS HELPER
# ============================================================

def evaluate(name, model, X, y, thresh=0.5):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    return {
        "model":       name,
        "auroc":       float(roc_auc_score(y, probs)),
        "f1":          float(f1_score(y, preds, zero_division=0)),
        "f2":          float(fbeta_score(y, preds, beta=2, zero_division=0)),
        "recall":      float(recall_score(y, preds, zero_division=0)),
        "precision":   float(precision_score(y, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def tune_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f2 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        preds = (probs >= t).astype(int)
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t


all_results = []

# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "="*60)
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight={0: 1.0, 1: 2.5},
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
rf.fit(X_res, y_res)
thresh_rf = tune_threshold(rf, X_val, y_val)
print(f"RF threshold tuned: {thresh_rf:.2f}")

joblib.dump({"model": rf, "threshold": thresh_rf, "imputer": imp}, "models/rf.pkl")

res_rf = evaluate("RandomForest", rf, X_test, y_test, thresh_rf)
print(f"RF TEST  auroc={res_rf['auroc']:.4f}  recall={res_rf['recall']:.4f}  f2={res_rf['f2']:.4f}")
all_results.append(res_rf)

# ============================================================
# XGBOOST
# ============================================================

print("\n" + "="*60)
print("Training XGBoost...")

try:
    from xgboost import XGBClassifier

    n_normal = int(np.sum(y_res == 0))
    n_afib   = int(np.sum(y_res == 1))
    scale_pos = n_normal / n_afib  # correct ratio

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )
    xgb.fit(
        X_res, y_res,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    thresh_xgb = tune_threshold(xgb, X_val, y_val)
    joblib.dump({"model": xgb, "threshold": thresh_xgb, "imputer": imp}, "models/xgb.pkl")
    res_xgb = evaluate("XGBoost", xgb, X_test, y_test, thresh_xgb)
    print(f"XGB TEST  auroc={res_xgb['auroc']:.4f}  recall={res_xgb['recall']:.4f}  f2={res_xgb['f2']:.4f}")
    all_results.append(res_xgb)

except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")

# ============================================================
# CATBOOST
# ============================================================

print("\n" + "="*60)
print("Training CatBoost...")

try:
    from catboost import CatBoostClassifier

    cb = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        class_weights={0: 1.0, 1: 2.5},
        random_seed=42,
        verbose=50,
    )
    cb.fit(X_res, y_res, eval_set=(X_val, y_val))
    thresh_cb = tune_threshold(cb, X_val, y_val)
    joblib.dump({"model": cb, "threshold": thresh_cb, "imputer": imp}, "models/catboost.pkl")
    res_cb = evaluate("CatBoost", cb, X_test, y_test, thresh_cb)
    print(f"CatBoost TEST  auroc={res_cb['auroc']:.4f}  recall={res_cb['recall']:.4f}  f2={res_cb['f2']:.4f}")
    all_results.append(res_cb)

except ImportError:
    print("CatBoost not installed. Run: pip install catboost")

# ============================================================
# SAVE RESULTS
# ============================================================

with open("results/ml_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*60)
print("ML RESULTS SUMMARY (MITB TEST SET)")
print("="*60)
print(f"{'Model':<15} {'AUROC':>7} {'Recall':>7} {'F2':>7} {'Spec':>7}")
print("-"*60)
for r in all_results:
    print(f"{r['model']:<15} {r['auroc']:>7.4f} {r['recall']:>7.4f} {r['f2']:>7.4f} {r['specificity']:>7.4f}")

print("\nSaved models/ and results/ml_results.json")