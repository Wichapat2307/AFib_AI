"""
test.py
=======
Evaluates ALL trained models on MITB test set.
Run test_xgb.py first to generate xgb_result.json and xgb_probs.npy
"""

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, fbeta_score,
    confusion_matrix, roc_curve,
)
from sklearn.impute import SimpleImputer
from tqdm import tqdm

from dataset import load_mitb_test, ECGDataset
from model import AFibCNN, AFibLSTM, AFibRNN, AFibCNNLSTM

Path("results").mkdir(exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

# ============================================================
# LOAD TEST DATA
# ============================================================

X_test, y_test = load_mitb_test()
print()

# ============================================================
# METRICS
# ============================================================

def compute_metrics(name, y_true, probs, thresh=0.5):
    preds = (probs >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "model":       name,
        "auroc":       float(roc_auc_score(y_true, probs)),
        "f1":          float(f1_score(y_true, preds, zero_division=0)),
        "f2":          float(fbeta_score(y_true, preds, beta=2, zero_division=0)),
        "recall":      float(recall_score(y_true, preds, zero_division=0)),
        "precision":   float(precision_score(y_true, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold":   float(thresh),
    }

# ============================================================
# DEEP MODEL INFERENCE
# ============================================================

def infer_deep(model, X, batch_size=256):
    ds     = ECGDataset(X, np.zeros(len(X), dtype=np.int64), mode="cnn")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference", leave=False):
            x   = batch["signal"].to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)

# ============================================================
# EVALUATE ALL MODELS
# ============================================================

all_results = []
all_probs   = {}

# -- BASELINE -------------------------------------------------
baseline_path = Path("models/Baseline_CNN.pth")
if baseline_path.exists():
    print("Evaluating Baseline CNN...")
    m = AFibCNN().to(DEVICE)
    m.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
    probs = infer_deep(m, X_test)
    r = compute_metrics("Baseline_CNN", y_test, probs)
    all_results.append(r)
    all_probs["Baseline_CNN"] = probs
    print(f"  AUROC={r['auroc']:.4f}  Recall={r['recall']:.4f}  F2={r['f2']:.4f}")
else:
    print("Baseline_CNN not found")

# -- CNN ------------------------------------------------------
cnn_path = Path("models/cnn_best.pth")
if cnn_path.exists():
    print("Evaluating CNN...")
    m = AFibCNN().to(DEVICE)
    m.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    probs = infer_deep(m, X_test)
    r = compute_metrics("CNN", y_test, probs)
    all_results.append(r)
    all_probs["CNN"] = probs
    print(f"  AUROC={r['auroc']:.4f}  Recall={r['recall']:.4f}  F2={r['f2']:.4f}")
else:
    print("CNN not found -- run train_cnn.py first")

# -- LSTM -----------------------------------------------------
lstm_path = Path("models/lstm_best.pth")
if lstm_path.exists():
    print("Evaluating LSTM...")
    m = AFibLSTM().to(DEVICE)
    m.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
    probs = infer_deep(m, X_test, batch_size=32)
    r = compute_metrics("LSTM", y_test, probs)
    all_results.append(r)
    all_probs["LSTM"] = probs
    print(f"  AUROC={r['auroc']:.4f}  Recall={r['recall']:.4f}  F2={r['f2']:.4f}")
else:
    print("LSTM not found -- run train_lstm.py first")

# -- RNN ------------------------------------------------------
rnn_path = Path("models/rnn_best.pth")
if rnn_path.exists():
    print("Evaluating RNN...")
    m = AFibRNN().to(DEVICE)
    m.load_state_dict(torch.load(rnn_path, map_location=DEVICE))
    probs = infer_deep(m, X_test, batch_size=32)
    r = compute_metrics("RNN", y_test, probs)
    all_results.append(r)
    all_probs["RNN"] = probs
    print(f"  AUROC={r['auroc']:.4f}  Recall={r['recall']:.4f}  F2={r['f2']:.4f}")
else:
    print("RNN not found -- run train_rnn.py first")

# -- CNN+LSTM -------------------------------------------------
cnnlstm_path = Path("models/cnn_lstm_best.pth")
if cnnlstm_path.exists():
    print("Evaluating CNN+LSTM...")
    m = AFibCNNLSTM().to(DEVICE)
    m.load_state_dict(torch.load(cnnlstm_path, map_location=DEVICE))
    probs = infer_deep(m, X_test)
    r = compute_metrics("CNN+LSTM", y_test, probs)
    all_results.append(r)
    all_probs["CNN+LSTM"] = probs
    print(f"  AUROC={r['auroc']:.4f}  Recall={r['recall']:.4f}  F2={r['f2']:.4f}")
else:
    print("CNN+LSTM not found -- run train_cnn_lstm.py first")

# -- ML MODELS ------------------------------------------------
hrv_test_path = Path("data/mitb/hrv_X_test.npy")
if hrv_test_path.exists():
    hrv_test = np.load(hrv_test_path)
    hrv_test = np.nan_to_num(hrv_test)

    for name, path in [("RandomForest", "models/rf.pkl"),
                       ("CatBoost",     "models/catboost.pkl")]:
        p = Path(path)
        if p.exists():
            print(f"Evaluating {name}...")
            bundle = joblib.load(p)
            model  = bundle["model"]
            thresh = bundle["threshold"]
            if hasattr(model, 'n_jobs'):
                model.n_jobs = 1
            imp    = bundle.get("imputer", SimpleImputer(strategy="median"))
            X_imp  = imp.transform(hrv_test)
            probs  = model.predict_proba(X_imp)[:, 1]
            r = compute_metrics(name, y_test, probs, thresh)
            all_results.append(r)
            all_probs[name] = probs
            print(f"  AUROC={r['auroc']:.4f}  Recall={r['recall']:.4f}  F2={r['f2']:.4f}")
        else:
            print(f"{name} not found -- run train_ml.py first")
else:
    print("HRV features not found -- run extract_features.py first")

# -- XGB ------------------------------------------------------
# XGBoost is evaluated in a subprocess to avoid macOS segfault.
xgb_path = Path("models/xgb.pkl")
if xgb_path.exists() and hrv_test_path.exists():
    print("Evaluating XGBoost (subprocess)...")
    import subprocess, sys
    xgb_script = """
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, joblib, json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, fbeta_score, confusion_matrix
)
from sklearn.impute import SimpleImputer
Path("results").mkdir(exist_ok=True)
hrv_test = np.load("data/mitb/hrv_X_test.npy")
hrv_test = np.nan_to_num(hrv_test)
y_test   = np.load("data/mitb/y_test.npy")
bundle = joblib.load("models/xgb.pkl")
model  = bundle["model"]
thresh = bundle["threshold"]
model.n_jobs = 1
imp   = bundle.get("imputer", SimpleImputer(strategy="median"))
X_imp = imp.transform(hrv_test)
probs = model.predict_proba(X_imp)[:, 1]
preds = (probs >= thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
result = {
    "model":       "XGBoost",
    "auroc":       float(roc_auc_score(y_test, probs)),
    "f1":          float(f1_score(y_test, preds, zero_division=0)),
    "f2":          float(fbeta_score(y_test, preds, beta=2, zero_division=0)),
    "recall":      float(recall_score(y_test, preds, zero_division=0)),
    "precision":   float(precision_score(y_test, preds, zero_division=0)),
    "sensitivity": float(tp / (tp + fn + 1e-8)),
    "specificity": float(tn / (tn + fp + 1e-8)),
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    "threshold":   float(thresh),
}
np.save("results/xgb_probs.npy", probs)
with open("results/xgb_result.json", "w") as f:
    json.dump(result, f, indent=2)
"""
    proc = subprocess.run(
        [sys.executable, "-c", xgb_script],
        capture_output=True, text=True
    )
    if proc.returncode == 0:
        with open("results/xgb_result.json") as f:
            xgb_result = json.load(f)
        all_results.append(xgb_result)
        all_probs["XGBoost"] = np.load("results/xgb_probs.npy")
        print(f"  AUROC={xgb_result['auroc']:.4f}  Recall={xgb_result['recall']:.4f}  F2={xgb_result['f2']:.4f}")
    else:
        print(f"XGBoost subprocess failed:\n{proc.stderr}")
else:
    print("XGBoost not found -- run train_ml.py first")

# ============================================================
# RESULTS TABLE
# ============================================================

print("\n" + "="*75)
print("FINAL RESULTS -- MITB TEST SET (Cross-Dataset Evaluation)")
print("="*75)
print(f"{'Model':<15} {'AUROC':>7} {'Recall':>7} {'Spec':>7} {'F1':>7} {'F2':>7} {'Prec':>7}")
print("-"*75)
for r in sorted(all_results, key=lambda x: x["auroc"], reverse=True):
    print(f"{r['model']:<15} {r['auroc']:>7.4f} {r['recall']:>7.4f} "
          f"{r['specificity']:>7.4f} {r['f1']:>7.4f} {r['f2']:>7.4f} {r['precision']:>7.4f}")

# ============================================================
# ROC CURVES
# ============================================================

BG     = "#050b12"
PANEL  = "#080f18"
BORDER = "#1a2d3d"

fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
ax.set_facecolor(PANEL)

colors = ["#2ab5b5", "#f04060", "#1fcc7a", "#f4a124", "#c084fc", "#60a5fa", "#fb923c", "#a3e635"]
ax.plot([0,1], [0,1], color=BORDER, linestyle="--", linewidth=1)

for (name, probs), color in zip(all_probs.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

ax.set_xlabel("False Positive Rate", color="#7a9bb8")
ax.set_ylabel("True Positive Rate",  color="#7a9bb8")
ax.set_title("ROC Curves -- All Models (MITB Test Set)", color="white", fontsize=13)
ax.tick_params(colors="#7a9bb8")
ax.legend(facecolor="#0c1620", edgecolor=BORDER, labelcolor="white", fontsize=9)
ax.grid(True, color=BORDER, linewidth=0.5)
for sp in ax.spines.values():
    sp.set_edgecolor(BORDER)

plt.tight_layout()
plt.savefig("results/roc_all_models.png", dpi=150, facecolor=BG)
plt.close()
print("\nSaved results/roc_all_models.png")

# ============================================================
# CONFUSION MATRICES
# ============================================================

n_models = len(all_results)
if n_models > 0:
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), facecolor=BG)
    axes = np.array(axes).flatten()

    for i, r in enumerate(all_results):
        ax = axes[i]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])

        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_facecolor(PANEL)

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "AFib"], color="white")
        ax.set_yticklabels(["Normal", "AFib"], color="white")
        ax.set_xlabel("Predicted", color="#7a9bb8")
        ax.set_ylabel("Actual",    color="#7a9bb8")
        ax.set_title(r["model"],   color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#7a9bb8")

        # Values as plain integers — no scientific notation
        labels = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
        text_colors = [["white", "#f04060"], ["#f04060", "white"]]
        for row_i in range(2):
            for col_j in range(2):
                ax.text(col_j, row_i, f"{labels[row_i][col_j]:,}",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color=text_colors[row_i][col_j])

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("Saved results/confusion_matrices.png")

# ============================================================
# SAVE JSON
# ============================================================

with open("results/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved results/all_results.json")