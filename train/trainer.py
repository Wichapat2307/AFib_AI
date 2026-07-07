"""
trainer.py
==========
Shared training loop for CNN / LSTM / RNN / CNN+LSTM.
Saves training curves after training completes.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, fbeta_score
)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(labels, preds, probs):
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "f2":        fbeta_score(labels, preds, beta=2, zero_division=0),
        "auroc":     roc_auc_score(labels, probs),
    }


# ============================================================
# ONE EPOCH
# ============================================================

def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train() if training else model.eval()

    running_loss = 0
    all_preds, all_probs, all_labels = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        loop = tqdm(loader, leave=False)
        for batch in loop:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)

            if training:
                optimizer.zero_grad()

            out  = model(x)
            loss = criterion(out, y)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)

            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

            loop.set_description(f"loss={loss.item():.4f}")

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = running_loss / len(loader)
    return metrics


# ============================================================
# PLOT CURVES
# ============================================================

def plot_curves(history, model_name, save_dir="results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    BG      = "#050b12"
    PANEL   = "#080f18"
    BORDER  = "#1a2d3d"
    ACCENT  = "#2ab5b5"
    DANGER  = "#f04060"
    SUCCESS = "#1fcc7a"
    WARN    = "#f4a124"
    PURPLE  = "#c084fc"
    TEXT    = "#c8dde8"
    TMID    = "#7a9bb8"

    metrics = [
        ("loss",     "Loss",     ACCENT,   DANGER),
        ("accuracy", "Accuracy", SUCCESS,  WARN),
        ("recall",   "Recall",   ACCENT,   DANGER),
        ("f1",       "F1 Score", SUCCESS,  WARN),
        ("f2",       "F2 Score", PURPLE,   DANGER),
        ("auroc",    "AUROC",    ACCENT,   SUCCESS),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BG)
    axes = axes.flatten()

    fig.suptitle(
        f"{model_name} — Training Curves",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98
    )

    for i, (key, label, train_color, val_color) in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor(PANEL)

        train_vals = history[f"train_{key}"]
        val_vals   = history[f"val_{key}"]

        ax.plot(epochs, train_vals, color=train_color, linewidth=2,
                marker="o", markersize=3, label="Train")
        ax.plot(epochs, val_vals,   color=val_color,   linewidth=2,
                marker="o", markersize=3, label="Val", linestyle="--")

        # Best val marker
        best_idx = int(np.argmax(val_vals) if key != "loss" else np.argmin(val_vals))
        best_val = val_vals[best_idx]
        ax.axvline(best_idx + 1, color=WARN, linewidth=1, linestyle=":", alpha=0.7)
        ax.scatter(best_idx + 1, best_val, color=WARN, s=60, zorder=5)
        ax.annotate(f"{best_val:.3f}", xy=(best_idx + 1, best_val),
                    xytext=(5, 5), textcoords="offset points",
                    color=WARN, fontsize=8)

        ax.set_ylim(0, 1)
        ax.set_title(label, color=TEXT, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", color=TMID, fontsize=9)
        ax.set_ylabel(label,   color=TMID, fontsize=9)
        ax.tick_params(colors=TMID)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.8)
        ax.legend(facecolor="#0c1620", edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    plt.tight_layout()
    out = f"{save_dir}/{model_name.lower().replace('+','_')}_curves.png"
    plt.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ============================================================
# TRAIN
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    save_path,
    model_name="Model",
    epochs=20,
    lr=1e-3,
    patience=5,
):
    device    = get_device()
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    best_f2 = 0
    patience_counter = 0

    # Track all metrics per epoch
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": [],
        "train_f2": [], "val_f2": [],
        "train_auroc": [], "val_auroc": [],
        "train_precision": [], "val_precision": [],
    }

    print(f"\nModel  : {model_name}")
    print(f"Device : {device}")
    print(f"Save   : {save_path}")
    print("=" * 60)

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_m   = run_epoch(model, val_loader,   criterion, None,      device, training=False)

        scheduler.step(val_m["f2"])

        # Store history
        for key in ["loss", "accuracy", "recall", "f1", "f2", "auroc", "precision"]:
            history[f"train_{key}"].append(train_m[key])
            history[f"val_{key}"].append(val_m[key])

        print(f"  TRAIN  loss={train_m['loss']:.4f}  auroc={train_m['auroc']:.4f}  "
              f"f2={train_m['f2']:.4f}  recall={train_m['recall']:.4f}")
        print(f"  VAL    loss={val_m['loss']:.4f}    auroc={val_m['auroc']:.4f}    "
              f"f2={val_m['f2']:.4f}    recall={val_m['recall']:.4f}")

        if val_m["f2"] > best_f2:
            best_f2 = val_m["f2"]
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (F2={best_f2:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\nEarly stopping triggered")
            break

    print(f"\nTraining complete. Best F2={best_f2:.4f}")

    # Plot and save curves
    plot_curves(history, model_name)

    return best_f2