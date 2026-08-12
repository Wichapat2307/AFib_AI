"""
dataset.py
==========
Loads LTAF (train/val) and MITB (test) data.
CNN/LSTM: raw signal (N, 3840)
RF/XGB/CatBoost: HRV features (N, n_features)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# ============================================================
# DATASET CLASS
# ============================================================

class ECGDataset(Dataset):

    def __init__(self, X, y, mode="cnn"):
        """
        mode: "cnn"  → adds channel dim (1, 3840)
              "ml"   → flat features (n_features,)
        """
        self.X    = torch.tensor(X, dtype=torch.float32)
        self.y    = torch.tensor(y, dtype=torch.long)
        self.mode = mode

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.mode == "cnn":
            x = x.unsqueeze(0)          # (1, 3840)
        return {"signal": x, "label": y}


# ============================================================
# LOAD SPLITS
# ============================================================

def load_ltaf():
    """Load LTAF train/val splits."""
    base = Path("data/ltaf")

    X_train = np.load(base / "X_train.npy")
    y_train = np.load(base / "y_train.npy")
    X_val   = np.load(base / "X_val.npy")
    y_val   = np.load(base / "y_val.npy")

    print(f"LTAF Train : {X_train.shape}  AFib={np.sum(y_train==1)}  Normal={np.sum(y_train==0)}")
    print(f"LTAF Val   : {X_val.shape}    AFib={np.sum(y_val==1)}    Normal={np.sum(y_val==0)}")

    return X_train, y_train, X_val, y_val


def load_mitb_test():
    base = Path("data/mitb")
    X_test = np.load(base / "X_test.npy")
    y_test = np.load(base / "y_test.npy")
    print(f"MITB Test: {X_test.shape}  AFib={np.sum(y_test==1)}  Normal={np.sum(y_test==0)}")
    return X_test, y_test


# ============================================================
# PYTORCH DATASETS
# ============================================================

def get_dataloaders(batch_size=64):
    from torch.utils.data import DataLoader

    X_train, y_train, X_val, y_val = load_ltaf()
    X_test,  y_test                = load_mitb_test()

    train_ds = ECGDataset(X_train, y_train, mode="cnn")
    val_ds   = ECGDataset(X_val,   y_val,   mode="cnn")
    test_ds  = ECGDataset(X_test,  y_test,  mode="cnn")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_ltaf()
    X_test,  y_test                = load_mitb_test()