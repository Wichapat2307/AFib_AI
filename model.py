"""
model.py
========
CNN, LSTM, RNN, CNN+LSTM architectures for AFib detection.
Input signal: (batch, 1, 3840)
Output: (batch, 2)
"""

import torch
import torch.nn as nn


# ============================================================
# CNN
# ============================================================

class AFibCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=7, padding=3), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,  64,  kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ============================================================
# LSTM
# ============================================================

class AFibLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (batch, 1, 3840) -> (batch, 3840, 1)
        x = x.squeeze(1).unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(out)


# ============================================================
# RNN
# ============================================================

class AFibRNN(nn.Module):

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (batch, 1, 3840) -> (batch, 3840, 1)
        x = x.squeeze(1).unsqueeze(-1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.classifier(out)


# ============================================================
# CNN + LSTM
# ============================================================

class AFibCNNLSTM(nn.Module):

    def __init__(self, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (batch, 1, 3840)
        x = self.cnn(x)             # (batch, 64, T)
        x = x.permute(0, 2, 1)     # (batch, T, 64)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(out)


# ============================================================
# UTILS
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(8, 1, 3840)
    for cls in [AFibCNN, AFibLSTM, AFibRNN, AFibCNNLSTM]:
        m = cls()
        y = m(x)
        print(f"{cls.__name__:<15} params={count_parameters(m):>8,}  output={y.shape}")