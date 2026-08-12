"""
raw.py
======
Raw ECG viewer — both leads, no filtering.
Auto-detects and corrects inverted leads.

Controls:
  ← →   scroll
  ↑ ↓   zoom in/out
  f     toggle flip on Lead 2
  q     quit
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

AFDB_PATH = Path("mit-bih-atrial-fibrillation-database-1.0.0/files")
LTAF_PATH = Path("long-term-af-database-1.0.0/files")

DATASET   = "afdb"
RECORD_ID = "04015"
AUTO_FLIP = True

# ============================================================
# INVERSION DETECTION
# ============================================================

def is_inverted(signal, fs, check_seconds=30):
    n      = min(len(signal), int(check_seconds * fs))
    sample = signal[:n]
    pos    = np.percentile(sample, 98)
    neg    = abs(np.percentile(sample, 2))
    return neg > pos

# ============================================================
# LOAD
# ============================================================

def load_record(dataset, record_id):
    db_path  = AFDB_PATH if dataset == "afdb" else LTAF_PATH
    rec_path = str(db_path / record_id)
    record   = wfdb.rdrecord(rec_path)
    print(f"Loaded  : {record_id}")
    print(f"Shape   : {record.p_signal.shape}")
    print(f"FS      : {record.fs} Hz")
    print(f"Leads   : {record.sig_name}")
    print(f"Duration: {record.p_signal.shape[0]/record.fs/3600:.2f} hours")
    return record

# ============================================================
# VIEWER
# ============================================================

class DualLeadViewer:

    def __init__(self, record):
        self.record         = record
        self.signal         = record.p_signal
        self.fs             = int(record.fs)
        self.n_leads        = self.signal.shape[1]
        self.leads          = record.sig_name if hasattr(record, "sig_name") \
                              else [f"Lead {i}" for i in range(self.n_leads)]
        self.total_samples  = len(self.signal)
        self.window_seconds = 5
        self.window_samples = int(self.window_seconds * self.fs)
        self.start          = 0

        # Auto-detect inverted leads
        if AUTO_FLIP:
            self.flip = []
            for i in range(self.n_leads):
                inv = is_inverted(self.signal[:, i], self.fs)
                self.flip.append(inv)
                status = "INVERTED — flipping" if inv else "normal"
                print(f"  Lead {self.leads[i]}: {status}")
        else:
            self.flip = [False] * self.n_leads

        self.fig, self.axes = plt.subplots(
            self.n_leads, 1,
            figsize=(18, 4 * self.n_leads),
            sharex=True,
        )
        if self.n_leads == 1:
            self.axes = [self.axes]

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.patch.set_facecolor("#f8f9fa")

        self.update()
        plt.tight_layout()
        plt.show()

    def update(self):
        end    = min(self.start + self.window_samples, self.total_samples)
        t      = np.arange(self.start, end) / self.fs
        colors = ["#1a5fa8", "#c0392b"]

        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.set_facecolor("#ffffff")
            ax.grid(True, color="#e0e0e0", linewidth=0.7)

            seg = self.signal[self.start:end, i]
            if self.flip[i]:
                seg = -seg

            ax.plot(t, seg, color=colors[i % len(colors)],
                    linewidth=0.9, label=self.leads[i])

            ax.set_ylabel(f"{self.leads[i]}"
                          f"{'  [FLIPPED]' if self.flip[i] else ''}\n(mV)",
                          fontsize=10)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
            ax.tick_params(labelsize=9)

        self.axes[-1].set_xlabel("Time (s)", fontsize=10)
        self.fig.suptitle(
            f"Record: {self.record.record_name}  |  "
            f"{self.start/self.fs:.1f}s → {end/self.fs:.1f}s  |  "
            f"Window: {self.window_seconds}s  |  "
            f"[← →] scroll   [↑ ↓] zoom   [f] flip Lead 2   [q] quit",
            fontsize=11, fontweight="bold", color="#1a2b3c",
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = int(self.window_samples * 0.5)
        if event.key == "right":
            self.start = min(self.start + step,
                             self.total_samples - self.window_samples)
        elif event.key == "left":
            self.start = max(self.start - step, 0)
        elif event.key == "up":
            self.window_seconds = max(1, self.window_seconds - 1)
            self.window_samples = int(self.window_seconds * self.fs)
        elif event.key == "down":
            self.window_seconds += 1
            self.window_samples = int(self.window_seconds * self.fs)
            if self.start + self.window_samples > self.total_samples:
                self.start = max(0, self.total_samples - self.window_samples)
        elif event.key == "f":
            if self.n_leads > 1:
                self.flip[1] = not self.flip[1]
                print(f"Lead 2 flip: {self.flip[1]}")
        elif event.key == "q":
            plt.close(self.fig)
            return
        self.update()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    record = load_record(DATASET, RECORD_ID)
    viewer = DualLeadViewer(record)