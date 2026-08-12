"""
processed.py
============
Interactive ECG viewer — both leads, bandpass filtered.
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
from scipy.signal import butter, filtfilt

# ============================================================
# CONFIG
# ============================================================

AFDB_PATH = Path("mit-bih-atrial-fibrillation-database-1.0.0/files")
LTAF_PATH = Path("long-term-af-database-1.0.0/files")

DATASET        = "afdb"
RECORD_ID      = "04015"
WINDOW_SECONDS = 5
AUTO_FLIP      = True

# ============================================================
# FILTER
# ============================================================

def bandpass_filter(signal, fs, lowcut=0.5, highcut=25.0, order=4):
    nyq     = fs / 2
    highcut = min(highcut, nyq * 0.99)
    b, a    = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, signal)

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
    db_path  = AFDB_PATH if dataset.lower() == "afdb" else LTAF_PATH
    rec_path = str(db_path / record_id)
    record   = wfdb.rdrecord(rec_path)
    print("=" * 50)
    print(f"Dataset : {dataset}")
    print(f"Record  : {record_id}")
    print(f"Channels: {record.sig_name}")
    print(f"Shape   : {record.p_signal.shape}")
    print(f"FS      : {record.fs} Hz")
    print(f"Duration: {record.p_signal.shape[0]/record.fs/3600:.2f} hours")
    print("=" * 50)
    return record

# ============================================================
# VIEWER
# ============================================================

class ECGViewer:

    def __init__(self, record):
        self.record         = record
        self.fs             = int(record.fs)
        self.signal         = record.p_signal
        self.n_leads        = self.signal.shape[1]
        self.leads          = record.sig_name if hasattr(record, "sig_name") \
                              else [f"Lead {i}" for i in range(self.n_leads)]
        self.total_samples  = len(self.signal)
        self.window_seconds = WINDOW_SECONDS
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

        self.fig.patch.set_facecolor("#f8f9fa")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update_plot()
        plt.tight_layout()
        plt.show()

    def update_plot(self):
        end    = min(self.start + self.window_samples, self.total_samples)
        t      = np.arange(self.start, end) / self.fs
        colors = ["#1a5fa8", "#c0392b"]

        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.set_facecolor("#ffffff")
            ax.grid(True, color="#e0e0e0", linewidth=0.7)

            seg      = self.signal[self.start:end, i]
            filtered = bandpass_filter(seg, self.fs)
            if self.flip[i]:
                filtered = -filtered

            ax.plot(t, filtered, color=colors[i % len(colors)], linewidth=1.0)

            ymin = np.min(filtered)
            ymax = np.max(filtered)
            pad  = (ymax - ymin) * 0.2 if (ymax - ymin) > 0 else 0.5
            ax.set_ylim(ymin - pad, ymax + pad)

            flip_tag = " [FLIPPED]" if self.flip[i] else ""
            ax.set_ylabel(f"{self.leads[i]}{flip_tag}\n(mV)", fontsize=10)
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
        self.update_plot()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    record = load_record(DATASET, RECORD_ID)
    viewer = ECGViewer(record)