"""
interactive_ecg_viewer.py
========================

Interactive ECG viewer for:
- MIT-BIH AF Database (AFDB)
- Long-Term AF Database (LTAF)

Features
--------
✓ Slide through the ENTIRE recording
✓ Keyboard controls
✓ Adjustable window size
✓ Automatic filtering
✓ Multi-lead support
✓ Better visualization for noisy Holter ECG

Controls
--------
RIGHT ARROW  -> move forward
LEFT ARROW   -> move backward
UP ARROW     -> zoom in
DOWN ARROW   -> zoom out
1            -> lead 0
2            -> lead 1
q            -> quit

Usage
-----
python src/interactive_ecg_viewer.py
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

AFDB_PATH = Path("mit-bih-atrial-fibrillation-database-1.0.0/files")
LTAF_PATH = Path("long-term-af-database-1.0.0/files")

# Choose dataset + record
DATASET = "ltaf"   # "afdb" or "ltaf"
RECORD_ID = "05"

# Initial viewing window
WINDOW_SECONDS = 5

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):

    nyq = fs / 2

    highcut = min(highcut, nyq * 0.99)

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, signal)

# ─────────────────────────────────────────────
# LOAD RECORD
# ─────────────────────────────────────────────

def load_record(dataset, record_id):

    if dataset.lower() == "afdb":
        db_path = AFDB_PATH
    else:
        db_path = LTAF_PATH

    rec_path = str(db_path / record_id)

    record = wfdb.rdrecord(rec_path)

    print("=" * 50)
    print(f"Dataset : {dataset}")
    print(f"Record  : {record_id}")
    print(f"Channels: {record.sig_name}")
    print(f"Shape   : {record.p_signal.shape}")
    print(f"FS      : {record.fs}")
    print("=" * 50)

    return record

# ─────────────────────────────────────────────
# INTERACTIVE VIEWER
# ─────────────────────────────────────────────

class ECGViewer:

    def __init__(self, record):

        self.record = record

        self.fs = int(record.fs)

        self.signal = record.p_signal

        self.total_samples = len(self.signal)

        self.window_seconds = WINDOW_SECONDS

        self.window_samples = int(self.window_seconds * self.fs)

        self.start = 0

        self.lead = 0

        self.fig, self.ax = plt.subplots(figsize=(16,5))

        self.fig.canvas.mpl_connect(
            'key_press_event',
            self.on_key
        )

        self.update_plot()

        plt.show()

    # ─────────────────────────────────────────
    # DRAW ECG WINDOW
    # ─────────────────────────────────────────

    def update_plot(self):

        self.ax.clear()

        end = self.start + self.window_samples

        end = min(end, self.total_samples)

        segment = self.signal[
            self.start:end,
            self.lead
        ]

        # Better visualization for Holter ECG
        filtered = bandpass_filter(
            segment,
            self.fs,
            lowcut=0.5,
            highcut=25
        )

        time = np.arange(self.start, end) / self.fs

        self.ax.plot(
            time,
            filtered,
            linewidth=1.2
        )

        self.ax.set_title(
            f"Record {self.record.record_name} | "
            f"Lead {self.lead} | "
            f"{self.start/self.fs:.1f}s → "
            f"{end/self.fs:.1f}s | "
            f"Window: {self.window_seconds}s"
)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        self.ax.grid(True)

        # ECG-style autoscaling
        ymin = np.min(filtered)
        ymax = np.max(filtered)

        pad = (ymax - ymin) * 0.2

        self.ax.set_ylim(
            ymin - pad,
            ymax + pad
        )

        self.fig.canvas.draw_idle()

    # ─────────────────────────────────────────
    # KEYBOARD CONTROLS
    # ─────────────────────────────────────────

    def on_key(self, event):

        step = int(self.window_samples * 0.5)

        # move forward
        if event.key == 'right':

            self.start += step

            self.start = min(
                self.start,
                self.total_samples - self.window_samples
            )

        # move backward
        elif event.key == 'left':

            self.start -= step

            self.start = max(self.start, 0)

        # zoom in
        elif event.key == 'up':

            self.window_seconds = max(
                1,
                self.window_seconds - 1
            )

            self.window_samples = int(
                self.window_seconds * self.fs
            )

        # zoom out
        elif event.key == 'down':

            self.window_seconds += 1

            self.window_samples = int(
                self.window_seconds * self.fs
            )

        # lead 0
        elif event.key == '1':

            self.lead = 0

        # lead 1
        elif event.key == '2':

            if self.signal.shape[1] > 1:
                self.lead = 1

        # quit
        elif event.key == 'q':

            plt.close(self.fig)
            return

        self.update_plot()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    record = load_record(
        DATASET,
        RECORD_ID
    )

    viewer = ECGViewer(record)