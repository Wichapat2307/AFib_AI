"""
colored_ecg_viewer.py
=====================

Interactive colored ECG viewer for:
- MIT-BIH AF Database (AFDB)
- Long-Term AF Database (LTAF)

Features
--------
✓ Slide through the ENTIRE recording
✓ Keyboard controls
✓ Adjustable window size
✓ Automatic filtering
✓ Multi-lead support
✓ Color-coded rhythm annotations
✓ Works for BOTH AFDB and LTAF

Colors
------
Green  = Normal Rhythm
Red    = AFib
Orange = Excluded / Other Rhythms

Controls
--------
RIGHT ARROW  -> move forward
LEFT ARROW   -> move backward
UP ARROW     -> zoom in
DOWN ARROW   -> zoom out
1            -> lead 0
2            -> lead 1
q            -> quit
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

# Choose dataset
DATASET = "ltaf"     # "afdb" or "ltaf"

# Example records:
# AFDB -> "04015"
# LTAF -> "00"

RECORD_ID = "05"

WINDOW_SECONDS = 5

# ─────────────────────────────────────────────
# RHYTHM DEFINITIONS
# ─────────────────────────────────────────────

AFIB_LABELS = {
    "(AFIB",
    "AFIB"
}

NORMAL_LABELS = {
    "(N",
    "N",
    "(NSR",
    "NSR"
}

EXCLUDED_RHYTHMS = {

    "(AFL", "AFL",
    "(VT", "VT",
    "(SVTA", "SVTA",
    "(B", "B",
    "(SBR", "SBR",
    "(T", "T",
    "(IVR", "IVR",
    "(AB", "AB",

    "MISSB",
    "PSE",
    "MB",
    "M"
}

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

def bandpass_filter(
    signal,
    fs,
    lowcut=0.5,
    highcut=40.0,
    order=4
):

    nyq = fs / 2

    highcut = min(highcut, nyq * 0.99)

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return filtfilt(b, a, signal)

# ─────────────────────────────────────────────
# LOAD RECORD + ANNOTATIONS
# ─────────────────────────────────────────────

def load_record(dataset, record_id):

    if dataset.lower() == "mitb":
        db_path = AFDB_PATH

    else:
        db_path = LTAF_PATH

    rec_path = str(db_path / record_id)

    record = wfdb.rdrecord(rec_path)

    ann = wfdb.rdann(rec_path, "atr")

    print("=" * 60)

    print(f"Dataset            : {dataset}")
    print(f"Record             : {record_id}")
    print(f"Channels           : {record.sig_name}")
    print(f"Shape              : {record.p_signal.shape}")
    print(f"Sampling Frequency : {record.fs}")
    print(f"Annotations Loaded : {len(ann.sample)}")

    print("=" * 60)

    return record, ann

# ─────────────────────────────────────────────
# BUILD LABEL ARRAY
# ─────────────────────────────────────────────

def build_labels(signal_length, ann):

    # 0 = Normal
    # 1 = AFib
    # 2 = Excluded / Other

    labels = np.full(
        signal_length,
        2,
        dtype=np.int8
    )

    samples = ann.sample
    notes = ann.aux_note

    current_label = 2

    for i in range(len(samples) - 1):

        start = samples[i]
        end = samples[i + 1]

        note = notes[i]

        # clean weird WFDB strings
        note = (
            note
            .strip()
            .replace("\x00", "")
        )

        # ─────────────────────────
        # AFIB
        # ─────────────────────────

        if any(x in note for x in AFIB_LABELS):

            current_label = 1

        # ─────────────────────────
        # NORMAL
        # ─────────────────────────

        elif any(x in note for x in NORMAL_LABELS):

            current_label = 0

        # ─────────────────────────
        # EXCLUDED / OTHER
        # ─────────────────────────

        elif any(x in note for x in EXCLUDED_RHYTHMS):

            current_label = 2

        labels[start:end] = current_label

    return labels

# ─────────────────────────────────────────────
# ECG VIEWER
# ─────────────────────────────────────────────

class ECGViewer:

    def __init__(self, record, labels):

        self.record = record

        self.fs = int(record.fs)

        self.signal = record.p_signal

        self.labels = labels

        self.total_samples = len(self.signal)

        self.window_seconds = WINDOW_SECONDS

        self.window_samples = int(
            self.window_seconds * self.fs
        )

        self.start = 0

        self.lead = 0

        self.fig, self.ax = plt.subplots(
            figsize=(16, 5)
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key
        )

        self.update_plot()

        plt.show()

    # ─────────────────────────────────────────
    # COLOR MAP
    # ─────────────────────────────────────────

    def get_color(self, label):

        if label == 0:
            return "green"

        elif label == 1:
            return "red"

        else:
            return "orange"

    # ─────────────────────────────────────────
    # DRAW ECG WINDOW
    # ─────────────────────────────────────────

    def update_plot(self):

        self.ax.clear()

        end = min(
            self.start + self.window_samples,
            self.total_samples
        )

        segment = self.signal[
            self.start:end,
            self.lead
        ]

        segment_labels = self.labels[
            self.start:end
        ]

        # ─────────────────────────
        # FILTER ECG
        # ─────────────────────────

        filtered = bandpass_filter(
            segment,
            self.fs,
            lowcut=0.5,
            highcut=25
        )

        # REAL RECORDING TIME

        time = np.arange(
            self.start,
            end
        ) / self.fs

        # ─────────────────────────
        # PLOT COLORED SEGMENTS
        # ─────────────────────────

        i = 0

        while i < len(filtered):

            current_label = segment_labels[i]

            j = i

            while (
                j < len(filtered)
                and segment_labels[j] == current_label
            ):
                j += 1

            self.ax.plot(
                time[i:j],
                filtered[i:j],
                color=self.get_color(current_label),
                linewidth=1.2
            )

            i = j

        # ─────────────────────────
        # TITLES / LABELS
        # ─────────────────────────

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

        # ─────────────────────────
        # ECG AUTOSCALE
        # ─────────────────────────

        ymin = np.min(filtered)
        ymax = np.max(filtered)

        pad = (ymax - ymin) * 0.2

        self.ax.set_ylim(
            ymin - pad,
            ymax + pad
        )

        # ─────────────────────────
        # LEGEND
        # ─────────────────────────

        self.ax.plot(
            [],
            [],
            color="green",
            label="Normal"
        )

        self.ax.plot(
            [],
            [],
            color="red",
            label="AFib"
        )

        self.ax.plot(
            [],
            [],
            color="orange",
            label="Excluded / Other"
        )

        self.ax.legend(loc="upper right")

        self.fig.canvas.draw_idle()

    # ─────────────────────────────────────────
    # KEYBOARD CONTROLS
    # ─────────────────────────────────────────

    def on_key(self, event):

        step = int(self.window_samples * 0.5)

        # move forward

        if event.key == "right":

            self.start += step

            self.start = min(
                self.start,
                self.total_samples
                - self.window_samples
            )

        # move backward

        elif event.key == "left":

            self.start -= step

            self.start = max(self.start, 0)

        # zoom in

        elif event.key == "up":

            self.window_seconds = max(
                1,
                self.window_seconds - 1
            )

            self.window_samples = int(
                self.window_seconds * self.fs
            )

        # zoom out

        elif event.key == "down":

            self.window_seconds += 1

            self.window_samples = int(
                self.window_seconds * self.fs
            )

        # lead 0

        elif event.key == "1":

            self.lead = 0

        # lead 1

        elif event.key == "2":

            if self.signal.shape[1] > 1:
                self.lead = 1

        # quit

        elif event.key == "q":

            plt.close(self.fig)
            return

        self.update_plot()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    record, ann = load_record(
        DATASET,
        RECORD_ID
    )

    labels = build_labels(
        len(record.p_signal),
        ann
    )

    viewer = ECGViewer(
        record,
        labels
    )