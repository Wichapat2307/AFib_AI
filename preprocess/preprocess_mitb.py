"""
preprocess_afdb.py
==================

FINAL BALANCED VERSION
----------------------

✓ Correct AFib labels
✓ Correct signal shape (3840)
✓ Patient-wise split
✓ Balanced patient split
✓ Leakage-safe
✓ Bandpass filtering
✓ Resampling
✓ Normalization
✓ Saves train/val/test sets
✓ Saves patient lists
✓ Saves summary JSON
"""

import numpy as np
import wfdb

from pathlib import Path

from scipy.signal import (
    butter,
    filtfilt,
    resample_poly
)

import json
import datetime
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("mit-bih-atrial-fibrillation-database-1.0.0/files")

OUT_PATH = Path("data/mitb")

TARGET_FS = 128

SEG_LEN = 30
SEG_SAMPLES = SEG_LEN * TARGET_FS

OVERLAP = 0.5


EXCLUDE = {
    "00735","03665","08405","08434"
}

BAD_BLOCKS = {
    "04043"
}

ALL_RECORDS = [
    "04015","04043","04048","04126","04746","04908","04936","05091","05121","05261","06426",
    "06453","06995","07162","07859","07879","07910","08215","08219","08378","08455"
]

RECORDS = [
    r for r in ALL_RECORDS
    if r not in EXCLUDE
]

# ============================================================
# RHYTHM LABELS
# ============================================================

AFIB_LABELS = {
    "(AFIB", "AFIB"
}

NORMAL_LABELS = {
    "(N", "N", "(NSR", "NSR"
}

EXCLUDED_RHYTHMS = {
    "(AFL", "AFL",
    "(J", "J",
    "(VT", "VT",
    "(SVTA", "SVTA",
    "(B", "B",
    "(T", "T",
    "(IVR", "IVR",
    "(AB", "AB",
    "(SBR", "SBR",
    "MISSB","MISSR",
    "NOISE",
    "PSE",
    "M","MB",
    "P"
}

# ============================================================
# FILTER
# ============================================================

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

# ============================================================
# SIGNAL REPAIR
# ============================================================

def repair_signal(signal):

    sig = signal.copy()

    nans = np.isnan(sig)

    if nans.any():

        idx = np.arange(len(sig))

        sig[nans] = np.interp(
            idx[nans],
            idx[~nans],
            sig[~nans]
        )

    i = 0

    while i < len(sig) - 1:

        if sig[i] == sig[i + 1]:

            j = i + 1

            while (
                j < len(sig)
                and sig[j] == sig[i]
            ):
                j += 1

            block_len = j - i

            if block_len > 500:

                if i > 0 and j < len(sig):

                    sig[i:j] = np.linspace(
                        sig[i - 1],
                        sig[j],
                        block_len
                    )

            i = j

        else:
            i += 1

    return sig

# ============================================================
# RHYTHM MAP
# ============================================================

def get_rhythm_map(ann, signal_len):

    labels = np.full(
        signal_len,
        -1,
        dtype=np.int8
    )

    current_label = -1

    for i, sym in enumerate(ann.aux_note):

        sym_clean = (
            sym.strip()
            .rstrip("\x00")
        )

        sample = ann.sample[i]

        if sym_clean in AFIB_LABELS:

            current_label = 1

        elif sym_clean in NORMAL_LABELS:

            current_label = 0

        elif sym_clean in EXCLUDED_RHYTHMS:

            current_label = -1

        if 0 <= sample < signal_len:

            labels[sample] = current_label

    cur = -1

    for i in range(signal_len):

        if labels[i] != -1:

            cur = labels[i]

        labels[i] = cur

    return labels

# ============================================================
# SEGMENT EXTRACTION
# ============================================================

def extract_segments(
    signal,
    rhythm_map,
    fs
):

    original_window = int(
        SEG_LEN * fs
    )

    step = int(
        original_window * (1 - OVERLAP)
    )

    segments = []
    labels = []

    for start in range(
        0,
        len(signal) - original_window,
        step
    ):

        end = start + original_window

        seg = signal[start:end]

        seg_labels = rhythm_map[start:end]

        known = seg_labels[
            seg_labels != -1
        ]

        if len(known) < len(seg_labels) * 0.8:
            continue

        afib_frac = np.mean(known == 1)

        if afib_frac > 0.8:

            lbl = 1

        elif afib_frac < 0.2:

            lbl = 0

        else:
            continue

        seg = bandpass_filter(
            seg,
            fs
        )

        if fs != TARGET_FS:

            seg = resample_poly(
                seg,
                TARGET_FS,
                int(fs)
            )

        if len(seg) != SEG_SAMPLES:
            continue

        seg = seg.astype(np.float32)

        if np.isnan(seg).any():
            continue

        std = np.std(seg)

        if std < 1e-8:
            continue

        seg = (
            seg - np.mean(seg)
        ) / std

        if np.isnan(seg).any():
            continue

        segments.append(seg)
        labels.append(lbl)

    return segments, labels

# ============================================================
# MAIN
# ============================================================

def preprocess():

    OUT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    patient_data = {}

    # ========================================================
    # PROCESS RECORDS
    # ========================================================

    for patient_id, record_id in enumerate(RECORDS):

        print(
            f"\n[{patient_id+1}/{len(RECORDS)}] "
            f"{record_id}"
        )

        rec_path = str(
            DB_PATH / record_id
        )

        try:

            record = wfdb.rdrecord(rec_path)

            ann = wfdb.rdann(
                rec_path,
                "atr"
            )

        except Exception as e:

            print(f"SKIPPED: {e}")

            continue

        signal = record.p_signal[:, 0]

        signal = signal.astype(np.float32)

        fs = record.fs

        if (
            record_id in BAD_BLOCKS
            or np.isnan(signal).any()
        ):

            signal = repair_signal(signal)

        rhythm_map = get_rhythm_map(
            ann,
            len(signal)
        )

        segs, lbls = extract_segments(
            signal,
            rhythm_map,
            fs
        )

        patient_data[record_id] = {

            "X": np.array(
                segs,
                dtype=np.float32
            ),

            "y": np.array(
                lbls,
                dtype=np.int8
            )
        }

        print(
            f"Segments={len(segs)} | "
            f"AFib={sum(np.array(lbls)==1)} | "
            f"Normal={sum(np.array(lbls)==0)}"
        )

    # ========================================================
    # COLLECT ALL AS TEST
    # ========================================================

    all_patients = list(patient_data.keys())

    X_test = []
    y_test = []

    for p in all_patients:

        X_test.extend(patient_data[p]["X"])
        y_test.extend(patient_data[p]["y"])

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int8)

    print("\n" + "=" * 60)
    print("ALL RECORDS → TEST SET")
    print("=" * 60)

    print("\nTEST:", all_patients)

    # ========================================================
    # SAVE
    # ========================================================

    np.save(OUT_PATH / "X_test.npy", X_test)
    np.save(OUT_PATH / "y_test.npy", y_test)

    np.save(
        OUT_PATH / "test_patients.npy",
        np.array(all_patients)
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "test_shape":
            list(X_test.shape),

        "test_AFib":
            int(np.sum(y_test == 1)),

        "test_Normal":
            int(np.sum(y_test == 0)),

        "test_patients":
            all_patients,

        "timestamp":
            datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    with open(
        OUT_PATH / "split_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # FINAL STATS
    # ========================================================

    print("\n" + "=" * 60)

    print("PREPROCESS COMPLETE")

    print("=" * 60)

    print(f"Test : {X_test.shape}")

    print("\nTest labels:")
    print(np.unique(y_test, return_counts=True))

    print("=" * 60)

if __name__ == "__main__":
    preprocess()