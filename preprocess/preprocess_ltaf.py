"""
preprocess_ltaf.py
==================

LTAF preprocessing with:
- Patient-level splitting
- AFib vs Normal classification
- Rhythm exclusion
- Filtering
- Resampling
- Normalization
"""

import numpy as np
import wfdb
from pathlib import Path
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.model_selection import train_test_split
import json
import datetime
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("long-term-af-database-1.0.0/files")
OUT_PATH = Path("data/ltaf")

TARGET_FS = 128

SEG_LEN = 30
SEG_SAMPLES = SEG_LEN * TARGET_FS

OVERLAP = 0.5

RANDOM_STATE = 42

AFIB_LABELS = {
    "(AFIB","AFIB"
}

NORMAL_LABELS = {
    "(N","N","(NSR","NSR"
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
    "(J", "J",

    "MISSB",
    "PSE",
    "MB",
    "M",
    "MISSR",
    "NOISE",
    "P"
}

EXCLUDED_RECORDS = ["00"]

RECORDS = [
    "01","03","05","06","07","08","10","100","101","102","103","104","105",
    "11","110","111","112","113","114","115","116","117","118","119","12",
    "120","121","122","13","15","16","17","18","19","20","200","201","202",
    "203","204","205","206","207","208","21","22","23","24","25","26","28",
    "30","32","33","34","35","37","38","39","42","43","44","45","47","48",
    "49","51","53","54","55","56","58","60","62","64","65","68","69","70",
    "71","72","74","75"
]

# ============================================================
# FILTER
# ============================================================

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):

    nyq = fs / 2

    highcut = min(highcut, nyq * 0.99)

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, signal)

# ============================================================
# RHYTHM MAP
# ============================================================

def get_rhythm_map(ann, signal_len):

    labels = np.full(signal_len, -1, dtype=np.int8)

    current_label = -1

    for i, sym in enumerate(ann.aux_note):

        sym_clean = sym.strip().rstrip("\x00")

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
# FIX NAN
# ============================================================

def fix_nan(signal):

    sig = signal.copy()

    nans = np.isnan(sig)

    if nans.any():

        idx = np.arange(len(sig))

        sig[nans] = np.interp(
            idx[nans],
            idx[~nans],
            sig[~nans]
        )

    return sig

# ============================================================
# SEGMENT EXTRACTION
# ============================================================

def extract_segments(signal, rhythm_map, fs):

    original_window = int(SEG_LEN * fs)

    step = int(original_window * (1 - OVERLAP))

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

        known = seg_labels[seg_labels != -1]

        if len(known) < len(seg_labels) * 0.8:
            continue

        afib_frac = np.mean(known == 1)

        if afib_frac > 0.8:
            lbl = 1

        elif afib_frac < 0.2:
            lbl = 0

        else:
            continue

        seg = bandpass_filter(seg, fs)

        if fs != TARGET_FS:

            seg = resample_poly(
                seg,
                TARGET_FS,
                int(fs)
            )

        if len(seg) != SEG_SAMPLES:
            continue

        seg = seg.astype(np.float32)

        seg = (
            seg - np.mean(seg)
        ) / (
            np.std(seg) + 1e-8
        )

        segments.append(seg)
        labels.append(lbl)

    return segments, labels

# ============================================================
# MAIN
# ============================================================

def preprocess_ltaf():

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    patient_data = {}

    for patient_id, record_id in enumerate(RECORDS):

        print(f"\n[{patient_id+1}/{len(RECORDS)}] {record_id}")

        rec_path = str(DB_PATH / record_id)

        try:

            rec = wfdb.rdrecord(rec_path)

            ann = wfdb.rdann(rec_path, "atr")

        except Exception as e:

            print(f"SKIPPED: {e}")

            continue

        signal = rec.p_signal[:, 0].astype(np.float32)

        signal = fix_nan(signal)

        fs = rec.fs

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
            "X": np.array(segs, dtype=np.float32),
            "y": np.array(lbls, dtype=np.int8)
        }

        print(
            f"Segments={len(segs)} | "
            f"AFib={sum(np.array(lbls)==1)} | "
            f"Normal={sum(np.array(lbls)==0)}"
        )

    patients = list(patient_data.keys())

    train_patients, val_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    def collect(patient_list):

        X = []
        y = []

        for p in patient_list:

            X.extend(patient_data[p]["X"])

            y.extend(patient_data[p]["y"])

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int8)
        )

    X_train, y_train = collect(train_patients)
    X_val, y_val = collect(val_patients)

    np.save(OUT_PATH / "X_train.npy", X_train)
    np.save(OUT_PATH / "y_train.npy", y_train)

    np.save(OUT_PATH / "X_val.npy", X_val)
    np.save(OUT_PATH / "y_val.npy", y_val)

    summary = {

        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),

        "train_patients": train_patients,
        "val_patients": val_patients,

        "timestamp":
            datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    with open(
        OUT_PATH / "split_summary.json",
        "w"
    ) as f:

        json.dump(summary, f, indent=2)

    print("\nDONE")

if __name__ == "__main__":
    preprocess_ltaf()