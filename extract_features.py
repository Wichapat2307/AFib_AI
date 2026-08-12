"""
extract_features.py
====================
Extracts 24 HRV features from preprocessed signals.npy files.
Run once — saves hrv_X_train.npy, hrv_X_val.npy, hrv_X_test.npy.

Input:  data/ltaf/X_train.npy, X_val.npy  +  data/mitb/X_test.npy
Output: data/ltaf/hrv_X_train.npy, hrv_X_val.npy
        data/mitb/hrv_X_test.npy
"""

import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from tqdm import tqdm

FS = 128  # sample rate after preprocessing

FEATURE_NAMES = [
    "mean_rr", "median_rr", "sdnn", "rmssd", "pnn50", "cv_rr",
    "mean_hr", "std_hr", "min_hr", "max_hr",
    "mean_diff", "max_diff", "sd1", "sd2", "sd_ratio",
    "skewness", "kurtosis", "iqr", "irr_score",
    "lf_hf", "lf_norm", "hf_norm",
    "dominant_freq", "n_beats",
]


# ============================================================
# R-PEAK DETECTION
# ============================================================

def detect_rpeaks(signal, fs=FS):
    nyq = fs / 2
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype="band")
    sig = filtfilt(b, a, signal)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    if np.abs(sig.min()) > np.abs(sig.max()):
        sig = -sig
    sig_max = float(np.max(sig))
    thr = max(0.3, sig_max * 0.3)
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.3*fs),
                          prominence=sig_max * 0.2)
    if len(peaks) < 3:
        peaks, _ = find_peaks(sig, height=max(0.2, sig_max*0.2),
                              distance=int(0.3*fs))
    return peaks


# ============================================================
# HRV FEATURES
# ============================================================

def extract_hrv(signal, fs=FS):
    peaks = detect_rpeaks(signal, fs)
    rr = np.diff(peaks) / fs * 1000  # ms
    rr = rr[(rr > 250) & (rr < 2000)]

    if len(rr) < 4:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # Time domain
    mean_rr   = float(np.mean(rr))
    median_rr = float(np.median(rr))
    sdnn      = float(np.std(rr))
    rmssd     = float(np.sqrt(np.mean(np.diff(rr)**2)))
    pnn50     = float(np.mean(np.abs(np.diff(rr)) > 50))
    cv_rr     = sdnn / mean_rr if mean_rr > 0 else 0.0
    hr        = 60000.0 / rr
    mean_hr   = float(np.mean(hr))
    std_hr    = float(np.std(hr))
    min_hr    = float(np.min(hr))
    max_hr    = float(np.max(hr))

    # Successive differences
    diff_rr   = np.diff(rr)
    mean_diff = float(np.mean(np.abs(diff_rr)))
    max_diff  = float(np.max(np.abs(diff_rr))) if len(diff_rr) > 0 else 0.0
    sd1       = float(np.sqrt(0.5) * np.std(diff_rr))
    sd2_sq    = max(0, 2 * sdnn**2 - 0.5 * np.std(diff_rr)**2)
    sd2       = float(np.sqrt(sd2_sq))
    sd_ratio  = sd1 / (sd2 + 1e-8)

    # Distribution
    sk  = float(skew(rr))
    ku  = float(kurtosis(rr))
    iqr = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    irr = float(np.sum(np.abs(diff_rr) / (rr[:-1]+1e-9) > 0.10) / max(len(diff_rr),1))

    # Frequency domain
    try:
        t_rr  = np.cumsum(rr) / 1000.0
        t_uni = np.arange(t_rr[0], t_rr[-1], 0.25)
        if len(t_uni) > 8:
            rr_uni = interp1d(t_rr, rr, kind="linear",
                              bounds_error=False, fill_value="extrapolate")(t_uni)
            rr_uni -= np.mean(rr_uni)
            freqs, psd = welch(rr_uni, fs=4.0, nperseg=min(len(rr_uni), 64))
            pos   = freqs > 0
            freqs, psd = freqs[pos], psd[pos]
            lf    = np.sum(psd[(freqs >= 0.04) & (freqs < 0.15)])
            hf    = np.sum(psd[(freqs >= 0.15) & (freqs < 0.40)])
            total = lf + hf + 1e-9
            lf_hf   = lf / (hf + 1e-9)
            lf_norm = lf / total
            hf_norm = hf / total
            dom_freq = float(freqs[np.argmax(psd)])
        else:
            lf_hf = lf_norm = hf_norm = dom_freq = 0.0
    except Exception:
        lf_hf = lf_norm = hf_norm = dom_freq = 0.0

    return np.array([
        mean_rr, median_rr, sdnn, rmssd, pnn50, cv_rr,
        mean_hr, std_hr, min_hr, max_hr,
        mean_diff, max_diff, sd1, sd2, sd_ratio,
        sk, ku, iqr, irr,
        lf_hf, lf_norm, hf_norm,
        dom_freq, float(len(peaks)),
    ], dtype=np.float32)


# ============================================================
# BATCH EXTRACTION
# ============================================================

def extract_batch(signals, desc="Extracting"):
    features = []
    for sig in tqdm(signals, desc=desc):
        features.append(extract_hrv(sig))
    return np.array(features, dtype=np.float32)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Features: {len(FEATURE_NAMES)}")
    print(FEATURE_NAMES)
    print()

    # LTAF train
    ltaf = Path("data/ltaf")
    X_train = np.load(ltaf / "X_train.npy")
    hrv_train = extract_batch(X_train, "LTAF train")
    np.save(ltaf / "hrv_X_train.npy", hrv_train)
    print(f"Saved data/ltaf/hrv_X_train.npy  {hrv_train.shape}")

    # LTAF val
    X_val = np.load(ltaf / "X_val.npy")
    hrv_val = extract_batch(X_val, "LTAF val")
    np.save(ltaf / "hrv_X_val.npy", hrv_val)
    print(f"Saved data/ltaf/hrv_X_val.npy    {hrv_val.shape}")

    # MITB test
    mitb = Path("data/mitb")
    X_test = np.load(mitb / "X_test.npy")
    hrv_test = extract_batch(X_test, "MITB test")
    np.save(mitb / "hrv_X_test.npy", hrv_test)
    print(f"Saved data/mitb/hrv_X_test.npy  {hrv_test.shape}")

    print("\nDone. Run train_rf.py / train_xgb.py / train_catboost.py next.")