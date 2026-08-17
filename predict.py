"""
predict.py — AFibAI ML/HRV pipeline (Streamlit-free)
=====================================================

A standalone copy of the HRV-feature-extraction + model-prediction pipeline
that also lives in app.py. This module has no Streamlit dependency so it can
run inside server.py's background threads (Firebase polling / live
prediction), which are not Streamlit script runs.

This is a deliberate COPY, not an import from app.py: app.py has its own
in-flight edits and this module intentionally doesn't touch it. If the
duplication ever becomes annoying, app.py could be changed later to import
from here instead — out of scope for now.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d

try:
    import xgboost as xgb  # noqa: F401  (availability check only)
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier  # noqa: F401
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

log = logging.getLogger("afibai.predict")

# ── Config ───────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent.resolve()
FS = 128
SLIDE_WIN_SEC = 15.0  # matches app.py's doctor-tab live analysis window

MODEL_PATHS = {
    "Random Forest": str(APP_DIR / "models" / "rf.pkl"),
    "XGBoost":        str(APP_DIR / "models" / "xgb.pkl"),
    "CatBoost":       str(APP_DIR / "models" / "catboost.pkl"),
}

FEATURE_NAMES = [
    "mean_rr", "median_rr", "sdnn", "rmssd", "pnn50", "cv_rr",
    "mean_hr", "std_hr", "min_hr", "max_hr",
    "mean_diff", "max_diff", "sd1", "sd2", "sd_ratio",
    "skewness", "kurtosis", "iqr", "irr_score",
    "lf_hf", "lf_norm", "hf_norm", "dominant_freq", "n_beats",
]


# ── Signal processing ───────────────────────────────────────────────────

def bandpass_filter(signal, fs=FS, low=0.5, high=40.0):
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    min_len = 3 * max(len(a), len(b))
    if len(signal) <= min_len:
        return signal
    return filtfilt(b, a, signal)


def preprocess(signal, fs=FS):
    sig = bandpass_filter(signal, fs)
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)


def detect_rpeaks(signal, fs=FS):
    sig = preprocess(signal, fs)
    if np.abs(sig.min()) > np.abs(sig.max()):
        sig = -sig
    sig_max = float(np.max(sig))
    thr = max(0.3, sig_max * 0.3)
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.3 * fs), prominence=sig_max * 0.2)
    if len(peaks) < 3:
        peaks, _ = find_peaks(sig, height=max(0.2, sig_max * 0.2), distance=int(0.3 * fs))
    return peaks


def estimate_signal_quality(signal, fs=FS):
    """Lightweight heuristic signal-quality score ("GOOD" / "WEAK" / "POOR").
    Not a validated clinical metric — just enough to warn if a recording is
    unusable."""
    duration_s = len(signal) / fs if fs else 0.0
    if duration_s <= 0:
        return "POOR", 0.0
    peaks = detect_rpeaks(signal, fs)
    if len(peaks) < 2:
        return "POOR", 0.0
    rr = np.diff(peaks) / fs * 1000
    valid = rr[(rr > 250) & (rr < 2000)]
    coverage = len(valid) / max(len(rr), 1)
    expected_beats = duration_s * (60.0 / 90.0)  # loose baseline expectation
    detect_ratio = min(len(peaks) / max(expected_beats, 1e-6), 1.5)
    score = coverage * 0.7 + min(detect_ratio, 1.0) * 0.3
    if score >= 0.75:
        return "GOOD", score
    elif score >= 0.45:
        return "WEAK", score
    return "POOR", score


def extract_hrv(signal, fs=FS):
    peaks = detect_rpeaks(signal, fs)
    rr = np.diff(peaks) / fs * 1000
    rr = rr[(rr > 250) & (rr < 2000)]
    if len(rr) < 4:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    mean_rr = float(np.mean(rr)); median_rr = float(np.median(rr))
    sdnn = float(np.std(rr)); rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    pnn50 = float(np.mean(np.abs(np.diff(rr)) > 50))
    cv_rr = sdnn / mean_rr if mean_rr > 0 else 0.0
    hr = 60000.0 / rr
    mean_hr = float(np.mean(hr)); std_hr = float(np.std(hr))
    min_hr = float(np.min(hr)); max_hr = float(np.max(hr))
    diff_rr = np.diff(rr)
    mean_diff = float(np.mean(np.abs(diff_rr)))
    max_diff = float(np.max(np.abs(diff_rr))) if len(diff_rr) > 0 else 0.0
    sd1 = float(np.sqrt(0.5) * np.std(diff_rr))
    sd2_sq = max(0, 2 * sdnn ** 2 - 0.5 * np.std(diff_rr) ** 2)
    sd2 = float(np.sqrt(sd2_sq))
    sd_ratio = sd1 / (sd2 + 1e-8)
    sk = float(skew(rr)); ku = float(kurtosis(rr))
    iqr_val = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    irr = float(np.sum(np.abs(diff_rr) / (rr[:-1] + 1e-9) > 0.10) / max(len(diff_rr), 1))
    try:
        t_rr = np.cumsum(rr) / 1000.0
        t_uni = np.arange(t_rr[0], t_rr[-1], 0.25)
        if len(t_uni) > 8:
            rr_uni = interp1d(t_rr, rr, kind="linear",
                               bounds_error=False, fill_value="extrapolate")(t_uni)
            rr_uni -= np.mean(rr_uni)
            freqs, psd = welch(rr_uni, fs=4.0, nperseg=min(len(rr_uni), 64))
            pos = freqs > 0; freqs, psd = freqs[pos], psd[pos]
            lf = np.sum(psd[(freqs >= 0.04) & (freqs < 0.15)])
            hf = np.sum(psd[(freqs >= 0.15) & (freqs < 0.40)])
            tot = lf + hf + 1e-9
            lf_hf = lf / (hf + 1e-9); lf_norm = lf / tot; hf_norm = hf / tot
            dom_freq = float(freqs[np.argmax(psd)])
        else:
            lf_hf = lf_norm = hf_norm = dom_freq = 0.0
    except Exception:
        lf_hf = lf_norm = hf_norm = dom_freq = 0.0
    return np.array([
        mean_rr, median_rr, sdnn, rmssd, pnn50, cv_rr,
        mean_hr, std_hr, min_hr, max_hr,
        mean_diff, max_diff, sd1, sd2, sd_ratio,
        sk, ku, iqr_val, irr,
        lf_hf, lf_norm, hf_norm, dom_freq, float(len(peaks)),
    ], dtype=np.float32)


# ── Model loading ────────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}


def _load_pkl(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        if JOBLIB_AVAILABLE:
            return joblib.load(str(p))
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log.warning("Failed to load %s: %s", path, e)
        return None


def _cached_load(path: str):
    if path not in _model_cache:
        _model_cache[path] = _load_pkl(path)
    return _model_cache[path]


def load_rf_model(path: str):
    return _cached_load(path)


def load_xgb_model(path: str):
    if not XGB_AVAILABLE:
        return None
    return _cached_load(path)


def load_catboost_model(path: str):
    if not CATBOOST_AVAILABLE:
        return None
    return _cached_load(path)


def _unpack_model(model):
    if isinstance(model, dict):
        return model["model"], model.get("imputer"), float(model.get("threshold", 0.5))
    return model, None, 0.5


def _predict_with(model, features):
    clf, imputer, threshold = _unpack_model(model)
    x = features.reshape(1, -1)
    if imputer is not None:
        x = imputer.transform(x)
    prob = float(clf.predict_proba(x)[0][1])
    return ("AFib" if prob >= threshold else "Normal"), prob, threshold


def predict_rf(model, features):
    return _predict_with(model, features)


def predict_xgb(model, features):
    return _predict_with(model, features)


def predict_catboost(model, features):
    return _predict_with(model, features)


def hrv_heuristic(features):
    feat = dict(zip(FEATURE_NAMES, features))
    score = 0.0; reasons = {}
    sdnn_n = min(feat["sdnn"] / 150.0, 1.0); reasons["SDNN (variability)"] = sdnn_n; score += sdnn_n * 0.25
    pnn50_n = min(feat["pnn50"] / 0.5, 1.0); reasons["pNN50"] = pnn50_n; score += pnn50_n * 0.20
    irr_n = min(feat["irr_score"] / 0.4, 1.0); reasons["Irregularity score"] = irr_n; score += irr_n * 0.25
    rmssd_n = min(feat["rmssd"] / 120.0, 1.0); reasons["RMSSD"] = rmssd_n; score += rmssd_n * 0.15
    lfhf_i = max(0.0, 1.0 - min(feat["lf_hf"] / 2.0, 1.0)); reasons["LF/HF imbalance"] = lfhf_i; score += lfhf_i * 0.10
    cv_n = min(feat["cv_rr"] / 0.25, 1.0); reasons["CV of RR"] = cv_n; score += cv_n * 0.05
    prob = float(np.clip(score, 0.0, 1.0))
    threshold = 0.45
    return ("AFib" if prob >= threshold else "Normal"), prob, threshold, reasons


def run_prediction(model_choice, features, silent=False):
    """Runs the selected model (or falls back to the HRV heuristic) on a
    single feature vector.

    Returns: label, prob, threshold, method_note, reasons(dict|None), individual_preds
    """
    reasons = None
    individual_preds = {}

    def _warn(msg):
        if not silent:
            log.warning(msg)

    if model_choice == "Random Forest":
        mdl = load_rf_model(MODEL_PATHS["Random Forest"])
        if mdl is None:
            _warn(f"Random Forest model not found at {MODEL_PATHS['Random Forest']}. "
                  "Falling back to HRV heuristic.")
            label, prob, threshold, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (Random Forest weights missing)"
        else:
            label, prob, threshold = predict_rf(mdl, features)
            method_note = f"Random Forest — {MODEL_PATHS['Random Forest']}"

    elif model_choice == "XGBoost":
        mdl = load_xgb_model(MODEL_PATHS["XGBoost"])
        if mdl is None:
            _warn(f"XGBoost model not found at {MODEL_PATHS['XGBoost']}. "
                  "Falling back to HRV heuristic.")
            label, prob, threshold, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (XGBoost weights missing)"
        else:
            label, prob, threshold = predict_xgb(mdl, features)
            method_note = f"XGBoost — {MODEL_PATHS['XGBoost']}"

    elif model_choice == "CatBoost":
        mdl = load_catboost_model(MODEL_PATHS["CatBoost"])
        if mdl is None:
            _warn(f"CatBoost model not found at {MODEL_PATHS['CatBoost']}. "
                  "Falling back to HRV heuristic.")
            label, prob, threshold, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (CatBoost weights missing)"
        else:
            label, prob, threshold = predict_catboost(mdl, features)
            method_note = f"CatBoost — {MODEL_PATHS['CatBoost']}"

    elif model_choice == "Ensemble":
        rf_model = load_rf_model(MODEL_PATHS["Random Forest"])
        xgb_model = load_xgb_model(MODEL_PATHS["XGBoost"])
        cat_model = load_catboost_model(MODEL_PATHS["CatBoost"])

        probs = []
        if rf_model is not None:
            _, p, _ = predict_rf(rf_model, features)
            probs.append(p); individual_preds["Random Forest"] = p
        if xgb_model is not None:
            _, p, _ = predict_xgb(xgb_model, features)
            probs.append(p); individual_preds["XGBoost"] = p
        if cat_model is not None:
            _, p, _ = predict_catboost(cat_model, features)
            probs.append(p); individual_preds["CatBoost"] = p

        if len(probs) == 0:
            _warn("None of the ensemble model weights were found "
                  f"({MODEL_PATHS['Random Forest']}, {MODEL_PATHS['XGBoost']}, "
                  f"{MODEL_PATHS['CatBoost']}). Falling back to HRV heuristic.")
            label, prob, threshold, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (no ensemble models found)"
        else:
            prob = float(np.mean(probs))
            threshold = 0.3
            label = "AFib" if prob >= threshold else "Normal"
            method_note = f"Ensemble — mean of {len(probs)}/3 models"

    else:
        label, prob, threshold, reasons = hrv_heuristic(features)
        method_note = "HRV heuristic"

    return label, prob, threshold, method_note, reasons, individual_preds


def compute_quick_result(signal, fs=FS, model_choice="Ensemble"):
    """Run HRV extraction + prediction on a signal and return a flat dict.
    Silent by default (no warning spam) — mirrors app.py's patient-facing
    compute_quick_result."""
    features = extract_hrv(signal, fs)
    feat = dict(zip(FEATURE_NAMES, features))
    peaks = detect_rpeaks(signal, fs)
    label, prob, threshold, method_note, reasons, individual_preds = run_prediction(
        model_choice, features, silent=True
    )
    if reasons is None:
        _, _, _, reasons = hrv_heuristic(features)
    quality, quality_score = estimate_signal_quality(signal, fs)
    return {
        "label": label,
        "prob": float(prob),
        "threshold": float(threshold),
        "method_note": method_note,
        "reasons": {k: float(v) for k, v in reasons.items()} if reasons else reasons,
        "hr": float(feat.get("mean_hr", 0.0)),
        "rmssd": float(feat.get("rmssd", 0.0)),
        "sdnn": float(feat.get("sdnn", 0.0)),
        "quality": quality,
        "quality_score": float(quality_score),
        "peaks": peaks,
        "n_peaks": len(peaks),
    }
