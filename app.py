"""
app.py — AFib Detection Web App
================================
Streamlit app for detecting Atrial Fibrillation from ECG signals.
Restyled to match AFibAI aesthetic.

Models supported:
  - Random Forest (HRV features — loads models/rf.pkl)
  - XGBoost  (HRV features — loads models/xgb.pkl)
  - CatBoost (HRV features — loads models/catboost.pkl)
  - Ensemble (mean of all available model probabilities)
  - HRV heuristic (no trained model required — always available, used as fallback)

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path
import io, pickle, time, json, urllib.request, urllib.error, base64
from datetime import datetime, timezone

import auth  # real SQLite + bcrypt auth (replaces the mock login)

# Initialize database + seed the admin account (idempotent).
auth.init_db()

# ── Live server config ──────────────────────────────────────────────────────
# The Wio (and the mock) POST samples to server.py; this app polls server.py
# for live data and recording history. Set AFIBAI_SERVER_URL to override the
# default (e.g. when running on a deployed instance).
AFIBAI_SERVER_URL = os.environ.get("AFIBAI_SERVER_URL", "http://127.0.0.1:5000")
DEFAULT_DEVICE_ID = os.environ.get("AFIBAI_DEVICE_ID", "wio_01")

def _server_get(path: str, timeout: float = 4.0):
    """GET a JSON endpoint from server.py. Returns parsed JSON or None on error."""
    try:
        with urllib.request.urlopen(AFIBAI_SERVER_URL + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def _server_post(path: str, body: dict, timeout: float = 4.0):
    """POST JSON to server.py. Returns parsed response or None on error."""
    try:
        req = urllib.request.Request(
            AFIBAI_SERVER_URL + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AFibAI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════
COLORS = {
    "bg":           "#f6faf7",
    "panel":        "#ffffff",
    "panel2":       "#eef6f1",
    "border":       "#d9e8dd",
    "border_light": "#bfe0cb",
    "text":         "#173023",
    "text_mid":     "#5b7568",
    "text_dim":     "#93aa9c",
    "accent":       "#16a34a",
    "accent2":      "#0d9488",
    "white":        "#ffffff",
    "success":      "#15803d",
    "danger":       "#dc2626",
    "warn":         "#d97706",
    "ecg_bg":       "#fffdf8",
    "ecg_grid_maj": "rgba(210,50,50,0.22)",
    "ecg_grid_min": "rgba(210,50,50,0.08)",
    "ecg_normal":   "#0d9488",
    "ecg_afib":     "#dc2626",
}

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Sora:wght@600;700&display=swap');
  * {{ box-sizing: border-box; }}
  html, body, :root {{ color-scheme: light !important; }}
  input, select, button {{ color-scheme: light !important; }}
  input[type="radio"], input[type="checkbox"] {{ accent-color: {COLORS["accent"]} !important; color-scheme: light !important; }}

  /* Streamlit/BaseWeb radios & checkboxes are custom-drawn (not native inputs),
     so accent-color above never reaches them — this is the "red dot leak".
     Override every layer of the custom control explicitly, scoped to the whole
     app (not just the sidebar) so it can't leak on any page. */
  [data-baseweb="radio"] div[aria-checked="true"] {{ border-color: {COLORS["accent"]} !important; }}
  [data-baseweb="radio"] div[aria-checked="true"] > div {{ background: {COLORS["accent"]} !important; border-color: {COLORS["accent"]} !important; }}
  [data-baseweb="radio"] svg {{ fill: {COLORS["accent"]} !important; color: {COLORS["accent"]} !important; }}
  [role="radiogroup"] div[aria-checked="true"] {{ border-color: {COLORS["accent"]} !important; }}
  [role="radiogroup"] div[aria-checked="true"] > div {{ background: {COLORS["accent"]} !important; }}
  [data-baseweb="checkbox"] div[aria-checked="true"] {{ background: {COLORS["accent"]} !important; border-color: {COLORS["accent"]} !important; }}
  [data-baseweb="checkbox"] svg {{ fill: {COLORS["white"]} !important; }}
  .stApp {{ background: {COLORS["bg"]}; font-family: 'Inter', sans-serif; color: {COLORS["text"]}; color-scheme: light !important; }}
  .main .block-container {{ padding: 1.5rem 2rem !important; max-width: 100% !important; }}

  [data-testid="stSidebar"] {{ background: {COLORS["panel"]} !important; border-right: 1px solid {COLORS["border"]} !important; }}
  [data-testid="stSidebar"] * {{ color: {COLORS["text"]} !important; }}
  [data-testid="stSidebar"] hr {{ border-color: {COLORS["border"]} !important; }}
  [data-testid="stSidebar"] .stRadio label span {{ font-size: 0.83rem !important; color: {COLORS["text_mid"]} !important; }}
  [data-testid="stSidebar"] .stSelectbox label {{ font-size: 0.75rem !important; color: {COLORS["text_dim"]} !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }}
  [data-testid="stSidebar"] [data-testid="stSelectbox"],
  [data-testid="stSidebar"] [data-testid="stSelectbox"] div,
  [data-testid="stSidebar"] [data-baseweb="select"],
  [data-testid="stSidebar"] [data-baseweb="select"] div,
  [data-testid="stSidebar"] [data-baseweb="select"] > div,
  [data-testid="stSidebar"] [data-baseweb="select"] > div > div {{
    background: {COLORS["panel2"]} !important;
    background-color: {COLORS["panel2"]} !important;
    border-color: {COLORS["border"]} !important;
    box-shadow: none !important;
    color-scheme: light !important;
  }}
  [data-testid="stSidebar"] [data-testid="stSelectbox"] *,
  [data-testid="stSidebar"] [data-baseweb="select"] * {{
    color: {COLORS["text"]} !important;
    -webkit-text-fill-color: {COLORS["text"]} !important;
    fill: {COLORS["text"]} !important;
  }}
  [data-testid="stSidebar"] [data-baseweb="select"] svg {{ display: none !important; width: 0 !important; height: 0 !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] div:has(> svg) {{ background: transparent !important; background-color: transparent !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] [class*="Indicator"],
  [data-testid="stSidebar"] [data-baseweb="select"] > div > div:last-child {{
    background: transparent !important; background-color: transparent !important;
    width: auto !important; min-width: 0 !important;
  }}
  [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] {{
    position: relative !important; overflow: hidden !important; border-radius: 8px !important;
  }}
  [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"]::after {{
    content: "";
    position: absolute; right: 16px; top: 50%; transform: translateY(-50%);
    width: 0; height: 0; pointer-events: none;
    border-left: 4.5px solid transparent;
    border-right: 4.5px solid transparent;
    border-top: 5.5px solid {COLORS["text_mid"]};
  }}
  div[data-baseweb="popover"],
  div[data-baseweb="popover"] div {{ background: {COLORS["panel2"]} !important; background-color: {COLORS["panel2"]} !important; color-scheme: light !important; }}
  div[data-baseweb="popover"] [data-baseweb="menu"] {{ background: {COLORS["panel2"]} !important; border: 1px solid {COLORS["border"]} !important; }}
  div[data-baseweb="popover"] [data-baseweb="menu"] * {{ background: {COLORS["panel2"]} !important; color: {COLORS["text"]} !important; -webkit-text-fill-color: {COLORS["text"]} !important; }}
  div[data-baseweb="popover"] li[role="option"]:hover,
  div[data-baseweb="popover"] li[aria-selected="true"] {{ background: {COLORS["border_light"]} !important; }}
  div[data-baseweb="popover"] li[aria-selected="true"] * {{ background: {COLORS["border_light"]} !important; }}
  [data-testid="stSidebar"] .stNumberInput label {{ font-size: 0.75rem !important; color: {COLORS["text_dim"]} !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }}
  [data-testid="stSidebar"] .stNumberInput > div > div {{ background: {COLORS["panel2"]} !important; border: 1px solid {COLORS["border"]} !important; border-radius: 8px !important; }}
  [data-testid="stSidebar"] .stNumberInput input {{ background: {COLORS["panel2"]} !important; color: {COLORS["text"]} !important; -webkit-text-fill-color: {COLORS["text"]} !important; }}
  [data-testid="stSidebar"] .stNumberInput button {{ background: {COLORS["panel2"]} !important; border-color: {COLORS["border"]} !important; }}
  [data-testid="stSidebar"] .stNumberInput button svg {{ fill: {COLORS["text"]} !important; }}

  .stTabs [data-baseweb="tab-list"] {{
    background: {COLORS["text"]} !important; border-bottom: none !important;
    border-radius: 10px !important; padding: 0.4rem 1rem !important; gap: 4px;
  }}
  .stTabs [data-baseweb="tab-highlight"] {{ background: transparent !important; }}
  .stTabs [data-baseweb="tab-border"] {{ background: transparent !important; }}
  .stTabs [data-baseweb="tab"] {{
    color: {COLORS["white"]} !important; -webkit-text-fill-color: {COLORS["white"]} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
    padding: 0.7rem 1.2rem !important; margin-bottom: 0 !important;
    background: transparent !important; border: none !important;
    border-radius: 8px !important;
    text-decoration: none !important;
    opacity: 0.68;
  }}
  .stTabs [data-baseweb="tab"]:link, .stTabs [data-baseweb="tab"]:visited, .stTabs [data-baseweb="tab"]:hover, .stTabs [data-baseweb="tab"]:active {{ color: {COLORS["white"]} !important; -webkit-text-fill-color: {COLORS["white"]} !important; text-decoration: none !important; }}
  .stTabs [data-baseweb="tab"] * {{ color: {COLORS["white"]} !important; -webkit-text-fill-color: {COLORS["white"]} !important; }}
  .stTabs [data-baseweb="tab"][aria-selected="false"],
  .stTabs [data-baseweb="tab"][aria-selected="false"] * {{
    color: {COLORS["white"]} !important; -webkit-text-fill-color: {COLORS["white"]} !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: {COLORS["accent"]} !important;
    color: {COLORS["white"]} !important; -webkit-text-fill-color: {COLORS["white"]} !important;
    border-bottom: none !important; box-shadow: none !important;
    opacity: 1;
  }}
  .stTabs [aria-selected="true"]:link, .stTabs [aria-selected="true"]:visited, .stTabs [aria-selected="true"]:hover, .stTabs [aria-selected="true"]:active {{ color: {COLORS["white"]} !important; -webkit-text-fill-color: {COLORS["white"]} !important; }}
  .stTabs [aria-selected="true"] * {{ color: {COLORS["white"]} !important; -webkit-text-fill-color: {COLORS["white"]} !important; }}
  .stTabs [data-baseweb="tab-panel"] {{ padding: 1.5rem 2rem !important; background: {COLORS["bg"]}; }}

  [data-testid="metric-container"] {{ background: {COLORS["panel"]}; border: 1px solid {COLORS["border"]}; border-radius: 10px; padding: 1rem !important; }}
  [data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace !important; font-size: 1.55rem !important; color: {COLORS["text"]} !important; font-weight: 500 !important; }}
  [data-testid="stMetricLabel"] {{ font-size: 0.65rem !important; font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: {COLORS["text_dim"]} !important; }}

  .cs-alert {{ border-radius: 10px; padding: 1rem 1.3rem; margin: 0.6rem 0; display: flex; align-items: flex-start; gap: 0.8rem; }}
  .cs-alert-afib     {{ background: rgba(220,38,38,0.07);  border: 1px solid rgba(220,38,38,0.3);  border-left: 4px solid {COLORS["danger"]}; }}
  .cs-alert-normal   {{ background: rgba(21,128,61,0.07); border: 1px solid rgba(21,128,61,0.25); border-left: 4px solid {COLORS["success"]}; }}
  .cs-alert-borderline {{ background: rgba(217,119,6,0.07); border: 1px solid rgba(217,119,6,0.25); border-left: 4px solid {COLORS["warn"]}; }}

  .cs-label {{ font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {COLORS["text_dim"]}; margin-bottom: 0.5rem; padding-bottom: 0.3rem; border-bottom: 1px solid {COLORS["border"]}; }}
  .cs-card  {{ background: {COLORS["panel"]}; border: 1px solid {COLORS["border"]}; border-radius: 12px; padding: 1.4rem; margin-bottom: 0.8rem; }}
  .cs-badge {{ display: inline-flex; align-items: center; gap: 5px; background: {COLORS["panel2"]}; border: 1px solid {COLORS["border"]}; border-radius: 16px; padding: 3px 10px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: {COLORS["text_mid"]}; margin: 2px 0; }}
  .cs-pred-row {{ display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:0.85rem; padding:4px 0; }}

  .stButton>button {{ background: linear-gradient(135deg, {COLORS["accent2"]}, {COLORS["accent"]}) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; padding: 0.5rem 1.4rem !important; }}
  .stButton>button:hover {{ opacity: 0.9; }}
  .stDownloadButton > button {{ background: linear-gradient(135deg, {COLORS["accent2"]}, {COLORS["accent"]}) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 0.8rem !important; }}
  [data-testid="stFileUploader"] {{ background: {COLORS["panel2"]}; border: 1.5px dashed {COLORS["border_light"]}; border-radius: 10px; }}
  .stDataFrame {{ border: 1px solid {COLORS["border"]} !important; border-radius: 8px !important; overflow: hidden; }}
  .streamlit-expanderHeader {{ background: {COLORS["panel"]} !important; border: 1px solid {COLORS["border"]} !important; border-radius: 8px !important; color: {COLORS["text"]} !important; font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; }}
  .streamlit-expanderContent {{ background: {COLORS["panel"]} !important; border: 1px solid {COLORS["border"]} !important; border-top: none !important; border-radius: 0 0 8px 8px !important; }}
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {COLORS["bg"]}; }}
  ::-webkit-scrollbar-thumb {{ background: {COLORS["border_light"]}; border-radius: 3px; }}
  code {{ background: {COLORS["panel2"]} !important; color: {COLORS["accent2"]} !important; border: 1px solid {COLORS["border"]} !important; border-radius: 4px !important; padding: 1px 5px !important; }}

  .cs-table-wrap {{ max-height: 420px; overflow-y: auto; border: 1px solid {COLORS["border"]}; border-radius: 10px; }}
  .cs-table {{ width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; font-size: 0.82rem; }}
  .cs-table thead th {{ position: sticky; top: 0; background: {COLORS["panel2"]}; color: {COLORS["text_dim"]}; text-transform: uppercase; font-size: 0.66rem; letter-spacing: 0.07em; font-weight: 700; text-align: left; padding: 10px 14px; border-bottom: 1px solid {COLORS["border"]}; }}
  .cs-table tbody td {{ padding: 8px 14px; color: {COLORS["text"]}; border-bottom: 1px solid {COLORS["border"]}; background: {COLORS["panel"]}; }}
  .cs-table tbody tr:last-child td {{ border-bottom: none; }}
  .cs-table tbody tr:nth-child(even) td {{ background: {COLORS["panel2"]}; }}
  .cs-table td:nth-child(2) {{ font-family: 'JetBrains Mono', monospace; color: {COLORS["accent2"]}; font-weight: 500; }}
  .cs-table td:nth-child(3) {{ color: {COLORS["text_mid"]}; }}
  .cs-table td:nth-child(4) {{ color: {COLORS["text_mid"]}; }}
  pre  {{ background: {COLORS["panel2"]} !important; border: 1px solid {COLORS["border"]} !important; border-radius: 8px !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
FS     = 128
WINDOW = 3840   # 30 s × 128 Hz

SLIDE_WIN_SEC  = 15.0   # width of the sliding analysis window, in seconds
SLIDE_STEP_SEC = 1.0    # how far the window advances per step, in seconds

APP_DIR = Path(__file__).parent.resolve()

MODEL_PATHS = {
    "Random Forest": str(APP_DIR / "models" / "rf.pkl"),
    "XGBoost":        str(APP_DIR / "models" / "xgb.pkl"),
    "CatBoost":        str(APP_DIR / "models" / "catboost.pkl"),
}

FEATURE_NAMES = [
    "mean_rr","median_rr","sdnn","rmssd","pnn50","cv_rr",
    "mean_hr","std_hr","min_hr","max_hr",
    "mean_diff","max_diff","sd1","sd2","sd_ratio",
    "skewness","kurtosis","iqr","irr_score",
    "lf_hf","lf_norm","hf_norm","dominant_freq","n_beats",
]
FEATURE_UNITS = {
    "mean_rr":"ms","median_rr":"ms","sdnn":"ms","rmssd":"ms","pnn50":"%",
    "cv_rr":"","mean_hr":"bpm","std_hr":"bpm","min_hr":"bpm","max_hr":"bpm",
    "mean_diff":"ms","max_diff":"ms","sd1":"ms","sd2":"ms","sd_ratio":"",
    "skewness":"","kurtosis":"","iqr":"ms","irr_score":"","lf_hf":"",
    "lf_norm":"","hf_norm":"","dominant_freq":"Hz","n_beats":"",
}
FEATURE_DESCRIPTIONS = {
    "mean_rr":"Average RR interval","median_rr":"Median RR interval",
    "sdnn":"Std dev of RR intervals","rmssd":"Root mean square of successive differences",
    "pnn50":"% RR differences > 50 ms","cv_rr":"Coefficient of variation",
    "mean_hr":"Mean heart rate","std_hr":"HR standard deviation",
    "min_hr":"Minimum heart rate","max_hr":"Maximum heart rate",
    "mean_diff":"Mean absolute successive difference",
    "max_diff":"Max absolute successive difference",
    "sd1":"Poincaré SD1 (short-term variability)","sd2":"Poincaré SD2 (long-term variability)",
    "sd_ratio":"SD1/SD2 ratio","skewness":"RR interval skewness",
    "kurtosis":"RR interval kurtosis","iqr":"Interquartile range of RR",
    "irr_score":"Irregularity score","lf_hf":"LF/HF power ratio",
    "lf_norm":"Normalized LF power","hf_norm":"Normalized HF power",
    "dominant_freq":"Dominant frequency","n_beats":"Number of detected R-peaks",
}

DEMO_FILES = {
    "Normal #1": {"path": "samples/normal_1.npy", "record": "04043", "time": "0.5s", "sample": "68"},
    "Normal #2": {"path": "samples/normal_2.npy", "record": "04043", "time": "2940.1s", "sample": "376,328"},
    "Normal #3": {"path": "samples/normal_3.npy", "record": "04043", "time": "20332.2s", "sample": "2,602,516"},
    "AFib #1": {"path": "samples/afib_1.npy", "record": "04043", "time": "2082.0s", "sample": "266,498"},
    "AFib #2": {"path": "samples/afib_2.npy", "record": "04043", "time": "20197.5s", "sample": "2,585,284"},
    "AFib #3": {"path": "samples/afib_3.npy", "record": "04043", "time": "20585.2s", "sample": "2,634,911"},
}

# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def bandpass_filter(signal, fs=FS, low=0.5, high=40.0):
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype="band")
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
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.3*fs), prominence=sig_max*0.2)
    if len(peaks) < 3:
        peaks, _ = find_peaks(sig, height=max(0.2, sig_max*0.2), distance=int(0.3*fs))
    return peaks

def estimate_signal_quality(signal, fs=FS):
    """Lightweight heuristic signal-quality score, used by both the patient
    and doctor views ("GOOD" / "WEAK" / "POOR"). Not a validated clinical
    metric — just enough to warn the patient if a recording is unusable."""
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
    mean_rr   = float(np.mean(rr));  median_rr = float(np.median(rr))
    sdnn      = float(np.std(rr));   rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)))
    pnn50     = float(np.mean(np.abs(np.diff(rr)) > 50))
    cv_rr     = sdnn / mean_rr if mean_rr > 0 else 0.0
    hr        = 60000.0 / rr
    mean_hr   = float(np.mean(hr)); std_hr = float(np.std(hr))
    min_hr    = float(np.min(hr));  max_hr = float(np.max(hr))
    diff_rr   = np.diff(rr)
    mean_diff = float(np.mean(np.abs(diff_rr)))
    max_diff  = float(np.max(np.abs(diff_rr))) if len(diff_rr) > 0 else 0.0
    sd1       = float(np.sqrt(0.5) * np.std(diff_rr))
    sd2_sq    = max(0, 2*sdnn**2 - 0.5*np.std(diff_rr)**2)
    sd2       = float(np.sqrt(sd2_sq))
    sd_ratio  = sd1 / (sd2 + 1e-8)
    sk        = float(skew(rr)); ku = float(kurtosis(rr))
    iqr_val   = float(np.percentile(rr,75) - np.percentile(rr,25))
    irr       = float(np.sum(np.abs(diff_rr)/(rr[:-1]+1e-9) > 0.10) / max(len(diff_rr),1))
    try:
        t_rr  = np.cumsum(rr) / 1000.0
        t_uni = np.arange(t_rr[0], t_rr[-1], 0.25)
        if len(t_uni) > 8:
            rr_uni = interp1d(t_rr, rr, kind="linear",
                              bounds_error=False, fill_value="extrapolate")(t_uni)
            rr_uni -= np.mean(rr_uni)
            freqs, psd = welch(rr_uni, fs=4.0, nperseg=min(len(rr_uni), 64))
            pos = freqs > 0; freqs, psd = freqs[pos], psd[pos]
            lf  = np.sum(psd[(freqs>=0.04)&(freqs<0.15)])
            hf  = np.sum(psd[(freqs>=0.15)&(freqs<0.40)])
            tot = lf + hf + 1e-9
            lf_hf=lf/(hf+1e-9); lf_norm=lf/tot; hf_norm=hf/tot
            dom_freq=float(freqs[np.argmax(psd)])
        else:
            lf_hf=lf_norm=hf_norm=dom_freq=0.0
    except Exception:
        lf_hf=lf_norm=hf_norm=dom_freq=0.0
    return np.array([
        mean_rr,median_rr,sdnn,rmssd,pnn50,cv_rr,
        mean_hr,std_hr,min_hr,max_hr,
        mean_diff,max_diff,sd1,sd2,sd_ratio,
        sk,ku,iqr_val,irr,
        lf_hf,lf_norm,hf_norm,dom_freq,float(len(peaks)),
    ], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# HISTORY HELPERS (read recordings from server.py)
# ═══════════════════════════════════════════════════════════════════════════

def list_history(device_id: str = "", limit: int = 50) -> list[dict]:
    """Fetch recording history from server.py. Returns [] if server is down."""
    qs = f"?device_id={device_id}&limit={limit}" if device_id else ""
    r = _server_get(f"/api/recordings{qs}")
    if not r or not r.get("ok"):
        return []
    return r.get("recordings", [])[:limit]


def load_recording_from_server(recording_id: int) -> tuple[np.ndarray | None, dict | None]:
    """Load one recording's samples from server.py. Returns (signal, meta)."""
    r = _server_get(f"/api/recording/{int(recording_id)}")
    if not r or not r.get("ok"):
        return None, None
    raw = base64.b64decode(r.get("samples_b64", ""))
    sig = np.frombuffer(raw, dtype=np.float32) if raw else np.array([], dtype=np.float32)
    meta = {k: v for k, v in r.items() if k not in ("ok", "samples_b64")}
    return sig, meta


def recording_summary(rec: dict) -> str:
    """Short one-line summary for the history table."""
    try:
        ts = rec["started_at"].replace("T", " ")[:16]
    except Exception:
        ts = "?"
    dur = float(rec.get("duration_s") or 0.0)
    return f"{ts} · {dur:.1f}s · {rec.get('n_samples', 0)} samples"


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_pkl(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        if JOBLIB_AVAILABLE:
            return joblib.load(str(p))
        else:
            with open(p, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
        return None

@st.cache_resource
def load_xgb_model(path: str):
    if not XGB_AVAILABLE:
        return None
    return _load_pkl(path)

@st.cache_resource
def load_rf_model(path: str):
    return _load_pkl(path)

@st.cache_resource
def load_catboost_model(path: str):
    if not CATBOOST_AVAILABLE:
        return None
    return _load_pkl(path)

# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def _unpack_model(model):
    if isinstance(model, dict):
        return model["model"], model.get("imputer"), float(model.get("threshold", 0.5))
    return model, None, 0.5

def predict_rf(model, features):
    clf, imputer, threshold = _unpack_model(model)
    x = features.reshape(1, -1)
    if imputer is not None:
        x = imputer.transform(x)
    prob = float(clf.predict_proba(x)[0][1])
    return ("AFib" if prob >= threshold else "Normal"), prob, threshold

def predict_xgb(model, features):
    clf, imputer, threshold = _unpack_model(model)
    x = features.reshape(1, -1)
    if imputer is not None:
        x = imputer.transform(x)
    prob = float(clf.predict_proba(x)[0][1])
    return ("AFib" if prob >= threshold else "Normal"), prob, threshold

def predict_catboost(model, features):
    clf, imputer, threshold = _unpack_model(model)
    x = features.reshape(1, -1)
    if imputer is not None:
        x = imputer.transform(x)
    prob = float(clf.predict_proba(x)[0][1])
    return ("AFib" if prob >= threshold else "Normal"), prob, threshold

def hrv_heuristic(features):
    feat = dict(zip(FEATURE_NAMES, features))
    score = 0.0; reasons = {}
    sdnn_n  = min(feat["sdnn"]/150.0, 1.0);      reasons["SDNN (variability)"]  = sdnn_n;  score += sdnn_n*0.25
    pnn50_n = min(feat["pnn50"]/0.5, 1.0);        reasons["pNN50"]               = pnn50_n; score += pnn50_n*0.20
    irr_n   = min(feat["irr_score"]/0.4, 1.0);    reasons["Irregularity score"]  = irr_n;   score += irr_n*0.25
    rmssd_n = min(feat["rmssd"]/120.0, 1.0);      reasons["RMSSD"]               = rmssd_n; score += rmssd_n*0.15
    lfhf_i  = max(0.0,1.0-min(feat["lf_hf"]/2.0,1.0)); reasons["LF/HF imbalance"] = lfhf_i; score += lfhf_i*0.10
    cv_n    = min(feat["cv_rr"]/0.25, 1.0);       reasons["CV of RR"]            = cv_n;    score += cv_n*0.05
    prob = float(np.clip(score, 0.0, 1.0))
    threshold = 0.45
    return ("AFib" if prob >= threshold else "Normal"), prob, threshold, reasons


def run_prediction(model_choice, features, silent=False):
    """
    Runs the selected model (or falls back to the HRV heuristic) on a single
    feature vector. Shared by the full-signal prediction and the sliding
    window, so both stay in sync.

    Returns: label, prob, threshold, method_note, reasons(dict|None), individual_preds(dict)
    """
    reasons = None
    individual_preds = {}

    def _warn(msg):
        if not silent:
            st.warning(msg)

    if model_choice == "Random Forest":
        mdl = load_rf_model(MODEL_PATHS["Random Forest"])
        if mdl is None:
            _warn(f"Random Forest model not found at `{MODEL_PATHS['Random Forest']}`. "
                  "Falling back to HRV heuristic.")
            label, prob, threshold, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (Random Forest weights missing)"
        else:
            label, prob, threshold = predict_rf(mdl, features)
            method_note = f"Random Forest — {MODEL_PATHS['Random Forest']}"

    elif model_choice == "XGBoost":
        mdl = load_xgb_model(MODEL_PATHS["XGBoost"])
        if mdl is None:
            _warn(f"XGBoost model not found at `{MODEL_PATHS['XGBoost']}`. "
                  "Falling back to HRV heuristic.")
            label, prob, threshold, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (XGBoost weights missing)"
        else:
            label, prob, threshold = predict_xgb(mdl, features)
            method_note = f"XGBoost — {MODEL_PATHS['XGBoost']}"

    elif model_choice == "CatBoost":
        mdl = load_catboost_model(MODEL_PATHS["CatBoost"])
        if mdl is None:
            _warn(f"CatBoost model not found at `{MODEL_PATHS['CatBoost']}`. "
                  "Falling back to HRV heuristic.")
            label, prob, threshold, reasons = hrv_heuristic(features)
            method_note = "HRV heuristic (CatBoost weights missing)"
        else:
            label, prob, threshold = predict_catboost(mdl, features)
            method_note = f"CatBoost — {MODEL_PATHS['CatBoost']}"

    elif model_choice == "Ensemble":
        rf_model  = load_rf_model(MODEL_PATHS["Random Forest"])
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
                  f"(`{MODEL_PATHS['Random Forest']}`, `{MODEL_PATHS['XGBoost']}`, "
                  f"`{MODEL_PATHS['CatBoost']}`). Falling back to HRV heuristic.")
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
    """Run HRV extraction + prediction on a signal and return a flat dict
    used by the patient Home/Result/Why-flagged screens. Silent (no
    st.warning spam) since patients shouldn't see model-loading messages."""
    features = extract_hrv(signal, fs)
    feat = dict(zip(FEATURE_NAMES, features))
    peaks = detect_rpeaks(signal, fs)
    label, prob, threshold, method_note, reasons, individual_preds = run_prediction(
        model_choice, features, silent=True
    )
    if reasons is None:
        # Model-based prediction has no HRV heuristic breakdown — build a
        # patient-friendly one anyway so "Why was this flagged?" always has
        # something to show.
        _, _, _, reasons = hrv_heuristic(features)
    quality, quality_score = estimate_signal_quality(signal, fs)
    return {
        "label": label,
        "prob": prob,
        "threshold": threshold,
        "method_note": method_note,
        "reasons": reasons,
        "hr": feat.get("mean_hr", 0.0),
        "rmssd": feat.get("rmssd", 0.0),
        "sdnn": feat.get("sdnn", 0.0),
        "quality": quality,
        "quality_score": quality_score,
        "peaks": peaks,
        "n_peaks": len(peaks),
    }

# ═══════════════════════════════════════════════════════════════════════════
# PLOTS  — all use AFibAI palette
# ═══════════════════════════════════════════════════════════════════════════

def _base_layout(**kwargs):
    base = dict(
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Inter"),
        margin=dict(l=55, r=20, t=45, b=45),
    )
    base.update(kwargs)
    return base

def plot_ecg(signal, peaks, fs=FS, title="ECG Signal", is_afib=False,
             window_range=None, window_color=None):
    max_pts = 1500
    step    = max(1, len(signal) // max_pts)
    disp    = signal[::step]
    t       = np.arange(len(disp)) * step / fs
    tc      = COLORS["ecg_afib"] if is_afib else COLORS["ecg_normal"]

    fig = go.Figure()

    if window_range is not None:
        w_start, w_end = window_range
        w_center = (w_start + w_end) / 2
        wc = window_color or COLORS["accent"]
        fig.add_vrect(
            x0=w_start, x1=w_end,
            fillcolor=wc, opacity=0.28,
            line=dict(color=wc, width=4),
            layer="below",
        )
        # A thick center line + callout label keep the window visible even
        # when it's a thin sliver relative to a very long recording.
        fig.add_vline(
            x=w_center, line=dict(color=wc, width=3, dash="solid"),
            opacity=0.9,
        )
        fig.add_annotation(
            x=w_center, y=1.0, xref="x", yref="paper",
            text=f"◆ WINDOW  {w_start:.0f}s–{w_end:.0f}s",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor=wc, ax=0, ay=-32,
            font=dict(family="JetBrains Mono", size=11, color=wc),
            bgcolor="rgba(255,255,255,0.92)", bordercolor=wc, borderwidth=1.5,
            borderpad=4,
        )

    fig.add_trace(go.Scatter(
        x=t, y=disp, mode="lines",
        line=dict(color=tc, width=1.4), name="ECG",
        hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3f}<extra></extra>",
    ))
    if len(peaks):
        valid = peaks[(peaks >= 0) & (peaks < len(signal))]
        fig.add_trace(go.Scatter(
            x=valid / fs, y=signal[valid], mode="markers",
            marker=dict(color=COLORS["danger"], size=8, symbol="circle",
                        line=dict(color="white", width=1.5)),
            name="R-peaks",
        ))
    fig.update_layout(
        **_base_layout(height=300, plot_bgcolor=COLORS["ecg_bg"],
                        margin=dict(l=55, r=20, t=(75 if window_range is not None else 45), b=45)),
        title=dict(text=title, font=dict(family="Inter", size=12, color=COLORS["text_mid"]), x=0.01),
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(color=COLORS["text_mid"])), color=COLORS["text_mid"],
            gridcolor=COLORS["ecg_grid_maj"], gridwidth=1, dtick=1, showgrid=True,
            minor=dict(dtick=0.08, gridcolor=COLORS["ecg_grid_min"], showgrid=True),
            tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"]),
        ),
        yaxis=dict(
            title=dict(text="Amplitude (norm.)", font=dict(color=COLORS["text_mid"])), color=COLORS["text_mid"],
            gridcolor=COLORS["ecg_grid_maj"], gridwidth=1, dtick=1, showgrid=True,
            minor=dict(dtick=0.1, gridcolor=COLORS["ecg_grid_min"], showgrid=True),
            tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"]),
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=COLORS["border_light"],
                    borderwidth=1.5, font=dict(family="Inter", size=11, color=COLORS["text"])),
        hovermode="x unified",
    )
    return fig

def plot_rr(rr_ms):
    m = float(np.mean(rr_ms))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rr_ms, mode="lines+markers",
        line=dict(color=COLORS["accent"], width=1.8),
        marker=dict(color=COLORS["accent"], size=4),
        hovertemplate="Beat %{x}<br>RR: %{y:.0f}ms<extra></extra>",
    ))
    fig.add_hline(y=m, line_dash="dash", line_color=COLORS["warn"], opacity=0.6,
                  annotation_text=f"Mean: {m:.0f}ms",
                  annotation_font=dict(color=COLORS["warn"], size=10))
    fig.add_hrect(y0=600, y1=1000, fillcolor="rgba(31,204,122,0.16)", line_width=0,
                  annotation_text="Normal range",
                  annotation_position="top left",
                  annotation=dict(font_color=COLORS["success"], font_size=11))
    fig.update_layout(
        **_base_layout(height=260),
        title=dict(text="RR Interval Series", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        xaxis=dict(title=dict(text="Beat #", font=dict(color=COLORS["text_mid"])),
                   color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)",
                   tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"])),
        yaxis=dict(title=dict(text="RR (ms)", font=dict(color=COLORS["text_mid"])),
                   color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)",
                   tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"])),
    )
    return fig

def plot_poincare(rr_ms, is_afib=False):
    if len(rr_ms) < 4:
        return go.Figure()
    arr   = np.array(rr_ms)
    color = COLORS["danger"] if is_afib else COLORS["accent"]
    lim   = [max(300, arr.min()-50), min(2000, arr.max()+50)]
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=arr[:-1], y=arr[1:], mode="markers",
        marker=dict(color=color, size=5, opacity=0.65,
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
        hovertemplate="RRn: %{x:.0f}ms<br>RRn+1: %{y:.0f}ms<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(color=COLORS["text_dim"], dash="dash", width=1.5), showlegend=False,
    ))
    fig.update_layout(
        **_base_layout(height=260),
        title=dict(text="Poincaré Plot", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        xaxis=dict(title=dict(text="RRₙ (ms)", font=dict(color=COLORS["text_mid"])),
                   color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)",
                   range=lim, tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"])),
        yaxis=dict(title=dict(text="RRₙ₊₁ (ms)", font=dict(color=COLORS["text_mid"])),
                   color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)",
                   range=lim, tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"])),
    )
    return fig


def plot_radar(features):
    keys   = ["sdnn","rmssd","pnn50","irr_score","cv_rr","sd1","sd2","lf_hf"]
    norms  = {"sdnn":150,"rmssd":120,"pnn50":1,"irr_score":0.5,
               "cv_rr":0.3,"sd1":80,"sd2":150,"lf_hf":5}
    labels = ["SDNN","RMSSD","pNN50","Irregularity","CV(RR)","SD1","SD2","LF/HF"]
    feat   = dict(zip(FEATURE_NAMES, features))
    vals   = [min(feat[k]/norms[k],1.0) for k in keys]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=labels+[labels[0]], fill="toself",
        fillcolor="rgba(42,181,181,0.12)", line=dict(color=COLORS["accent"], width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["panel"],
            radialaxis=dict(visible=True, range=[0,1], color=COLORS["text_mid"],
                            gridcolor="rgba(91,117,104,0.25)",
                            tickfont=dict(color=COLORS["text_mid"])),
            angularaxis=dict(color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)"),
        ),
        paper_bgcolor=COLORS["panel"],
        title=dict(text="HRV Radar", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        margin=dict(l=40,r=40,t=50,b=40), showlegend=False,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# LOGIN / REGISTRATION GATE  (mock auth — no real password check, no DB)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# LICENSE AGREEMENT  (Thai + English)
# ═══════════════════════════════════════════════════════════════════════════

LICENSE_THAI = (
"ซอฟต์แวร์นี้เป็นผลงานที่พัฒนาขึ้นโดย นายวิชญ์พัศ โสทรทวีพงศ์ "
"นายพีระวิทย์ ประสิทธิ์ผล และ นางสาวกานต์พิชชา มุขแก้ว "
"จาก โรวเรียนวารีเชียงใหม่ ภายใต้การดูแลของ นาย สรพงษ์ สมสอน "
"ภายใต้โครงการ ระบบตรวจจับภาวะหัวใจห้องบนสั่นพลิ้วจากสัญญาณ ECG ด้วยปัญญาประดิษฐ์ (AFib AI)  "
"ซึ่งสนับสนุนโดย สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ "
"โดยมีวัตถุประสงค์เพื่อส่งเสริมให้นักเรียนและนักศึกษาได้เรียนรู้และฝึกทักษะในการพัฒนาซอฟต์แวร์ "
"ลิขสิทธิ์ของซอฟต์แวร์นี้จึงเป็นของผู้พัฒนา ซึ่งผู้พัฒนาได้อนุญาตให้สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ "
"เผยแพร่ซอฟต์แวร์นี้ตาม \"ต้นฉบับ\" โดยไม่มีการแก้ไขดัดแปลงใด ๆ ทั้งสิ้น "
"ให้แก่บุคคลทั่วไปได้ใช้เพื่อประโยชน์ส่วนบุคคลหรือประโยชน์ทางการศึกษาที่ไม่มีวัตถุประสงค์ในเชิงพาณิชย์ "
"โดยไม่คิดค่าตอบแทนการใช้ซอฟต์แวร์ "
"ดังนั้น สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ "
"จึงไม่มีหน้าที่ในการดูแล บำรุงรักษา จัดการอบรมการใช้งาน หรือพัฒนาประสิทธิภาพซอฟต์แวร์ "
"รวมทั้งไม่รับรองความถูกต้องหรือประสิทธิภาพการทำงานของซอฟต์แวร์ "
"ตลอดจนไม่รับประกันความเสียหายต่าง ๆ อันเกิดจากการใช้ซอฟต์แวร์ทั้งสิ้น"
)

LICENSE_ENGLISH = (
"This software is a work developed by Mr.Wichapat Sothorntweepong, "
"Mr.Peerawit Prasitphon and Ms.Kanpitcha Mookkaew from Varee Chiangmai School "
"under the provision of Mr. Sorapong Somsorn under AI-Based Atrial Fibrillation "
"Detection System Using ECG Signals (AFib AI) , which has been supported by the "
"National Science and Technology Development Agency (NSTDA), in order to encourage "
"pupils and students to learn and practice their skills in developing software. "
"Therefore, the intellectual property of this software shall belong to the developer "
"and the developer gives NSTDA a permission to distribute this software as an \"as is\" "
"and non-modified software for a temporary and non-exclusive use without remuneration "
"to anyone for his or her own purpose or academic purposes, which are not commercial "
"purposes. In this connection, NSTDA shall not be responsible to the user for taking "
"care, maintaining, training, or developing the efficiency of this software. Moreover, "
"NSTDA shall not be liable for any error, software efficiency and damages in connection "
"with or arising out of the use of the software."
)

def _agreement_css():
    """Extra CSS for the license agreement screen + the in-app agreement dialog."""
    st.markdown(f"""
    <style>
      .agreement-wrap {{
        max-width: 760px; margin: 2.5rem auto 2rem; padding: 0 1.5rem;
      }}
      .agreement-hero {{
        text-align: center; margin-bottom: 1.2rem;
      }}
      .agreement-hero .heart {{
        font-size: 2.4rem; line-height: 1; margin-bottom: 8px;
      }}
      .agreement-hero h1 {{
        font-family: 'Sora', sans-serif; font-size: 1.7rem; font-weight: 700;
        color: {COLORS['text']}; margin: 0 0 4px;
      }}
      .agreement-hero .tag {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
        color: {COLORS['text_dim']}; letter-spacing: 0.18em;
        text-transform: uppercase; margin-bottom: 6px;
      }}
      .agreement-hero .sub {{
        font-size: 0.82rem; color: {COLORS['text_mid']}; line-height: 1.5;
        max-width: 520px; margin: 0 auto;
      }}
      .agreement-card {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['border']};
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 1px 0 rgba(255,255,255,0.6) inset, 0 6px 24px rgba(20,40,30,0.04);
      }}
      .agreement-section-title {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {COLORS['accent2']};
        margin: 0 0 0.5rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid {COLORS['border']};
      }}
      .agreement-section-title + .agreement-section-title {{ margin-top: 1.2rem; }}
      .agreement-body {{
        font-size: 0.85rem;
        line-height: 1.75;
        color: {COLORS['text']};
        text-align: justify;
        white-space: pre-wrap;
        max-height: 320px;
        overflow-y: auto;
        padding: 0.6rem 0.4rem 0.6rem 0.2rem;
        background: {COLORS['panel2']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
      }}
      .agreement-body::-webkit-scrollbar {{ width: 6px; }}
      .agreement-body::-webkit-scrollbar-thumb {{ background: {COLORS['border_light']}; border-radius: 3px; }}
      .agreement-foot {{
        text-align: center; margin-top: 1.2rem;
        font-size: 0.7rem; color: {COLORS['text_dim']}; line-height: 1.6;
      }}
      /* When the agreement is shown inside an st.dialog, hide our custom
         wrapper centering (the dialog already centers its own content). */
      [data-testid="stDialog"] .agreement-wrap {{ max-width: 100%; margin: 0; padding: 0; }}
      [data-testid="stDialog"] .agreement-hero {{ text-align: left; margin-bottom: 0.8rem; }}
      [data-testid="stDialog"] .agreement-hero .heart {{ display: none; }}
      [data-testid="stDialog"] .agreement-card {{ box-shadow: none; padding: 1rem 0; border: none; background: transparent; }}
      [data-testid="stDialog"] .agreement-body {{ max-height: 50vh; }}
    </style>
    """, unsafe_allow_html=True)


def _render_agreement_body(*, show_titles: bool = True):
    """Renders the Thai + English agreement text. Used by both the
    pre-login agreement screen and the in-app sidebar dialog."""
    if show_titles:
        st.markdown(
            '<div class="agreement-section-title">ข้อตกลงการใช้ซอฟต์แวร์ (ภาษาไทย)</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="agreement-body">{LICENSE_THAI}</div>',
        unsafe_allow_html=True,
    )
    if show_titles:
        st.markdown(
            '<div class="agreement-section-title">License Agreement (English)</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="agreement-body">{LICENSE_ENGLISH}</div>',
        unsafe_allow_html=True,
    )


def license_agreement_screen():
    """Step 0: shown before the login screen. User must click 'Next'
    to proceed to sign-in. Pure UI gate — acceptance isn't stored
    anywhere, the user can re-open the agreement from the sidebar."""
    _agreement_css()

    st.markdown(f"""
    <div class="agreement-wrap">
      <div class="agreement-hero">
        <div class="heart">🫀</div>
        <div class="tag">BEFORE YOU BEGIN</div>
        <h1>License Agreement</h1>
        <div class="sub">
          Please read the software license agreement below before continuing<br>
          to the AFibAI application.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="agreement-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="agreement-card">', unsafe_allow_html=True)
    _render_agreement_body(show_titles=True)
    st.markdown('</div>', unsafe_allow_html=True)  # /.agreement-card

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    _spacer_l, _btn_c, _spacer_r = st.columns([1, 1, 1])
    with _btn_c:
        if st.button("Next  →", key="agreement_next", use_container_width=True):
            st.session_state["agreement_accepted"] = True
            st.rerun()

    st.markdown(f"""
      <div class="agreement-foot">
        Clicking <b>Next</b> indicates that you have read and acknowledge<br>
        the license terms above.
      </div>
    </div>
    """, unsafe_allow_html=True)


@st.dialog("ข้อตกลง / License Agreement", width="large")
def _show_agreement_dialog():
    """In-app dialog opened from the sidebar. Shows the same text and
    closes when the user clicks 'OK'."""
    _agreement_css()
    st.markdown(
        '<div class="agreement-section-title">ข้อตกลงการใช้ซอฟต์แวร์ (ภาษาไทย)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="agreement-body">{LICENSE_THAI}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="agreement-section-title">License Agreement (English)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="agreement-body">{LICENSE_ENGLISH}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    _ok_l, _ok_c, _ok_r = st.columns([1, 1, 1])
    with _ok_c:
        if st.button("OK", key="agreement_ok", use_container_width=True):
            st.rerun()


def _login_css():
    """Extra CSS for the login + medical-info screen. Reuses the AFibAI palette."""
    st.markdown(f"""
    <style>
      .login-wrap {{
        max-width: 520px; margin: 4rem auto 2rem; padding: 0 1.5rem;
      }}
      .login-hero {{
        text-align: center; margin-bottom: 1.6rem;
      }}
      .login-hero .heart {{
        font-size: 2.6rem; line-height: 1; margin-bottom: 10px;
      }}
      .login-hero h1 {{
        font-family: 'Sora', sans-serif; font-size: 1.9rem; font-weight: 700;
        color: {COLORS['text']}; margin: 0 0 4px;
      }}
      .login-hero .tag {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
        color: {COLORS['text_dim']}; letter-spacing: 0.18em;
        text-transform: uppercase; margin-bottom: 6px;
      }}
      .login-hero .sub {{
        font-size: 0.82rem; color: {COLORS['text_mid']}; line-height: 1.5;
        max-width: 420px; margin: 0 auto;
      }}
      .login-card {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['border']};
        border-radius: 14px;
        padding: 1.8rem 1.8rem 1.4rem;
        box-shadow: 0 1px 0 rgba(255,255,255,0.6) inset, 0 6px 24px rgba(20,40,30,0.04);
      }}
      .login-tabs [data-baseweb="tab-list"] {{
        background: {COLORS['panel2']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 10px !important;
        padding: 4px !important; gap: 4px;
      }}
      .login-tabs [data-baseweb="tab"] {{
        color: {COLORS['text_mid']} !important;
        -webkit-text-fill-color: {COLORS['text_mid']} !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 7px !important;
        background: transparent !important;
        opacity: 1 !important;
      }}
      .login-tabs [aria-selected="true"] {{
        background: {COLORS['panel']} !important;
        color: {COLORS['text']} !important;
        -webkit-text-fill-color: {COLORS['text']} !important;
        box-shadow: 0 1px 3px rgba(20,40,30,0.08) !important;
        opacity: 1 !important;
      }}
      .login-tabs [data-baseweb="tab-panel"] {{
        padding: 1.2rem 0.2rem 0 !important;
        background: transparent !important;
      }}
      .login-foot {{
        text-align: center; margin-top: 1.2rem;
        font-size: 0.7rem; color: {COLORS['text_dim']}; line-height: 1.6;
      }}
      .login-warn {{
        background: rgba(217,119,6,0.08);
        border: 1px solid rgba(217,119,6,0.25);
        border-left: 4px solid {COLORS['warn']};
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        font-size: 0.78rem; color: {COLORS['text_mid']};
        margin: 0.8rem 0 1rem;
      }}
      .login-card .stTextInput input,
      .login-card .stNumberInput input {{
        background: {COLORS['panel2']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
        color: {COLORS['text']} !important;
        -webkit-text-fill-color: {COLORS['text']} !important;
      }}
      .login-card .stTextInput label,
      .login-card .stNumberInput label,
      .login-card .stCheckbox label {{
        font-size: 0.72rem !important;
        color: {COLORS['text_dim']} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 600 !important;
      }}
    </style>
    """, unsafe_allow_html=True)


def login_screen():
    """Real login/signup via auth.py. On success, routes to the intake
    screen for patients, or straight into the app for clinicians (doctors
    don't need to fill in their own medical history)."""
    _login_css()

    st.markdown(f"""
    <div class="login-wrap">
      <div class="login-hero">
        <div class="heart">🫀</div>
        <div class="tag">HRV ANALYSIS · v1.0</div>
        <h1>AFibAI</h1>
        <div class="sub">
          Sign in to access the Atrial Fibrillation screening tool.<br>
          New here? Create an account to get started.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-tabs">', unsafe_allow_html=True)

    tab_in, tab_up = st.tabs(["🔒  Sign In", "✨  Create Account"])

    with tab_in:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            li_user = st.text_input("Username", placeholder="e.g. jane.doe",
                                     key="li_user")
            li_pass = st.text_input("Password", type="password",
                                     placeholder="Enter your password", key="li_pass")
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Log In", use_container_width=True)
            if submitted:
                if not li_user.strip() or not li_pass.strip():
                    st.error("Please enter both a username and a password.")
                else:
                    ok, user, msg = auth.signin(li_user.strip(), li_pass)
                    if not ok:
                        st.error(msg)
                    else:
                        st.session_state["auth_user"] = user["username"]
                        st.session_state["auth_user_id"] = user["id"]
                        role = user.get("role", auth.ROLE_CLINICIAN)
                        st.session_state["auth_role"] = role
                        if role == auth.ROLE_CLINICIAN:
                            # Doctors/clinicians skip the personal medical
                            # intake form entirely — it's only relevant for
                            # patients recording their own ECG.
                            st.session_state["auth_user_data"] = {
                                "full_name": user["username"], "role": "clinician",
                            }
                            st.session_state["authenticated"] = True
                        else:
                            st.session_state["auth_stage"] = "intake"
                        st.rerun()

    with tab_up:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        with st.form("signup_form", clear_on_submit=False):
            su_user = st.text_input("Choose a username", placeholder="e.g. jane.doe",
                                     key="su_user")
            su_pass = st.text_input("Choose a password", type="password",
                                     placeholder="At least 6 characters", key="su_pass")
            su_conf = st.text_input("Confirm password", type="password",
                                     placeholder="Re-enter your password", key="su_conf")
            su_role = st.selectbox(
                "I am a…",
                [auth.ROLE_PATIENT, auth.ROLE_CLINICIAN],
                format_func=lambda r: "Patient" if r == auth.ROLE_PATIENT else "Doctor / Clinician",
                key="su_role",
                help="Patients see a simple phone-friendly interface. "
                     "Clinicians see the full analysis dashboard.",
            )
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                u = su_user.strip()
                if not u or not su_pass:
                    st.error("Username and password are required.")
                elif len(su_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif su_pass != su_conf:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = auth.signup(u, su_pass, role=su_role)
                    if not ok:
                        st.error(msg)
                    else:
                        # Auto-sign-in the newly created user.
                        ok2, user, _ = auth.signin(u, su_pass)
                        if ok2 and user:
                            st.session_state["auth_user"] = user["username"]
                            st.session_state["auth_user_id"] = user["id"]
                            role = user.get("role", auth.ROLE_CLINICIAN)
                            st.session_state["auth_role"] = role
                            if role == auth.ROLE_CLINICIAN:
                                # Same skip for freshly-created doctor accounts.
                                st.session_state["auth_user_data"] = {
                                    "full_name": user["username"], "role": "clinician",
                                }
                                st.session_state["authenticated"] = True
                            else:
                                st.session_state["auth_stage"] = "intake"
                            st.rerun()
                        else:
                            st.success("Account created. Please sign in.")

    st.markdown('</div>', unsafe_allow_html=True)  # /.login-tabs
    st.markdown('</div>', unsafe_allow_html=True)  # /.login-card
    st.markdown(f"""
      <div class="login-foot">
        🔒 This is a research prototype. Accounts are not stored on a server<br>
        and no data leaves your browser session.
      </div>
    </div>
    """, unsafe_allow_html=True)


def medical_intake_screen():
    """Step 2: collect basic medical / demographic info before unlocking the
    main app. Patients only — clinicians skip this screen entirely (see
    login_screen). There's a Skip button at the top right that bypasses the
    form. Mock — values are stored in session state only."""
    _login_css()

    user = st.session_state.get("auth_user", "User")

    # ── Top-right "Skip" button — sits above the card so it doesn't move
    # when the form changes. Right-aligned with a narrow left column.
    _spacer, _skip_col = st.columns([11, 1])
    with _skip_col:
        if st.button("Skip →", key="intake_skip_top",
                      help="Skip the intake form and go straight to the app."):
            st.session_state["auth_user_data"] = {
                "full_name": user, "age": None, "skipped": True,
            }
            # Re-resolve role from DB so we don't lose it across the skip.
            user_id = st.session_state.get("auth_user_id")
            if user_id is not None:
                st.session_state["auth_role"] = auth.get_role(int(user_id))
            st.session_state["authenticated"] = True
            st.rerun()

    st.markdown(f"""
    <div class="login-wrap">
      <div class="login-hero">
        <div class="heart">🫀</div>
        <div class="tag">STEP 2 OF 2</div>
        <h1>About You</h1>
        <div class="sub">
          Welcome, <b>{user}</b>. Please share a few details so we can
          contextualize your results. All fields stay in this session.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown(f"""
      <div class="login-warn">
        ⚠️ The information below is <b>not</b> sent to any server and is used
        only to add context to your analysis. This tool is a research
        prototype, not a medical device.
      </div>
    """, unsafe_allow_html=True)

    with st.form("intake_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full Name", placeholder="e.g. Jane Doe",
                                       key="mi_name")
            age = st.number_input("Age (years)", min_value=0, max_value=120,
                                   value=30, step=1, key="mi_age")
            gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"],
                                   key="mi_gender")
        with c2:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=400.0,
                                      value=65.0, step=0.1, key="mi_weight")
            height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0,
                                      value=165.0, step=0.1, key="mi_height")
            smoke = st.selectbox("Smoking status",
                                  ["Never", "Former", "Current", "Prefer not to say"],
                                  key="mi_smoke")

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        med_hist = st.text_area("Medical History",
                                 placeholder="e.g. Hypertension, diabetes, prior arrhythmia, …",
                                 key="mi_hist", height=70)
        meds = st.text_input("Current Medications",
                              placeholder="e.g. Metoprolol 25 mg, Aspirin 81 mg",
                              key="mi_meds")
        allergies = st.text_input("Known Allergies",
                                   placeholder="e.g. Penicillin, latex",
                                   key="mi_allergies")
        c3, c4 = st.columns(2)
        with c3:
            family = st.text_input("Family History of Heart Disease",
                                    placeholder="e.g. Father — AFib; Mother — HTN",
                                    key="mi_family")
        with c4:
            prev_ecg = st.text_input("Previous ECG Results (if any)",
                                      placeholder="e.g. Normal sinus rhythm, 2024",
                                      key="mi_prev")
        reason = st.text_area("Reason for using AFibAI",
                               placeholder="e.g. Routine screening, symptom follow-up, research, …",
                               key="mi_reason", height=60)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        agreed = st.checkbox("I understand this is a research tool, not a medical device.",
                              key="mi_agree")
        c5, c6 = st.columns([1, 1])
        with c5:
            back = st.form_submit_button("← Back", use_container_width=True)
        with c6:
            submit = st.form_submit_button("Submit & Enter  →", use_container_width=True)

    if back:
        for k in ("auth_stage", "auth_user"):
            st.session_state.pop(k, None)
        st.rerun()

    if submit:
        if not agreed:
            st.error("Please confirm that you understand the research-prototype notice.")
        elif not full_name.strip():
            st.error("Please enter your full name.")
        else:
            st.session_state["auth_user_data"] = {
                "full_name": full_name.strip(),
                "age": int(age),
                "gender": gender,
                "weight_kg": float(weight),
                "height_cm": float(height),
                "smoking": smoke,
                "medical_history": med_hist.strip(),
                "medications": meds.strip(),
                "allergies": allergies.strip(),
                "family_history": family.strip(),
                "previous_ecg": prev_ecg.strip(),
                "reason": reason.strip(),
            }
            # Persist intake fields to the user row if we have a user_id,
            # and resolve the role now so the post-login view knows what to show.
            user_id = st.session_state.get("auth_user_id")
            if user_id is not None:
                try:
                    auth.save_intake(int(user_id), st.session_state["auth_user_data"])
                except Exception:
                    pass
                st.session_state["auth_role"] = auth.get_role(int(user_id))
            st.session_state["authenticated"] = True
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # /.login-card
    st.markdown('</div>', unsafe_allow_html=True)  # /.login-wrap


# ═══════════════════════════════════════════════════════════════════════════
# PATIENT VIEW  (phone-first — minimal UI, big buttons)
# ═══════════════════════════════════════════════════════════════════════════

def _patient_css():
    """CSS scoped to the patient view. Big tap targets, narrow column,
    friendly colors. Hides most Streamlit chrome."""
    st.markdown(f"""
    <style>
      [data-testid="stSidebar"] {{ display: none; }}
      .main .block-container {{ max-width: 520px !important; padding: 1rem !important; }}
      .patient-hero {{
        text-align: center; padding: 1rem 0 0.4rem;
      }}
      .patient-hero .heart {{ font-size: 2.6rem; line-height: 1; }}
      .patient-hero h1 {{
        font-family: 'Sora', sans-serif; font-size: 1.6rem; font-weight: 700;
        color: {COLORS['text']}; margin: 6px 0 4px;
      }}
      .patient-hero .sub {{
        font-family: 'Inter', sans-serif; font-size: 0.82rem;
        color: {COLORS['text_mid']}; line-height: 1.5;
      }}
      .patient-status {{
        background: {COLORS['panel']}; border: 1px solid {COLORS['border']};
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.8rem 0;
        text-align: center; font-family: 'Inter', sans-serif;
      }}
      .patient-status .big {{
        font-family: 'JetBrains Mono', monospace; font-size: 1.6rem;
        color: {COLORS['text']}; font-weight: 600;
      }}
      .patient-status .small {{
        font-size: 0.7rem; color: {COLORS['text_dim']};
        text-transform: uppercase; letter-spacing: 0.1em;
      }}
      .patient-btn button {{
        font-size: 1.05rem !important; padding: 1rem 1.2rem !important;
        font-weight: 700 !important; letter-spacing: 0.04em;
      }}
      .patient-btn .stButton>button {{
        border-radius: 12px !important;
      }}
      .patient-alert button {{
        background: linear-gradient(135deg, {COLORS['danger']}, #b91c1c) !important;
      }}
      .patient-record-start button {{
        background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent2']}) !important;
      }}
      .patient-record-stop button {{
        background: linear-gradient(135deg, {COLORS['warn']}, #b45309) !important;
      }}
      .patient-history-row {{
        background: {COLORS['panel']}; border: 1px solid {COLORS['border']};
        border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0;
        font-family: 'Inter', sans-serif; font-size: 0.82rem;
        color: {COLORS['text']}; display:flex; justify-content:space-between;
        align-items:center; gap: 0.6rem;
      }}
      .patient-history-row .meta {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        color: {COLORS['text_dim']};
      }}
      .patient-history-row .pill {{
        font-size: 0.65rem; font-weight: 700; letter-spacing: 0.05em;
        padding: 3px 9px; border-radius: 999px; text-transform: uppercase;
        white-space: nowrap;
      }}

      /* ── Bottom navigation bar ─────────────────────────────────────── */
      .patient-navbar {{
        position: sticky; bottom: 0; left: 0; right: 0;
        background: {COLORS['panel']}; border-top: 1px solid {COLORS['border']};
        margin: 1rem -1rem -1rem -1rem; padding: 0.4rem 0.4rem calc(0.4rem + env(safe-area-inset-bottom));
        box-shadow: 0 -4px 14px rgba(0,0,0,0.05); z-index: 999;
      }}
      .patient-navbar .stButton>button {{
        border: none !important; background: transparent !important;
        border-radius: 10px !important; padding: 0.5rem 0.2rem !important;
        font-size: 0.68rem !important; font-weight: 600 !important;
        color: {COLORS['text_dim']} !important; width: 100%;
        text-transform: uppercase; letter-spacing: 0.04em;
      }}
      .patient-navbar .nav-active button {{
        background: {COLORS['panel2']} !important; color: {COLORS['accent']} !important;
      }}

      /* ── Result / status banner ────────────────────────────────────── */
      .patient-banner {{
        border-radius: 14px; padding: 1.3rem 1.4rem; text-align: center;
        margin: 0.6rem 0 1rem; border: 1px solid;
      }}
      .patient-banner.ok {{
        background: rgba(21,128,61,0.08); border-color: {COLORS['success']};
      }}
      .patient-banner.review {{
        background: rgba(220,38,38,0.08); border-color: {COLORS['danger']};
      }}
      .patient-banner .headline {{
        font-family: 'Sora', sans-serif; font-size: 1.4rem; font-weight: 700;
        letter-spacing: 0.02em;
      }}
      .patient-banner.ok .headline {{ color: {COLORS['success']}; }}
      .patient-banner.review .headline {{ color: {COLORS['danger']}; }}
      .patient-banner .desc {{
        font-size: 0.82rem; color: {COLORS['text_mid']}; margin-top: 6px;
        line-height: 1.5;
      }}

      .patient-metric-row {{ display:flex; gap:0.6rem; margin: 0.6rem 0; }}
      .patient-metric {{
        flex:1; background: {COLORS['panel']}; border:1px solid {COLORS['border']};
        border-radius: 10px; padding: 0.7rem 0.5rem; text-align:center;
      }}
      .patient-metric .v {{
        font-family:'JetBrains Mono', monospace; font-size:1.15rem;
        font-weight:700; color:{COLORS['text']};
      }}
      .patient-metric .l {{
        font-size:0.65rem; color:{COLORS['text_dim']}; text-transform:uppercase;
        letter-spacing:0.06em; margin-top:2px;
      }}

      .patient-factor {{
        background:{COLORS['panel2']}; border-radius:8px; padding:0.6rem 0.8rem;
        margin:0.4rem 0; font-size:0.82rem; color:{COLORS['text']};
      }}
      .patient-section-title {{
        font-size: 0.72rem; color:{COLORS['text_dim']}; text-transform:uppercase;
        letter-spacing:0.08em; margin: 1rem 0 0.4rem; font-weight:700;
      }}
    </style>
    """, unsafe_allow_html=True)


def _patient_load_result(recording_id: int):
    """Load a saved recording and compute its quick result, cached in
    session_state so switching screens doesn't reprocess the signal."""
    cache = st.session_state.setdefault("patient_result_cache", {})
    if recording_id in cache:
        return cache[recording_id]
    signal, meta = load_recording_from_server(recording_id)
    if signal is None or len(signal) == 0:
        return None
    fs = int((meta or {}).get("fs") or FS)
    result = compute_quick_result(signal, fs)
    result["signal"] = signal
    result["fs"] = fs
    result["meta"] = meta
    cache[recording_id] = result
    return result


def _patient_banner(label: str, prob: float):
    is_afib = label == "AFib"
    cls = "review" if is_afib else "ok"
    headline = "POSSIBLE AFib" if is_afib else "NORMAL"
    desc = (
        "Possible irregular rhythm detected. Consider sharing this with your doctor."
        if is_afib else
        "No abnormal rhythm detected in this recording."
    )
    st.markdown(f"""
    <div class="patient-banner {cls}">
      <div class="headline">{headline}</div>
      <div class="desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def _patient_metrics_row(prob, hr, quality):
    st.markdown(f"""
    <div class="patient-metric-row">
      <div class="patient-metric"><div class="v">{prob*100:.0f}%</div><div class="l">AFib Prob.</div></div>
      <div class="patient-metric"><div class="v">{hr:.0f}</div><div class="l">Heart Rate</div></div>
      <div class="patient-metric"><div class="v">{quality}</div><div class="l">Signal</div></div>
    </div>
    """, unsafe_allow_html=True)


def _patient_result_screen(recording_id: int):
    """PATIENT — RESULT screen. Shown after a recording, or tapped from
    Home/History. Deliberately avoids raw HRV numbers/charts up front —
    those live behind 'View ECG' / 'Why was this flagged?'."""
    result = _patient_load_result(recording_id)
    if result is None:
        st.error("Couldn't load that recording. It may still be processing.")
        if st.button("← Back", key="result_back_err"):
            st.session_state["patient_result_id"] = None
            st.rerun()
        return

    meta = result.get("meta") or {}
    ts = (meta.get("started_at") or "")[:16].replace("T", " ")
    if ts:
        st.caption(f"Recording #{recording_id} · {ts}")

    _patient_banner(result["label"], result["prob"])
    _patient_metrics_row(result["prob"], result["hr"], result["quality"])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📈  View ECG", use_container_width=True, key="result_view_ecg"):
            st.session_state["patient_show_ecg"] = not st.session_state.get("patient_show_ecg", False)
    with c2:
        if st.button("❓  Why was this flagged?", use_container_width=True, key="result_why"):
            st.session_state["patient_show_why"] = True
            st.rerun()

    if st.session_state.get("patient_show_ecg"):
        st.plotly_chart(
            plot_ecg(result["signal"], result["peaks"], fs=result["fs"],
                     title="Your ECG recording", is_afib=(result["label"] == "AFib")),
            use_container_width=True,
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='patient-btn patient-alert'>", unsafe_allow_html=True)
    if st.button("📤  Share with Doctor", use_container_width=True, key="result_share"):
        user_id = st.session_state.get("auth_user_id")
        r = _server_post(
            "/api/alert",
            {"device_id": DEFAULT_DEVICE_ID, "user_id": user_id, "kind": "share_result",
             "message": f"Patient shared recording #{recording_id} "
                        f"({result['label']}, {result['prob']*100:.0f}%) with their doctor."},
        )
        if r and r.get("ok"):
            st.toast("✅ Shared with your doctor.", icon="🫀")
        else:
            st.error("Couldn't share right now — check your connection.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='text-align:center; font-size:0.7rem; color:{COLORS['text_dim']}; margin-top:0.6rem;'>"
        "⚠ Research tool only. Not a medical diagnosis.</div>",
        unsafe_allow_html=True,
    )

    if st.button("← Back", key="result_back"):
        st.session_state["patient_result_id"] = None
        st.session_state["patient_show_ecg"] = False
        st.rerun()


def _patient_why_screen(recording_id: int):
    """PATIENT — WHY WAS THIS FLAGGED? Patient-friendly XAI, with a
    technical breakdown tucked away in an expander."""
    result = _patient_load_result(recording_id)
    if result is None:
        st.error("Couldn't load that recording.")
        if st.button("← Back", key="why_back_err"):
            st.session_state["patient_show_why"] = False
            st.rerun()
        return

    is_afib = result["label"] == "AFib"
    st.markdown(f"""
    <div class="patient-hero" style="padding-top:0.3rem;">
      <h1 style="font-size:1.25rem;">Why was this flagged?</h1>
      <div class="sub">The AI identified patterns associated with an
        {'irregular' if is_afib else 'a typical'} heart rhythm.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='patient-section-title'>Main contributing factors</div>", unsafe_allow_html=True)
    reasons = result.get("reasons") or {}
    top_reasons = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
    plain_language = {
        "SDNN (variability)": "Overall heart-rate variability contributed to the prediction",
        "pNN50": "Beat-to-beat timing changes contributed to the prediction",
        "Irregularity score": "R-R intervals were irregular",
        "RMSSD": "Short-term heart-rate variability contributed to the prediction",
        "LF/HF imbalance": "Balance between heart-rhythm rhythms contributed to the prediction",
        "CV of RR": "Beat-to-beat timing spread contributed to the prediction",
    }
    if top_reasons:
        for name, _v in top_reasons:
            st.markdown(
                f"<div class='patient-factor'>• {plain_language.get(name, name)}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div class='patient-factor'>• R-R intervals were irregular</div>"
            "<div class='patient-factor'>• Heart-rate variability contributed to the prediction</div>",
            unsafe_allow_html=True,
        )

    with st.expander("🔍  View technical explanation"):
        st.caption(f"Method: {result['method_note']}")
        for k, v in reasons.items():
            bw = int(v * 100)
            clr = COLORS["danger"] if v > 0.6 else COLORS["warn"] if v > 0.3 else COLORS["success"]
            st.markdown(
                f"<div style='font-size:.78rem;color:{COLORS['text_mid']};margin:2px 0'>{k}"
                f"<span style='float:right;color:{clr};font-family:JetBrains Mono'>{v*100:.0f}%</span></div>"
                f"<div style='background:{COLORS['panel2']};border-radius:4px;height:6px;margin-bottom:8px'>"
                f"<div style='width:{bw}%;background:{clr};height:100%;border-radius:4px'></div></div>",
                unsafe_allow_html=True,
            )
        st.metric("RMSSD", f"{result['rmssd']:.1f} ms")
        st.metric("R-Peaks Detected", str(result["n_peaks"]))

    if st.button("← Back to result", key="why_back"):
        st.session_state["patient_show_why"] = False
        st.rerun()


def _patient_home():
    user = st.session_state.get("auth_user", "Patient")
    st.markdown(f"""
    <div class="patient-hero">
      <div class="heart">🫀</div>
      <h1>AFibAI</h1>
      <div class="sub">Welcome, <strong>{user}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    history = list_history(limit=1)
    if history:
        latest = history[0]
        result = _patient_load_result(latest["id"])
        if result:
            _patient_banner(result["label"], result["prob"])
            _patient_metrics_row(result["prob"], result["hr"], result["quality"])
            ts = (latest.get("started_at") or "")[:16].replace("T", " ")
            st.caption(f"Last recording: {ts}")
            if st.button("View full result →", key="home_view_result", use_container_width=True):
                st.session_state["patient_result_id"] = latest["id"]
                st.rerun()
    else:
        st.markdown(
            f"<div class='patient-status'><div class='small'>Current Status</div>"
            f"<div class='big' style='color:{COLORS['text_dim']}'>No recordings yet</div></div>",
            unsafe_allow_html=True,
        )

    # ── Wio connection status (live, polled every 2s) ──────────────────────
    @st.fragment(run_every="2s")
    def _connection_card():
        status = _server_get(f"/api/wio/status?device_id={DEFAULT_DEVICE_ID}")
        if status is None:
            st.markdown(
                f"<div class='patient-status'>"
                f"<div class='small'>Wio connection</div>"
                f"<div class='big' style='color:{COLORS['danger']}'>🔴 Server offline</div>"
                f"<div class='small' style='margin-top:6px;'>Ask your clinician to start the server.</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            return
        connected = status.get("connected", False)
        age = status.get("last_chunk_age_s")
        age_str = f"{age:.1f}s ago" if age is not None and age < 999 else "—"
        recording = status.get("recording_active", False)
        if recording:
            msg_color, msg = COLORS["warn"], "🟠 Recording in progress"
        elif connected:
            msg_color, msg = COLORS["success"], "🟢 Connected — sensing ECG"
        else:
            msg_color, msg = COLORS["text_dim"], "⏳ Waiting for Wio…"
        total = status.get("total_samples_received", 0)
        st.markdown(
            f"<div class='patient-status'>"
            f"<div class='small'>Wio connection</div>"
            f"<div class='big' style='color:{msg_color}'>{msg}</div>"
            f"<div class='small' style='margin-top:6px;'>Last sample: {age_str} · "
            f"Total: {total} samples</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    _connection_card()

    st.markdown("<div class='patient-btn patient-record-start'>", unsafe_allow_html=True)
    if st.button("⏺  Start Recording", use_container_width=True, key="home_start_recording"):
        st.session_state["patient_nav"] = "Record"
        st.session_state["patient_record_step"] = "prep"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='patient-btn patient-alert'>", unsafe_allow_html=True)
    if st.button("⚠  Alert the Doctor", use_container_width=True, key="home_alert"):
        user_id = st.session_state.get("auth_user_id")
        r = _server_post(
            "/api/alert",
            {"device_id": DEFAULT_DEVICE_ID, "user_id": user_id,
             "kind": "symptom", "message": "Patient pressed 'Alert the Doctor'"},
        )
        if r and r.get("ok"):
            st.toast("✅ Alert sent to the clinician.", icon="🫀")
        else:
            st.error(f"Could not send alert: {r}")
    st.markdown("</div>", unsafe_allow_html=True)


def _patient_record():
    st.markdown("<div class='patient-section-title'>Record</div>", unsafe_allow_html=True)
    step = st.session_state.get("patient_record_step", "prep")
    rec_state = _server_get(f"/api/recording/active?device_id={DEFAULT_DEVICE_ID}")
    is_recording = bool(rec_state and rec_state.get("active"))
    if is_recording:
        step = "recording"

    if step == "prep" and not is_recording:
        st.markdown(f"""
        <div class="patient-status" style="text-align:left;">
          <ol style="margin:0; padding-left:1.1rem; font-size:0.88rem; color:{COLORS['text']}; line-height:1.9;">
            <li>Sit still</li>
            <li>Attach the electrodes</li>
            <li>Keep your arm still</li>
            <li>Follow the electrode placement diagram on your Wio device</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='patient-btn patient-record-start'>", unsafe_allow_html=True)
        if st.button("✅  Ready to Record", use_container_width=True, key="record_ready"):
            user_id = st.session_state.get("auth_user_id")
            r = _server_post(
                "/api/recording/start",
                {"device_id": DEFAULT_DEVICE_ID, "user_id": user_id, "fs": FS},
            )
            if r and r.get("ok"):
                st.session_state["patient_record_step"] = "recording"
                st.rerun()
            else:
                st.error(f"Could not start recording: {r}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # step == "recording"
    @st.fragment(run_every="1s")
    def _recording_fragment():
        rec_state = _server_get(f"/api/recording/active?device_id={DEFAULT_DEVICE_ID}")
        if not rec_state or not rec_state.get("active"):
            st.info("Recording ended.")
            return
        dur = float(rec_state.get("duration_s", 0.0))
        n_samples = int(rec_state.get("n_samples", 0))
        live = _server_get(f"/api/wio/stream?device_id={DEFAULT_DEVICE_ID}")
        hr_str, quality = "—", "—"
        if live and live.get("samples"):
            tail = np.asarray(live["samples"][-min(len(live["samples"]), FS * 10):], dtype=np.float32)
            if len(tail) >= FS * 3:
                peaks = detect_rpeaks(tail, FS)
                if len(peaks) >= 2:
                    rr = np.diff(peaks) / FS * 1000
                    rr = rr[(rr > 250) & (rr < 2000)]
                    if len(rr):
                        hr_str = f"{60000.0/np.mean(rr):.0f}"
                quality, _ = estimate_signal_quality(tail, FS)
        st.markdown(
            f"<div class='patient-status' style='border-color:{COLORS['warn']};'>"
            f"<div class='small'>Recording</div>"
            f"<div class='big' style='color:{COLORS['warn']}'>{dur:.0f}s</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="patient-metric-row">
          <div class="patient-metric"><div class="v">{hr_str}</div><div class="l">Heart Rate</div></div>
          <div class="patient-metric"><div class="v">{quality}</div><div class="l">Signal</div></div>
        </div>
        """, unsafe_allow_html=True)
        if live and live.get("samples") and len(live["samples"]) > FS * 2:
            tail = np.asarray(live["samples"][-FS * 8:], dtype=np.float32)
            st.plotly_chart(
                plot_ecg(tail, detect_rpeaks(tail, FS), fs=FS, title="Live ECG preview"),
                use_container_width=True,
            )
    _recording_fragment()

    st.markdown("<div class='patient-btn patient-record-stop'>", unsafe_allow_html=True)
    if st.button("⏹  Stop Recording", use_container_width=True, key="record_stop"):
        r = _server_post("/api/recording/stop", {"device_id": DEFAULT_DEVICE_ID})
        if r and r.get("ok"):
            st.session_state["patient_record_step"] = "prep"
            st.session_state["patient_result_id"] = r.get("recording_id")
            st.session_state.setdefault("patient_result_cache", {}).pop(r.get("recording_id"), None)
            st.rerun()
        else:
            st.error(f"Could not stop recording: {r}")
    st.markdown("</div>", unsafe_allow_html=True)


def _patient_history():
    st.markdown("<div class='patient-section-title'>My Recordings</div>", unsafe_allow_html=True)
    history = list_history(limit=20)
    if not history:
        st.caption("You haven't recorded anything yet. Tap **Record** below to get started.")
        return
    for rec in history:
        ts = (rec.get("started_at") or "")[:16].replace("T", " ")
        dur = float(rec.get("duration_s") or 0)
        result = st.session_state.get("patient_result_cache", {}).get(rec["id"])
        if result is not None:
            is_afib = result["label"] == "AFib"
            pill_color = COLORS["danger"] if is_afib else COLORS["success"]
            pill_text = "Review" if is_afib else "Normal"
            pill_html = (f"<span class='pill' style='background:{pill_color}22;"
                         f"color:{pill_color};'>{pill_text}</span>")
        else:
            pill_html = "<span class='pill' style='background:#eee;color:#888;'>—</span>"
        cols = st.columns([5, 1])
        with cols[0]:
            st.markdown(
                f"<div class='patient-history-row'>"
                f"<span>{ts} · {dur:.0f}s</span>{pill_html}"
                f"<span class='meta'>#{rec['id']}</span></div>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.button("View", key=f"hist_view_{rec['id']}"):
                st.session_state["patient_result_id"] = rec["id"]
                st.rerun()


def _patient_profile():
    user = st.session_state.get("auth_user", "Patient")
    st.markdown("<div class='patient-section-title'>Profile</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="patient-status" style="text-align:left;">
      <div style="font-size:0.8rem; color:{COLORS['text_mid']}; line-height:2;">
        <strong>Patient ID:</strong> {user}<br>
        <strong>Device:</strong> {DEFAULT_DEVICE_ID}<br>
        <strong>Connected device status:</strong> see Home tab
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔔  Notifications"):
        st.caption("Alerts to your clinician are sent instantly when you tap "
                   "**Alert the Doctor** or **Share with Doctor**.")
    with st.expander("🔒  Privacy"):
        st.caption("Your ECG recordings and history are only visible to you "
                   "and your care team.")
    with st.expander("❓  Help"):
        st.caption("Having trouble with electrode placement or the device? "
                   "Contact your clinic.")
    with st.expander("📄  Legal / Agreement"):
        st.caption("AFibAI is a research/screening prototype, not a diagnostic "
                   "medical device. See the agreement you accepted at sign-in.")
    with st.expander("ℹ️  About"):
        st.caption("AFibAI — ECG monitoring & AFib screening research prototype.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("Sign out", key="patient_signout", use_container_width=True):
        for k in ("authenticated", "auth_user", "auth_user_id", "auth_role",
                  "auth_stage", "agreement_accepted", "patient_nav",
                  "patient_result_id", "patient_show_why", "patient_show_ecg",
                  "patient_record_step", "patient_result_cache"):
            st.session_state.pop(k, None)
        st.rerun()


def patient_view():
    """Phone-friendly patient UI.

    "What did my recording mean, and what has happened over time?"
    Four sections (Home / Record / History / Profile), plus a Result screen
    and a patient-friendly "Why was this flagged?" explanation. Patients see
    simplified results — probability, HR, signal quality, plain-language
    explanation — never the raw HRV/SHAP dashboard (that's the doctor view).
    """
    _patient_css()
    st.session_state.setdefault("patient_nav", "Home")

    # ── Overlay screens take priority over the tab content ─────────────────
    if st.session_state.get("patient_result_id") is not None:
        if st.session_state.get("patient_show_why"):
            _patient_why_screen(st.session_state["patient_result_id"])
        else:
            _patient_result_screen(st.session_state["patient_result_id"])
        return

    nav = st.session_state["patient_nav"]
    if nav == "Home":
        _patient_home()
    elif nav == "Record":
        _patient_record()
    elif nav == "History":
        _patient_history()
    elif nav == "Profile":
        _patient_profile()

    # ── Bottom navigation ────────────────────────────────────────────────
    st.markdown("<div class='patient-navbar'>", unsafe_allow_html=True)
    tabs = [("Home", "🏠"), ("Record", "⏺"), ("History", "🕘"), ("Profile", "👤")]
    cols = st.columns(4)
    for col, (name, icon) in zip(cols, tabs):
        with col:
            active = "nav-active" if nav == name else ""
            st.markdown(f"<div class='{active}'>", unsafe_allow_html=True)
            if st.button(f"{icon}\n{name}", key=f"nav_{name}", use_container_width=True):
                st.session_state["patient_nav"] = name
                st.session_state["patient_result_id"] = None
                st.session_state["patient_show_why"] = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN (clinician view)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Patient alerts (consume so they're shown once) ──────────────────────
    alerts = _server_get("/api/alerts?consume=1") or {}
    seen_alerts = st.session_state.setdefault("doctor_seen_alerts", [])
    for a in alerts.get("alerts", []):
        st.toast(
            f"🚨 Patient alert: {a.get('message') or a.get('kind','')} "
            f"· device {a.get('device_id','?')}",
            icon="🚨",
        )
        seen_alerts.append(a)
    # Keep only the most recent alerts around for the Dashboard.
    st.session_state["doctor_seen_alerts"] = seen_alerts[-20:]

    with st.sidebar:
        st.markdown(f"""
        <div style='padding:1rem 0 0.8rem;'>
          <div style='font-size:1.8rem; margin-bottom:6px;'>🫀</div>
          <div style='font-family:"Sora",sans-serif; font-size:1.3rem; color:{COLORS["text"]};
                      font-weight:700; line-height:1;'>AFibAI</div>
          <div style='font-family:"JetBrains Mono",monospace; font-size:0.55rem;
                      color:{COLORS["text_dim"]}; letter-spacing:0.12em; margin-top:4px;'>
            DOCTOR / RESEARCH MODE
          </div>
          <div style='font-size:0.7rem; color:{COLORS["text_mid"]}; margin-top:6px; line-height:1.5;'>
            ECG Signal Analysis<br>AFib Detection Engine
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown(f'<div class="cs-label">Input Source</div>', unsafe_allow_html=True)

        _mode_options = ["Demo ECG", "Upload .npy file", "Upload .csv file", "Wio Live (Server)"]
        # Honor an override set by the History → "Load" buttons, then clear it
        # so the user can switch modes again afterwards.
        _override = st.session_state.pop("input_mode_override", None)
        _default_idx = _mode_options.index(_override) if _override in _mode_options else 0

        input_mode = st.radio(
            "Input Source",
            _mode_options,
            index=_default_idx,
            label_visibility="collapsed",
        )

        if input_mode == "Demo ECG":
            demo_choice = st.selectbox(
                "Demo ECG",
                list(DEMO_FILES.keys())
            )

        if input_mode == "ESP32 (WiFi)":
            st.markdown(f'<div class="cs-label">ESP32 Connection</div>', unsafe_allow_html=True)
            esp32_ip = st.text_input("ESP32 IP address", value="192.168.4.1",
                                      placeholder="e.g. 192.168.1.42")
            esp32_port = st.number_input("Port", min_value=1, max_value=65535, value=80, step=1)
            esp32_path = st.text_input("Endpoint path", value="/ecg",
                                        help="Path on the ESP32 that returns the ECG samples, "
                                             "as JSON array or comma-separated values.")
            fetch_clicked = st.button("🔄  Fetch from ESP32", use_container_width=True)
            st.caption(
                "Expects the ESP32 to serve raw ECG samples over HTTP — either a JSON "
                "array of numbers, or a comma/newline-separated list of values."
            )

        st.divider()
        st.markdown(f'<div class="cs-label">Signal Settings</div>', unsafe_allow_html=True)
        fs_input      = st.number_input("Sampling rate (Hz)", min_value=64, max_value=2000,
                                        value=128, step=1)
        window_index  = st.number_input("Window index (multi-window files)",
                                        min_value=0, value=0, step=1)

        st.divider()
        st.markdown(f'<div class="cs-label">Model Selection</div>', unsafe_allow_html=True)

        MODEL_OPTIONS = ["Random Forest", "XGBoost", "CatBoost", "Ensemble ⭐"]
        model_choice_raw = st.radio(
            "Model",
            MODEL_OPTIONS,
            index=3,
            label_visibility="collapsed",
        )
        model_choice = model_choice_raw.replace(" ⭐", "")

        show_individual = True
        if model_choice == "Ensemble":
            show_individual = st.checkbox("Show individual model predictions", value=False)

        st.divider()
        with st.expander("🛠️ Model path debug", expanded=False):
            st.caption(f"app.py location: `{APP_DIR}`")
            for name, path in MODEL_PATHS.items():
                exists = Path(path).exists()
                dot = "🟢" if exists else "🔴"
                st.markdown(f"{dot} **{name}**")
                st.code(path, language=None)

        st.divider()
        st.markdown(f'<div class="cs-label">Recording History</div>', unsafe_allow_html=True)
        history = list_history(limit=20)
        if not history:
            st.caption(
                "No recordings yet. The patient can tap **Start recording** "
                "in their view, or `mock_wio.py` can simulate one."
            )
        else:
            for rec in history:
                rid = rec["id"]
                label = recording_summary(rec)
                # Each row: button to load the recording into the analysis view.
                if st.button(
                    f"#{rid}  ·  {label}",
                    key=f"history_load_{rid}",
                    use_container_width=True,
                    help="Click to load this recording into the analysis view.",
                ):
                    sig, meta = load_recording_from_server(int(rid))
                    if sig is not None and len(sig) > 0:
                        st.session_state["wio_signal"] = sig
                        st.session_state["wio_label"] = (
                            f"History #{rid}  ·  {meta.get('device_id','?')}  ·  "
                            f"{meta.get('duration_s',0):.1f}s"
                        )
                        st.session_state["wio_device_id"] = meta.get("device_id", DEFAULT_DEVICE_ID)
                        st.session_state["input_mode_override"] = "Wio Live (Server)"
                        st.session_state["doctor_nav"] = "ECG Analysis"
                        st.rerun()
                    else:
                        st.error("Could not load that recording.")
            if st.button("🔄  Refresh history", key="history_refresh",
                          use_container_width=True):
                st.rerun()

        st.divider()
        st.markdown(f'<div class="cs-label">Library Status</div>', unsafe_allow_html=True)
        for name, ok in [
            ("Random Forest", SKLEARN_AVAILABLE),
            ("XGBoost",       XGB_AVAILABLE),
            ("CatBoost",      CATBOOST_AVAILABLE),
            ]:
            dot = "🟢" if ok else "🔴"
            st.markdown(f'<div class="cs-badge">{dot} {name}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown(f'<div class="cs-label">Legal</div>', unsafe_allow_html=True)
        if st.button("📄  ข้อตกลง / Agreement",
                      key="sidebar_open_agreement",
                      use_container_width=True,
                      help="View the software license agreement."):
            _show_agreement_dialog()

        st.divider()
        if st.button("Sign out", key="doctor_signout", use_container_width=True):
            for k in ("authenticated", "auth_user", "auth_user_id", "auth_role",
                      "auth_stage", "agreement_accepted", "doctor_nav",
                      "wio_signal", "wio_label", "wio_device_id",
                      "doctor_seen_alerts"):
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown(f"""
        <div style='font-size:0.6rem; color:{COLORS["text_dim"]}; line-height:1.8;'>
          ⚠️ Research tool only.<br>Not a certified medical device.<br>Consult a physician for diagnosis.
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:{COLORS["panel"]}; border-bottom:1px solid {COLORS["border"]};
                padding:0.85rem 2rem; display:flex; align-items:center;
                justify-content:space-between; margin:-1.5rem -2rem 1.5rem;'>
      <div style='display:flex; align-items:center; gap:12px;'>
        <span style='font-size:1.6rem;'>🫀</span>
        <div>
          <span style='font-family:"Sora",sans-serif; font-size:1.25rem; color:{COLORS["text"]}; font-weight:700;'>
            AFibAI
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    signal = None
    signal_label = "Unknown"
    demo_meta = None

    if input_mode == "Demo ECG":
        try:
            signal = np.load(DEMO_FILES[demo_choice]["path"])
            signal_label = demo_choice
            demo_meta = DEMO_FILES[demo_choice]
        except FileNotFoundError:
            st.error(f"⚠️ Demo file not found at `{DEMO_FILES[demo_choice]['path']}`.")
            st.stop()

    elif input_mode == "Upload .npy file":
        uploaded_file = st.file_uploader("Upload .npy ECG File", type=["npy"])
        if uploaded_file is not None:
            signal = np.load(uploaded_file)
            signal_label = uploaded_file.name

    elif input_mode == "Upload .csv file":
        uploaded_file = st.file_uploader("Upload .csv ECG File", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            signal = df.iloc[:, 0].values
            signal_label = uploaded_file.name

    elif input_mode == "ESP32 (WiFi)":
        if fetch_clicked:
            url = f"http://{esp32_ip}:{int(esp32_port)}{esp32_path}"
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(raw)
                    values = np.array(parsed, dtype=np.float32).flatten()
                except (json.JSONDecodeError, TypeError, ValueError):
                    values = np.array(
                        [float(v) for v in raw.replace("\n", ",").split(",") if v.strip() != ""],
                        dtype=np.float32,
                    )
                if len(values) == 0:
                    st.warning(f"ESP32 at `{url}` responded but no samples were parsed.")
                else:
                    st.session_state["esp32_signal"] = values
                    st.session_state["esp32_label"] = f"ESP32 @ {esp32_ip}{esp32_path}"
                    st.success(f"Fetched {len(values)} samples from {url}")
            except urllib.error.URLError as e:
                reason = getattr(e, "reason", e)
                st.error(f"⚠️ Could not reach ESP32 at `{url}`: {reason}")
            except Exception as e:
                st.error(f"⚠️ Failed to parse data from ESP32: {e}")

        if "esp32_signal" in st.session_state:
            signal = st.session_state["esp32_signal"]
            signal_label = st.session_state["esp32_label"]
        else:
            st.info("Enter the ESP32's IP address and click **Fetch from ESP32** to load live samples.")

    elif input_mode == "Wio Live (Server)":
        # server.py holds the live 30-second ring buffer for this device.
        # The first render shows whatever is already there; subsequent
        # renders rely on st.fragment(run_every=...) further down the page.
        status = _server_get(f"/api/wio/status?device_id={DEFAULT_DEVICE_ID}")
        if status is None:
            st.error(f"⚠️ Could not reach AFibAI server at `{AFIBAI_SERVER_URL}`. "
                     "Make sure `python server.py` is running.")
        else:
            st.session_state["wio_status"] = status
            st.session_state["wio_device_id"] = DEFAULT_DEVICE_ID
            stream = _server_get(f"/api/wio/stream?device_id={DEFAULT_DEVICE_ID}")
            if stream and stream.get("samples"):
                st.session_state["wio_signal"] = np.asarray(
                    stream["samples"], dtype=np.float32
                )
                st.session_state["wio_label"] = (
                    f"Live · {DEFAULT_DEVICE_ID} · "
                    f"{stream['n']} samples"
                )
            else:
                st.session_state.pop("wio_signal", None)

        if "wio_signal" in st.session_state:
            signal = st.session_state["wio_signal"]
            signal_label = st.session_state["wio_label"]
        else:
            st.info(
                "Waiting for the Wio (or `mock_wio.py`) to start sending samples "
                f"to `{AFIBAI_SERVER_URL}`. Once data arrives it will stream "
                "automatically."
            )

    if signal is not None:
        total_seconds = len(signal) / fs_input
        duration_str = f"{total_seconds:.1f}s"

        if demo_meta:
            start_time = float(demo_meta['time'].replace("s", ""))
            end_time = start_time + total_seconds
            time_display = f"{start_time:.1f}s — {end_time:.1f}s"

            extra_html = f"""
<div>
<div class="cs-label">Source Record</div>
<div style="font-size: 1.1rem; font-weight: 600; color: {COLORS['text']};">MIT-BIH {demo_meta['record']}</div>
</div>
<div>
<div class="cs-label">Original Time</div>
<div style="font-size: 1.1rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: {COLORS['text']};">{time_display}</div>
</div>
"""
        else:
            extra_html = ""

        st.markdown(f"""
<div class="cs-card" style="display: flex; gap: 3rem; align-items: center; padding: 1rem 1.5rem; flex-wrap: wrap;">
<div>
<div class="cs-label">File / Label</div>
<div style="font-size: 1.1rem; font-weight: 600; color: {COLORS['text']};">{signal_label}</div>
</div>
{extra_html}
<div>
<div class="cs-label">Segment Length</div>
<div style="font-size: 1.1rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: {COLORS['text']};">{duration_str}</div>
</div>
<div>
<div class="cs-label">Sampling Rate</div>
<div style="font-size: 1.1rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: {COLORS['text']};">{fs_input} Hz</div>
</div>
</div>
""", unsafe_allow_html=True)

    if signal is not None and len(signal) > 0:
        view_s = len(signal) / fs_input
        n_view = len(signal)

        # ── Live connection indicator ─────────────────────────────────────
        # In Wio Live mode, re-check the server every second so the
        # "connected" dot and last-update time stay fresh. The fragment
        # runs independently of the rest of the page, so it ticks without
        # re-rendering the analysis or plots below.
        if input_mode == "Wio Live (Server)":
            @st.fragment(run_every="1s")
            def _live_status():
                status = _server_get(f"/api/wio/status?device_id={DEFAULT_DEVICE_ID}")
                if status is None:
                    st.markdown(
                        f"<div class='cs-badge' style='border-color:{COLORS['danger']};'>"
                        "🔴 Server unreachable</div>",
                        unsafe_allow_html=True,
                    )
                    return
                connected = status.get("connected", False)
                age = status.get("last_chunk_age_s")
                age_str = (
                    f"{age:.1f}s ago" if age is not None and age < 999
                    else "never"
                )
                recording = status.get("recording_active", False)
                dot = "🟢" if connected else "🔴"
                rec_dot = "🔴 REC" if recording else ""
                st.markdown(
                    f"<div class='cs-badge'>{dot} Wio {age_str} &nbsp; "
                    f"<strong>{status.get('total_samples_received', 0)}</strong> samples "
                    f"&nbsp; {rec_dot}</div>",
                    unsafe_allow_html=True,
                )
            _live_status()

            # Also poll the latest samples — this is what makes the ECG
            # scroll and the analysis update as new data arrives. The
            # analysis window (sw_slider) defaults to the last 15s.
            live = _server_get(f"/api/wio/stream?device_id={DEFAULT_DEVICE_ID}")
            if live and live.get("samples"):
                st.session_state["wio_signal"] = np.asarray(
                    live["samples"], dtype=np.float32
                )
                st.session_state["wio_label"] = (
                    f"Live · {DEFAULT_DEVICE_ID} · {live['n']} samples"
                )
                # Force the sw_slider to "follow the latest" — snap to end.
                buf_seconds = live["n"] / float(live.get("fs") or fs_input)
                snap_to = max(0.0, buf_seconds - SLIDE_WIN_SEC)
                if abs(st.session_state.get("sw_slider", -1) - snap_to) > 0.5:
                    st.session_state["sw_slider"] = snap_to
                signal = st.session_state["wio_signal"]
                signal_label = st.session_state["wio_label"]

        # ─────────────────────────────────────────────────────────────────
        # ANALYSIS WINDOW
        # A SLIDE_WIN_SEC-wide window you scroll across the recording.
        # Every prediction, metric, and plot below reflects only the
        # samples inside this window — there is no separate "whole
        # signal" prediction.
        #
        # The slider widget itself is rendered inside the ECG Analysis
        # tab, right under the ECG plot. We read its current value from
        # session_state here so the rest of the page can react to it
        # immediately.
        # ─────────────────────────────────────────────────────────────────
        window_len_s = min(SLIDE_WIN_SEC, view_s)
        max_start = max(0.0, view_s - window_len_s)

        sw_pos = float(st.session_state.get("sw_slider", 0.0))
        sw_pos = min(sw_pos, max_start)

        start_sample = int(round(sw_pos * fs_input))
        end_sample   = start_sample + int(round(window_len_s * fs_input))
        window_sig   = signal[start_sample:end_sample]
        window_range = (sw_pos, sw_pos + window_len_s)

        with st.spinner("Analyzing window…"):
            proc     = preprocess(window_sig, fs=fs_input)
            peaks    = detect_rpeaks(window_sig, fs=fs_input)
            rr_ms    = np.diff(peaks) / fs_input * 1000
            rr_ms    = rr_ms[(rr_ms > 250) & (rr_ms < 2000)]
            features = extract_hrv(window_sig, fs=fs_input)
            feat     = dict(zip(FEATURE_NAMES, features))

        label, prob, threshold, method_note, reasons, individual_preds = run_prediction(
            model_choice, features, silent=False
        )
        is_afib = label == "AFib"
        quality, quality_score = estimate_signal_quality(window_sig, fs_input)

        # Full-signal trace, used only to draw the ECG plot with the window
        # highlighted on it — all numbers still come from the window above.
        proc_full  = preprocess(signal, fs=fs_input)
        peaks_full = detect_rpeaks(signal, fs=fs_input)

        df_feat = pd.DataFrame([
            {"Feature": n, "Value": f"{v:.4f}", "Unit": FEATURE_UNITS.get(n, ""),
             "Description": FEATURE_DESCRIPTIONS.get(n, "")}
            for n, v in zip(FEATURE_NAMES, features)
        ])
    else:
        label = prob = threshold = method_note = reasons = individual_preds = None
        is_afib = False
        quality, quality_score = "—", 0.0
        peaks = np.array([]); rr_ms = np.array([]); features = None; feat = {}
        proc_full = peaks_full = None
        window_range = (0.0, 0.0); window_len_s = 0.0; max_start = 0.0; sw_pos = 0.0
        view_s = 0.0
        df_feat = None

    # ═════════════════════════════════════════════════════════════════════
    # DOCTOR NAVIGATION — Dashboard / Patients / Recordings / ECG Analysis /
    # HRV / AI-XAI / Trends / Reports  (per AFibAI interface spec)
    # ═════════════════════════════════════════════════════════════════════
    section_names = [
        "🏠 Dashboard", "👥 Patients", "📼 Recordings", "🫀 ECG Analysis",
        "💓 HRV", "🤖 AI / XAI", "📈 Trends", "📄 Reports",
    ]
    all_history = list_history(limit=100)

    tabs = st.tabs(section_names)

    # ── DASHBOARD ────────────────────────────────────────────────────────
    with tabs[0]:
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_count = sum(1 for rec in all_history if (rec.get("started_at") or "").startswith(today_str))
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Recordings", str(len(all_history)))
        d2.metric("Today's Recordings", str(today_count))
        d3.metric("Recent Alerts", str(len(st.session_state.get("doctor_seen_alerts", []))))
        d4.metric("Current Signal Quality", quality if signal is not None else "—")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="cs-label">Recent Alerts</div>', unsafe_allow_html=True)
        recent = list(reversed(st.session_state.get("doctor_seen_alerts", [])))[:8]
        if not recent:
            st.caption("No alerts received yet. Patient alerts (symptom reports, shared "
                       "results) will appear here as they come in.")
        else:
            for a in recent:
                kind = a.get("kind", "alert")
                pill = "⚠️" if kind == "symptom" else "📤"
                st.markdown(
                    f"<div class='cs-card' style='padding:0.7rem 1.1rem; margin-bottom:0.4rem;'>"
                    f"<span style='font-family:JetBrains Mono; font-size:0.8rem;'>{pill} "
                    f"device {a.get('device_id','?')}</span> — "
                    f"<span style='color:{COLORS['text_mid']}; font-size:0.82rem;'>"
                    f"{a.get('message','')}</span></div>",
                    unsafe_allow_html=True,
                )

        if signal is not None:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown(f'<div class="cs-label">Current Recording — Quick Summary</div>', unsafe_allow_html=True)
            if is_afib:
                st.markdown(f"""
                <div class='cs-alert cs-alert-afib'>
                  <span style='font-size:1.5rem; flex-shrink:0;'>⚠️</span>
                  <div>
                    <div style='font-weight:700; font-size:0.95rem; color:{COLORS["danger"]}; font-family:"Sora",sans-serif;'>
                      Possible AFib
                    </div>
                    <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-top:3px;'>
                      AFib probability: <strong>{prob*100:.1f}%</strong> — flagged for review.
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='cs-alert cs-alert-normal'>
                  <span style='font-size:1.5rem; flex-shrink:0;'>✅</span>
                  <div>
                    <div style='font-weight:700; font-size:0.95rem; color:{COLORS["success"]}; font-family:"Sora",sans-serif;'>
                      Normal Sinus Rhythm
                    </div>
                    <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-top:3px;'>
                      AFib probability: <strong>{prob*100:.1f}%</strong> — no AFib detected.
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Load a recording from the sidebar (or the **Recordings** tab) to see a summary here.")

    # ── PATIENTS ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.caption(
            "This prototype currently tracks a single connected device/patient "
            f"(`{DEFAULT_DEVICE_ID}`). Once multi-patient accounts are wired up "
            "server-side, this section will list every patient with their "
            "latest status."
        )
        latest_result = None
        if all_history:
            latest_rec = all_history[0]
            sig0, meta0 = load_recording_from_server(int(latest_rec["id"]))
            if sig0 is not None and len(sig0) > 0:
                latest_result = compute_quick_result(sig0, int((meta0 or {}).get("fs") or FS), model_choice)

        p_status = latest_result["label"] if latest_result else "No data"
        p_prob = f"{latest_result['prob']*100:.1f}%" if latest_result else "—"
        pill_color = COLORS["danger"] if latest_result and latest_result["label"] == "AFib" else COLORS["success"]
        st.markdown(f"""
        <div class="cs-card">
          <div class="cs-label">Patient · Device {DEFAULT_DEVICE_ID}</div>
          <div style="display:flex; gap:2.5rem; flex-wrap:wrap; align-items:center;">
            <div>
              <div style="font-size:0.7rem; color:{COLORS['text_dim']}; text-transform:uppercase;">Status</div>
              <div style="font-weight:700; color:{pill_color}; font-family:'Sora',sans-serif; font-size:1.1rem;">{p_status}</div>
            </div>
            <div>
              <div style="font-size:0.7rem; color:{COLORS['text_dim']}; text-transform:uppercase;">AFib Probability</div>
              <div style="font-weight:700; font-family:'JetBrains Mono',monospace; font-size:1.1rem;">{p_prob}</div>
            </div>
            <div>
              <div style="font-size:0.7rem; color:{COLORS['text_dim']}; text-transform:uppercase;">Total Recordings</div>
              <div style="font-weight:700; font-family:'JetBrains Mono',monospace; font-size:1.1rem;">{len(all_history)}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Tip: open the **Trends** tab for a flagged-vs-normal breakdown over recent recordings.")

    # ── RECORDINGS ───────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(f'<div class="cs-label">All Recordings</div>', unsafe_allow_html=True)
        if not all_history:
            st.caption("No recordings yet.")
        else:
            rows_html = ""
            for rec in all_history:
                ts = (rec.get("started_at") or "")[:16].replace("T", " ")
                dur = float(rec.get("duration_s") or 0.0)
                rows_html += (
                    f"<tr><td>#{rec['id']}</td><td>{ts}</td><td>{dur:.1f}s</td>"
                    f"<td>{rec.get('n_samples', 0)}</td></tr>"
                )
            st.markdown(f"""
            <div class="cs-table-wrap">
              <table class="cs-table">
                <thead><tr><th>ID</th><th>Started</th><th>Duration</th><th>Samples</th></tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            load_id = st.selectbox(
                "Open a recording in ECG Analysis",
                [r["id"] for r in all_history],
                format_func=lambda rid: f"#{rid}  ·  " + recording_summary(
                    next(r for r in all_history if r["id"] == rid)
                ),
                key="recordings_tab_select",
            )
            if st.button("📂  Open in ECG Analysis", key="recordings_tab_open"):
                sig, meta = load_recording_from_server(int(load_id))
                if sig is not None and len(sig) > 0:
                    st.session_state["wio_signal"] = sig
                    st.session_state["wio_label"] = (
                        f"History #{load_id}  ·  {meta.get('device_id','?')}  ·  "
                        f"{meta.get('duration_s',0):.1f}s"
                    )
                    st.session_state["wio_device_id"] = meta.get("device_id", DEFAULT_DEVICE_ID)
                    st.session_state["input_mode_override"] = "Wio Live (Server)"
                    st.rerun()
                else:
                    st.error("Could not load that recording.")

    # ── ECG ANALYSIS ─────────────────────────────────────────────────────
    with tabs[3]:
        if signal is None or len(signal) == 0:
            st.markdown(
                f"""
                <div style='padding:60px 20px; text-align:center;'>
                <div style='font-size:3rem; margin-bottom:12px;'>🫀</div>
                <div style='font-size:0.9rem; color:{COLORS["text_dim"]};'>
                    Select an input source in the sidebar to begin.
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if is_afib:
                st.markdown(f"""
                <div class='cs-alert cs-alert-afib'>
                  <span style='font-size:1.5rem; flex-shrink:0;'>⚠️</span>
                  <div>
                    <div style='font-weight:700; font-size:0.95rem; color:{COLORS["danger"]}; font-family:"Sora",sans-serif;'>
                      Atrial Fibrillation Detected
                    </div>
                    <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-top:3px;'>
                      AFib probability (current window): <strong>{prob*100:.1f}%</strong> — Consult a physician immediately.
                      <br>
                      Method: <strong>{method_note}</strong> &nbsp;|&nbsp; Decision threshold: <strong>{threshold*100:.0f}%</strong>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='cs-alert cs-alert-normal'>
                  <span style='font-size:1.5rem; flex-shrink:0;'>✅</span>
                  <div>
                    <div style='font-weight:700; font-size:0.95rem; color:{COLORS["success"]}; font-family:"Sora",sans-serif;'>
                      Normal Sinus Rhythm
                    </div>
                    <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-top:3px;'>
                      AFib probability (current window): <strong>{prob*100:.1f}%</strong> — No atrial fibrillation detected.
                      <br>
                      Method: <strong>{method_note}</strong> &nbsp;|&nbsp; Decision threshold: <strong>{threshold*100:.0f}%</strong>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("AFib Probability", f"{prob*100:.1f}%",
                      delta="HIGH ⚠" if is_afib else "Normal",
                      delta_color="inverse" if is_afib else "normal")
            m2.metric("Mean Heart Rate",  f"{feat['mean_hr']:.1f} bpm")
            m3.metric("R-Peaks Detected", str(len(peaks)))
            m4.metric("Signal Quality",   quality)
            m5.metric("Window", f"{window_len_s:.0f}s")

            st.plotly_chart(
                plot_ecg(proc_full[:n_view], peaks_full[peaks_full < n_view], fs=fs_input,
                         title=f"ECG  ·  Full {view_s:.0f}s  ·  {signal_label}  ·  window {window_range[0]:.0f}s–{window_range[1]:.0f}s",
                         is_afib=is_afib,
                         window_range=window_range,
                         window_color=(COLORS["danger"] if is_afib else COLORS["accent"])),
                use_container_width=True,
            )

            st.markdown(f'<div class="cs-label">Analysis Window · {window_len_s:.0f}s wide</div>', unsafe_allow_html=True)
            if max_start > 0:
                st.slider(
                    "Scroll window across signal (s)", 0.0, float(max_start),
                    value=sw_pos, step=SLIDE_STEP_SEC,
                    key="sw_slider",
                    help="Drag to scan the recording — everything else on this "
                         "page reflects only the highlighted window.",
                    label_visibility="collapsed",
                )
            else:
                st.caption("Signal is shorter than the analysis window — using the full segment.")

    # ── HRV ──────────────────────────────────────────────────────────────
    with tabs[4]:
        if signal is None or len(signal) == 0:
            st.info("Load a recording to see HRV analysis.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("RMSSD", f"{feat['rmssd']:.1f} ms",
                      help="Root Mean Square Successive Differences — elevated in AFib")
            h2.metric("SDNN",  f"{feat['sdnn']:.1f} ms")
            h3.metric("Mean RR", f"{feat['mean_rr']:.0f} ms")
            h4.metric("Mean HR", f"{feat['mean_hr']:.1f} bpm")

            hrv_tabs = st.tabs(["💓  RR Tachogram", "🌀  Poincaré", "📊  HRV Features"])
            with hrv_tabs[0]:
                if len(rr_ms) >= 3:
                    st.plotly_chart(plot_rr(rr_ms), use_container_width=True)
                else:
                    st.warning("Not enough RR intervals detected.")
            with hrv_tabs[1]:
                if len(rr_ms) >= 4:
                    st.plotly_chart(plot_poincare(rr_ms, is_afib=is_afib), use_container_width=True)
                else:
                    st.warning("Not enough RR intervals for Poincaré plot.")
            with hrv_tabs[2]:
                left, right = st.columns([1, 1])
                with left:
                    st.markdown(f'<div class="cs-label">HRV Feature Values</div>', unsafe_allow_html=True)
                    table_rows_html = "".join(
                        f"<tr><td>{n}</td><td>{v:.4f}</td>"
                        f"<td>{FEATURE_UNITS.get(n,'')}</td><td>{FEATURE_DESCRIPTIONS.get(n,'')}</td></tr>"
                        for n, v in zip(FEATURE_NAMES, features)
                    )
                    st.markdown(f"""
                    <div class="cs-table-wrap">
                      <table class="cs-table">
                        <thead><tr><th>Feature</th><th>Value</th><th>Unit</th><th>Description</th></tr></thead>
                        <tbody>{table_rows_html}</tbody>
                      </table>
                    </div>
                    """, unsafe_allow_html=True)
                with right:
                    st.plotly_chart(plot_radar(features), use_container_width=True)
                st.download_button(
                    "⬇  Download HRV Features (CSV)",
                    data=df_feat.to_csv(index=False).encode(),
                    file_name="hrv_features.csv",
                    mime="text/csv",
                    key="hrv_tab_download",
                )

    # ── AI / XAI ─────────────────────────────────────────────────────────
    with tabs[5]:
        if signal is None or len(signal) == 0:
            st.info("Load a recording to see the model's prediction and explanation.")
        else:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("AFib Probability", f"{prob*100:.1f}%")
            a2.metric("Classification", label)
            a3.metric("Model", model_choice)
            a4.metric("Decision Threshold", f"{threshold*100:.0f}%")

            if model_choice == "Ensemble" and individual_preds:
                st.markdown(f'<div class="cs-label">Model Agreement</div>', unsafe_allow_html=True)
                rows_html = ""
                afib_votes = 0
                for name, p in individual_preds.items():
                    vote = "AFib" if p >= 0.5 else "Normal"
                    if vote == "AFib":
                        afib_votes += 1
                    vote_color = COLORS["danger"] if vote == "AFib" else COLORS["success"]
                    rows_html += (
                        f"<div class='cs-pred-row'><span>{name} → "
                        f"<span style='color:{vote_color}'>{vote}</span></span>"
                        f"<span style='color:{COLORS['text']}'>{p*100:.1f}%</span></div>"
                    )
                rows_html += (
                    f"<div style='border-top:1px solid {COLORS['border']}; margin:6px 0;'></div>"
                    f"<div class='cs-pred-row' style='color:{COLORS['accent']}; font-weight:700;'>"
                    f"<span>Final — {afib_votes}/{len(individual_preds)} models → AFib</span>"
                    f"<span>{prob*100:.1f}%</span></div>"
                )
                st.markdown(
                    f"<div class='cs-card'>{rows_html}</div>",
                    unsafe_allow_html=True,
                )

            if reasons:
                st.markdown(f'<div class="cs-label">SHAP-style Feature Importance</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style='font-size:0.8rem; color:{COLORS["text_mid"]}; margin-bottom:0.8rem; line-height:1.6;'>
                  Each bar shows how much a feature pushed toward AFib.
                  Higher scores contribute more to the overall AFib probability.
                </div>""", unsafe_allow_html=True)
                for k, v in sorted(reasons.items(), key=lambda kv: kv[1], reverse=True):
                    bw  = int(v*100)
                    clr = COLORS["danger"] if v > 0.6 else COLORS["warn"] if v > 0.3 else COLORS["success"]
                    st.markdown(
                        f"<div style='font-size:.8rem;color:{COLORS['text_mid']};margin:2px 0'>{k}"
                        f"<span style='float:right;color:{clr};font-family:JetBrains Mono'>{v*100:.0f}%</span></div>"
                        f"<div style='background:{COLORS['panel2']};border-radius:4px;height:6px;margin-bottom:8px'>"
                        f"<div style='width:{bw}%;background:{clr};height:100%;border-radius:4px'></div></div>",
                        unsafe_allow_html=True,
                    )

    # ── TRENDS ───────────────────────────────────────────────────────────
    with tabs[6]:
        st.caption("AFib probability and key HRV metrics across recent recordings.")
        if not all_history:
            st.info("No recordings yet — trends will appear once there's history.")
        else:
            trend_source = list(reversed(all_history[:20]))  # oldest → newest
            trend_rows = []
            with st.spinner("Computing trend history…"):
                for rec in trend_source:
                    sig_t, meta_t = load_recording_from_server(int(rec["id"]))
                    if sig_t is None or len(sig_t) == 0:
                        continue
                    res_t = compute_quick_result(sig_t, int((meta_t or {}).get("fs") or FS), model_choice)
                    ts = (rec.get("started_at") or "")[:16].replace("T", " ")
                    trend_rows.append({
                        "Recording": f"#{rec['id']}", "Time": ts,
                        "AFib Prob (%)": res_t["prob"] * 100,
                        "Heart Rate (bpm)": res_t["hr"],
                        "RMSSD (ms)": res_t["rmssd"],
                        "SDNN (ms)": res_t["sdnn"],
                    })
            if not trend_rows:
                st.info("Couldn't compute trends for the available recordings.")
            else:
                df_trend = pd.DataFrame(trend_rows)
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(
                    x=df_trend["Recording"], y=df_trend["AFib Prob (%)"], mode="lines+markers",
                    name="AFib Probability (%)", line=dict(color=COLORS["danger"], width=2),
                ))
                fig_t.update_layout(
                    **_base_layout(height=280),
                    title=dict(text="AFib Probability Over Time", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                    xaxis=dict(color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)"),
                    yaxis=dict(title="AFib Probability (%)", color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)", range=[0, 100]),
                )
                st.plotly_chart(fig_t, use_container_width=True)

                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=df_trend["Recording"], y=df_trend["Heart Rate (bpm)"],
                                            mode="lines+markers", name="Heart Rate",
                                            line=dict(color=COLORS["accent2"], width=2)))
                fig_h.update_layout(
                    **_base_layout(height=240),
                    title=dict(text="Heart Rate Over Time", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                    xaxis=dict(color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)"),
                    yaxis=dict(title="bpm", color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)"),
                )
                st.plotly_chart(fig_h, use_container_width=True)

                rr1, rr2 = st.columns(2)
                with rr1:
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Scatter(x=df_trend["Recording"], y=df_trend["RMSSD (ms)"],
                                                mode="lines+markers", line=dict(color=COLORS["accent"], width=2)))
                    fig_r.update_layout(
                        **_base_layout(height=220),
                        title=dict(text="RMSSD", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                        xaxis=dict(color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)"),
                        yaxis=dict(color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)"),
                    )
                    st.plotly_chart(fig_r, use_container_width=True)
                with rr2:
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(x=df_trend["Recording"], y=df_trend["SDNN (ms)"],
                                                mode="lines+markers", line=dict(color=COLORS["warn"], width=2)))
                    fig_s.update_layout(
                        **_base_layout(height=220),
                        title=dict(text="SDNN", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
                        xaxis=dict(color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)"),
                        yaxis=dict(color=COLORS["text_mid"], gridcolor="rgba(91,117,104,0.25)"),
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

                with st.expander("View trend data table"):
                    st.dataframe(df_trend, use_container_width=True, hide_index=True)

    # ── REPORTS ──────────────────────────────────────────────────────────
    with tabs[7]:
        if signal is None or len(signal) == 0:
            st.info("Load a recording to generate a report.")
        else:
            st.markdown(f'<div class="cs-label">Patient Report</div>', unsafe_allow_html=True)
            report_text = f"""PATIENT REPORT
================

Device / Patient ID: {DEFAULT_DEVICE_ID}
Recording: {signal_label}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Duration: {view_s:.1f}s
Heart Rate: {feat['mean_hr']:.1f} bpm
Signal Quality: {quality}

AI Result: {'Possible AFib' if is_afib else 'Normal Sinus Rhythm'}
AFib Probability: {prob*100:.1f}%
Model: {model_choice}
Decision Threshold: {threshold*100:.0f}%

Contributing Features:
""" + "\n".join(f"  - {k}: {v*100:.0f}%" for k, v in (reasons or {}).items()) + """

--------------------------------------------------
This is a research/screening prototype, not a
certified medical diagnosis. Consult a physician
for clinical decisions.
"""
            st.code(report_text, language=None)
            rc1, rc2 = st.columns(2)
            with rc1:
                st.download_button(
                    "⬇  Export Report (TXT)",
                    data=report_text.encode(),
                    file_name=f"afibai_report_{DEFAULT_DEVICE_ID}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with rc2:
                st.download_button(
                    "⬇  Download HRV Features (CSV)",
                    data=df_feat.to_csv(index=False).encode(),
                    file_name="hrv_features.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="reports_tab_download",
                )


if __name__ == "__main__":
    # Auth gate, in order:
    #   1. License agreement not yet acknowledged → agreement screen
    #   2. Not logged in                                  → login screen
    #   3. Logged in, patient, no intake yet               → medical-info screen
    #      (clinicians skip this — see login_screen)
    #   4. Fully authenticated                            → main app.
    if not st.session_state.get("agreement_accepted", False):
        license_agreement_screen()
        st.stop()
    if not st.session_state.get("authenticated", False):
        if st.session_state.get("auth_stage") == "intake":
            medical_intake_screen()
        else:
            login_screen()
        st.stop()
    # Role-based dispatch. Patients get a phone-friendly UI; clinicians get
    # the full analysis dashboard. The role is set during login (clinicians
    # go straight to authenticated=True, patients pass through intake first).
    role = st.session_state.get("auth_role") or auth.ROLE_CLINICIAN
    if role == auth.ROLE_PATIENT:
        patient_view()
    else:
        main()
