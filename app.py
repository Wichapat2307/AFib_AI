"""
app.py — AFib Detection Web App
================================
Streamlit app for detecting Atrial Fibrillation from ECG signals.
Restyled to match CardioSense aesthetic.

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
from pathlib import Path
import io, pickle, time

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
    page_title="CardioSense",
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
  .stApp {{ background: {COLORS["bg"]}; font-family: 'Inter', sans-serif; color: {COLORS["text"]}; }}
  .main .block-container {{ padding: 1.5rem 2rem !important; max-width: 100% !important; }}

  [data-testid="stSidebar"] {{ background: {COLORS["panel"]} !important; border-right: 1px solid {COLORS["border"]} !important; }}
  [data-testid="stSidebar"] * {{ color: {COLORS["text"]} !important; }}
  [data-testid="stSidebar"] hr {{ border-color: {COLORS["border"]} !important; }}
  [data-testid="stSidebar"] .stRadio label span {{ font-size: 0.83rem !important; color: {COLORS["text_mid"]} !important; }}
  [data-testid="stSidebar"] .stSelectbox label {{ font-size: 0.75rem !important; color: {COLORS["text_dim"]} !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] {{ background: {COLORS["panel2"]} !important; border-color: {COLORS["border"]} !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: {COLORS["panel2"]} !important; border-color: {COLORS["border"]} !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] * {{ background: {COLORS["panel2"]} !important; color: {COLORS["text"]} !important; -webkit-text-fill-color: {COLORS["text"]} !important; fill: {COLORS["text"]} !important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: {COLORS["text_mid"]} !important; }}
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

  .stTabs [data-baseweb="tab-list"] {{ background: {COLORS["panel"]}; border-bottom: 1px solid {COLORS["border"]}; padding: 0 1.5rem; gap: 0; }}
  .stTabs [data-baseweb="tab"] {{ color: {COLORS["text_mid"]} !important; font-family: 'Inter', sans-serif !important; font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.07em !important; text-transform: uppercase !important; padding: 0.9rem 1.4rem !important; border-bottom: 2px solid transparent !important; margin-bottom: -1px !important; background: transparent !important; }}
  .stTabs [data-baseweb="tab"] * {{ color: {COLORS["text_mid"]} !important; }}
  .stTabs [aria-selected="true"] {{ color: {COLORS["accent"]} !important; border-bottom: 2px solid {COLORS["accent"]} !important; }}
  .stTabs [aria-selected="true"] * {{ color: {COLORS["accent"]} !important; }}
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
  pre  {{ background: {COLORS["panel2"]} !important; border: 1px solid {COLORS["border"]} !important; border-radius: 8px !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
FS     = 128
WINDOW = 3840   # 30 s × 128 Hz

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
    # filtfilt needs signal longer than 3x filter order
    min_len = 3 * max(len(a), len(b))
    if len(signal) <= min_len:
        return signal  # return as-is if too short
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
    """Return (clf, imputer, threshold) whether model is a bare classifier
    or a dict bundle like {"model":..., "imputer":..., "threshold":...}."""
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

# ═══════════════════════════════════════════════════════════════════════════
# PLOTS  — all use CardioSense palette
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

def plot_ecg(signal, peaks, fs=FS, title="ECG Signal", is_afib=False):
    max_pts = 1500
    step    = max(1, len(signal) // max_pts)
    disp    = signal[::step]
    t       = np.arange(len(disp)) * step / fs
    tc      = COLORS["ecg_afib"] if is_afib else COLORS["ecg_normal"]

    fig = go.Figure()
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
        **_base_layout(height=300, plot_bgcolor=COLORS["ecg_bg"]),
        title=dict(text=title, font=dict(family="Inter", size=12, color=COLORS["text_mid"]), x=0.01),
        xaxis=dict(
            title="Time (s)", color=COLORS["text_mid"],
            gridcolor=COLORS["ecg_grid_maj"], gridwidth=1, dtick=1, showgrid=True,
            minor=dict(dtick=0.08, gridcolor=COLORS["ecg_grid_min"], showgrid=True),
            tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"]),
        ),
        yaxis=dict(
            title="Amplitude (norm.)", color=COLORS["text_mid"],
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
    fig.add_hrect(y0=600, y1=1000, fillcolor="rgba(31,204,122,0.05)", line_width=0,
                  annotation_text="Normal range",
                  annotation_position="top left",
                  annotation=dict(font_color=COLORS["success"], font_size=11))
    fig.update_layout(
        **_base_layout(height=260),
        title=dict(text="RR Interval Series", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        xaxis=dict(title="Beat #", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                   tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="RR (ms)", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                   tickfont=dict(family="JetBrains Mono", size=10)),
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
        line=dict(color=COLORS["border_light"], dash="dash", width=1), showlegend=False,
    ))
    fig.update_layout(
        **_base_layout(height=260),
        title=dict(text="Poincaré Plot", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        xaxis=dict(title="RRₙ (ms)", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                   range=lim, tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="RRₙ₊₁ (ms)", color=COLORS["text_mid"], gridcolor=COLORS["border"],
                   range=lim, tickfont=dict(family="JetBrains Mono", size=10)),
    )
    return fig

def plot_gauge(prob):
    color = COLORS["success"] if prob < 0.35 else COLORS["warn"] if prob < 0.65 else COLORS["danger"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(color=color, size=34, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor=COLORS["text_dim"],
                      tickfont=dict(color=COLORS["text_dim"], family="JetBrains Mono", size=9)),
            bar=dict(color=color, thickness=0.28),
            bgcolor=COLORS["panel2"],
            borderwidth=1, bordercolor=COLORS["border"],
            steps=[
                dict(range=[0,   35], color="rgba(31,204,122,0.07)"),
                dict(range=[35,  65], color="rgba(244,161,36,0.07)"),
                dict(range=[65, 100], color="rgba(240,64,96,0.07)"),
            ],
            threshold=dict(line=dict(color=COLORS["danger"], width=2), value=65),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["panel"], height=190,
        margin=dict(l=15, r=15, t=15, b=10),
        font=dict(color=COLORS["text"]),
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
                            gridcolor=COLORS["border"],
                            tickfont=dict(color=COLORS["text_mid"])),
            angularaxis=dict(color=COLORS["text_mid"], gridcolor=COLORS["border"]),
        ),
        paper_bgcolor=COLORS["panel"],
        title=dict(text="HRV Radar", font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
        margin=dict(l=40,r=40,t=50,b=40), showlegend=False,
    )
    return fig

def plot_feature_importance_xgb(model):
    try:
        clf, _, _ = _unpack_model(model)
        scores = clf.get_booster().get_fscore()
        if not scores:
            scores = dict(zip([f"f{i}" for i in range(len(FEATURE_NAMES))],
                              clf.feature_importances_))
        named = {}
        for k, v in scores.items():
            try:
                idx = int(k.replace("f",""))
                named[FEATURE_NAMES[idx]] = v
            except Exception:
                named[k] = v
        df = pd.DataFrame({"Feature":list(named.keys()),"Importance":list(named.values())})
        df = df.sort_values("Importance", ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=df["Importance"], y=df["Feature"], orientation="h",
            marker_color=COLORS["accent"],
            text=[f"{v:.0f}" for v in df["Importance"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"]),
        ))
        fig.update_layout(
            **_base_layout(height=380),
            title=dict(text="XGBoost Feature Importance",
                       font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
            xaxis=dict(color=COLORS["text_mid"], gridcolor=COLORS["border"],
                       tickfont=dict(family="JetBrains Mono", size=10)),
            yaxis=dict(color=COLORS["text"], tickfont=dict(family="Inter", size=11)),
            margin=dict(l=130, r=60, t=45, b=40),
        )
        return fig
    except Exception:
        return None

def plot_feature_importance_cb(model):
    try:
        clf, _, _ = _unpack_model(model)
        imps = clf.get_feature_importance()
        df = pd.DataFrame({"Feature": FEATURE_NAMES[:len(imps)], "Importance": imps})
        df = df.sort_values("Importance", ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=df["Importance"], y=df["Feature"], orientation="h",
            marker_color=COLORS["warn"],
            text=[f"{v:.1f}" for v in df["Importance"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_mid"]),
        ))
        fig.update_layout(
            **_base_layout(height=380),
            title=dict(text="CatBoost Feature Importance",
                       font=dict(family="Inter", size=12, color=COLORS["text_mid"])),
            xaxis=dict(color=COLORS["text_mid"], gridcolor=COLORS["border"],
                       tickfont=dict(family="JetBrains Mono", size=10)),
            yaxis=dict(color=COLORS["text"], tickfont=dict(family="Inter", size=11)),
            margin=dict(l=130, r=60, t=45, b=40),
        )
        return fig
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── SIDEBAR ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:1rem 0 0.8rem;'>
          <div style='font-size:1.8rem; margin-bottom:6px;'>🫀</div>
          <div style='font-family:"Sora",sans-serif; font-size:1.3rem; color:{COLORS["text"]};
                      font-weight:700; line-height:1;'>CardioSense</div>
          <div style='font-family:"JetBrains Mono",monospace; font-size:0.55rem;
                      color:{COLORS["text_dim"]}; letter-spacing:0.12em; margin-top:4px;'>
            HRV ANALYSIS v1.0
          </div>
          <div style='font-size:0.7rem; color:{COLORS["text_mid"]}; margin-top:6px; line-height:1.5;'>
            ECG Signal Analysis<br>AFib Detection Engine
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown(f'<div class="cs-label">Input Source</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Input Source",
            ["Demo ECG", "Upload .npy file", "Upload .csv file"],
            label_visibility="collapsed",
        )

        if input_mode == "Demo ECG":
            demo_choice = st.selectbox(
                "Demo ECG",
                list(DEMO_FILES.keys())
            )

        st.divider()
        st.markdown(f'<div class="cs-label">Signal Settings</div>', unsafe_allow_html=True)
        fs_input      = st.number_input("Sampling rate (Hz)", min_value=64, max_value=2000,
                                        value=128, step=1)
        window_index  = st.number_input("Window index (multi-window files)",
                                        min_value=0, value=0, step=1)

        # ── MODEL SELECTION ─────────────────────────────────────────────
        st.divider()
        st.markdown(f'<div class="cs-label">Model Selection</div>', unsafe_allow_html=True)

        MODEL_OPTIONS = ["Random Forest", "XGBoost", "CatBoost", "Ensemble ⭐"]
        model_choice_raw = st.radio(
            "Model",
            MODEL_OPTIONS,
            index=3,  # Ensemble is the default / recommended option
            label_visibility="collapsed",
        )
        model_choice = model_choice_raw.replace(" ⭐", "")

        show_individual = True
        if model_choice == "Ensemble":
            show_individual = st.checkbox("Show individual model predictions", value=True)

        st.divider()
        with st.expander("🛠️ Model path debug", expanded=False):
            st.caption(f"app.py location: `{APP_DIR}`")
            for name, path in MODEL_PATHS.items():
                exists = Path(path).exists()
                dot = "🟢" if exists else "🔴"
                st.markdown(f"{dot} **{name}**")
                st.code(path, language=None)

        st.divider()
        # Model availability badges
        st.markdown(f'<div class="cs-label">Library Status</div>', unsafe_allow_html=True)
        for name, ok in [
            ("Random Forest", SKLEARN_AVAILABLE),
            ("XGBoost",       XGB_AVAILABLE),
            ("CatBoost",      CATBOOST_AVAILABLE),
            ]:
            dot = "🟢" if ok else "🔴"
            st.markdown(f'<div class="cs-badge">{dot} {name}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown(f"""
        <div style='font-size:0.6rem; color:{COLORS["text_dim"]}; line-height:1.8;'>
          ⚠️ Research tool only.<br>Not a certified medical device.<br>Consult a physician for diagnosis.
        </div>""", unsafe_allow_html=True)

    # ── TOP BAR ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{COLORS["panel"]}; border-bottom:1px solid {COLORS["border"]};
                padding:0.85rem 2rem; display:flex; align-items:center;
                justify-content:space-between; margin:-1.5rem -2rem 1.5rem;'>
      <div style='display:flex; align-items:center; gap:12px;'>
        <span style='font-size:1.6rem;'>🫀</span>
        <div>
          <span style='font-family:"Sora",sans-serif; font-size:1.25rem; color:{COLORS["text"]}; font-weight:700;'>
            CardioSense
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIGNAL LOADING ───────────────────────────────────────────────────
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

    # ── SIGNAL INFO DISPLAY ──────────────────────────────────────────────
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

    # ── PREPROCESS + FEATURES ────────────────────────────────────────────
    if signal is not None and len(signal) > 0:
        with st.spinner("Processing signal…"):
            proc     = preprocess(signal, fs=fs_input)
            peaks    = detect_rpeaks(signal, fs=fs_input)
            rr_ms    = np.diff(peaks) / fs_input * 1000
            rr_ms    = rr_ms[(rr_ms > 250) & (rr_ms < 2000)]
            features = extract_hrv(signal, fs=fs_input)
            feat     = dict(zip(FEATURE_NAMES, features))

        # ── RUN MODEL ────────────────────────────────────────────────────
        label = prob = threshold = method_note = reasons = None
        imp_fig = None
        individual_preds = {}   # name -> probability, populated in Ensemble mode

        if model_choice == "Random Forest":
            mdl = load_rf_model(MODEL_PATHS["Random Forest"])
            if mdl is None:
                st.warning(
                    f"Random Forest model not found at `{MODEL_PATHS['Random Forest']}`. "
                    "Falling back to HRV heuristic."
                )
                label, prob, threshold, reasons = hrv_heuristic(features)
                method_note = "HRV heuristic (Random Forest weights missing)"
            else:
                label, prob, threshold = predict_rf(mdl, features)
                method_note = f"Random Forest — {MODEL_PATHS['Random Forest']}"

        elif model_choice == "XGBoost":
            mdl = load_xgb_model(MODEL_PATHS["XGBoost"])
            if mdl is None:
                st.warning(
                    f"XGBoost model not found at `{MODEL_PATHS['XGBoost']}`. "
                    "Falling back to HRV heuristic."
                )
                label, prob, threshold, reasons = hrv_heuristic(features)
                method_note = "HRV heuristic (XGBoost weights missing)"
            else:
                label, prob, threshold = predict_xgb(mdl, features)
                method_note = f"XGBoost — {MODEL_PATHS['XGBoost']}"
                imp_fig = plot_feature_importance_xgb(mdl)

        elif model_choice == "CatBoost":
            mdl = load_catboost_model(MODEL_PATHS["CatBoost"])
            if mdl is None:
                st.warning(
                    f"CatBoost model not found at `{MODEL_PATHS['CatBoost']}`. "
                    "Falling back to HRV heuristic."
                )
                label, prob, threshold, reasons = hrv_heuristic(features)
                method_note = "HRV heuristic (CatBoost weights missing)"
            else:
                label, prob, threshold = predict_catboost(mdl, features)
                method_note = f"CatBoost — {MODEL_PATHS['CatBoost']}"
                imp_fig = plot_feature_importance_cb(mdl)

        elif model_choice == "Ensemble":
            rf_model  = load_rf_model(MODEL_PATHS["Random Forest"])
            xgb_model = load_xgb_model(MODEL_PATHS["XGBoost"])
            cat_model = load_catboost_model(MODEL_PATHS["CatBoost"])

            probs = []
            if rf_model is not None:
                _, p, _ = predict_rf(rf_model, features)
                probs.append(p)
                individual_preds["Random Forest"] = p
            if xgb_model is not None:
                _, p, _ = predict_xgb(xgb_model, features)
                probs.append(p)
                individual_preds["XGBoost"] = p
            if cat_model is not None:
                _, p, _ = predict_catboost(cat_model, features)
                probs.append(p)
                individual_preds["CatBoost"] = p

            if len(probs) == 0:
                st.warning(
                    "None of the ensemble model weights were found "
                    f"(`{MODEL_PATHS['Random Forest']}`, `{MODEL_PATHS['XGBoost']}`, "
                    f"`{MODEL_PATHS['CatBoost']}`). Falling back to HRV heuristic."
                )
                label, prob, threshold, reasons = hrv_heuristic(features)
                method_note = "HRV heuristic (no ensemble models found)"
            else:
                # final_prob = mean of the probabilities from every model that loaded
                prob = float(np.mean(probs))
                threshold = 0.3
                label = "AFib" if prob >= threshold else "Normal"
                method_note = f"Ensemble — mean of {len(probs)}/3 models"

        is_afib = label == "AFib"

    else:
        st.markdown(
            f"""
            <div style='padding:80px 20px; text-align:center;'>
            <div style='font-size:3rem; margin-bottom:12px;'>🫀</div>
            <div style='font-size:0.9rem; color:{COLORS["text_dim"]};'>
                Input file to begin
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # ── ALERT BANNER ─────────────────────────────────────────────────────
    if is_afib:
        st.markdown(f"""
        <div class='cs-alert cs-alert-afib'>
          <span style='font-size:1.5rem; flex-shrink:0;'>⚠️</span>
          <div>
            <div style='font-weight:700; font-size:0.95rem; color:{COLORS["danger"]}; font-family:"Sora",sans-serif;'>
              Atrial Fibrillation Detected
            </div>
            <div style='font-size:0.78rem; color:{COLORS["text_mid"]}; margin-top:3px;'>
              AFib probability: <strong>{prob*100:.1f}%</strong> — Consult a physician immediately.
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
              AFib probability: <strong>{prob*100:.1f}%</strong> — No atrial fibrillation detected.
              <br>
              Method: <strong>{method_note}</strong> &nbsp;|&nbsp; Decision threshold: <strong>{threshold*100:.0f}%</strong>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── INDIVIDUAL MODEL PREDICTIONS (Ensemble mode) ────────────────────
    if model_choice == "Ensemble" and show_individual and individual_preds:
        rows_html = ""
        for name, p in individual_preds.items():
            rows_html += (
                f"<div class='cs-pred-row'><span>{name}</span>"
                f"<span style='color:{COLORS['text']}'>{p*100:.1f}%</span></div>"
            )
        rows_html += (
            f"<div style='border-top:1px solid {COLORS['border']}; margin:6px 0;'></div>"
            f"<div class='cs-pred-row' style='color:{COLORS['accent']}; font-weight:700;'>"
            f"<span>Ensemble</span><span>{prob*100:.1f}%</span></div>"
        )
        st.markdown(
            f"<div class='cs-card'><div class='cs-label'>Individual Model Predictions</div>{rows_html}</div>",
            unsafe_allow_html=True,
        )

    # ── METRICS ROW ──────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AFib Probability", f"{prob*100:.1f}%",
              delta="HIGH ⚠" if is_afib else "Normal",
              delta_color="inverse" if is_afib else "normal")
    m2.metric("Mean Heart Rate",  f"{feat['mean_hr']:.1f} bpm")
    m3.metric("RMSSD",            f"{feat['rmssd']:.1f} ms",
              help="Root Mean Square Successive Differences — elevated in AFib")
    m4.metric("SDNN",             f"{feat['sdnn']:.1f} ms")
    m5.metric("R-Peaks Detected", str(len(peaks)))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── ECG + GAUGE ──────────────────────────────────────────────────────
    ecg_col, gauge_col = st.columns([3, 1])
    view_s = min(15, len(signal)/fs_input)
    n_view = int(view_s * fs_input)
    with ecg_col:
        st.plotly_chart(
            plot_ecg(proc[:n_view], peaks[peaks < n_view], fs=fs_input,
                     title=f"ECG  ·  First {view_s:.0f}s  ·  {signal_label}",
                     is_afib=is_afib),
            use_container_width=True,
        )
    with gauge_col:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.plotly_chart(plot_gauge(prob), use_container_width=True)
        lbl_c = COLORS["danger"] if is_afib else COLORS["success"]
        st.markdown(
            f"<div style='text-align:center; font-family:Inter; font-size:0.72rem;"
            f" color:{lbl_c}; font-weight:700; margin-top:-10px;'>{label.upper()}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────────────────
    tabs = st.tabs(["💓  RR Tachogram", "🌀  Poincaré", "📊  HRV Features", "🌲  Feature Importance"])

    with tabs[0]:
        if len(rr_ms) >= 3:
            st.plotly_chart(plot_rr(rr_ms), use_container_width=True)
        else:
            st.warning("Not enough RR intervals detected.")

    with tabs[1]:
        if len(rr_ms) >= 4:
            st.plotly_chart(plot_poincare(rr_ms, is_afib=is_afib), use_container_width=True)
        else:
            st.warning("Not enough RR intervals for Poincaré plot.")

    with tabs[2]:
        left, right = st.columns([1,1])
        with left:
            st.markdown(f'<div class="cs-label">HRV Feature Values</div>', unsafe_allow_html=True)
            rows = [{"Feature":n, "Value":f"{v:.4f}",
                     "Unit":FEATURE_UNITS.get(n,""),
                     "Description":FEATURE_DESCRIPTIONS.get(n,"")}
                    for n, v in zip(FEATURE_NAMES, features)]
            df_feat = pd.DataFrame(rows)
            st.dataframe(df_feat, use_container_width=True, height=420, hide_index=True)
        with right:
            st.plotly_chart(plot_radar(features), use_container_width=True)

        # Score breakdown (heuristic only)
        if reasons:
            with st.expander("🔍  Score breakdown (HRV heuristic)", expanded=False):
                st.markdown(f"""
                <div style='font-size:0.8rem; color:{COLORS["text_mid"]}; margin-bottom:0.8rem; line-height:1.6;'>
                  Each bar shows how much a feature pushed toward AFib.
                  Higher scores contribute more to the overall AFib probability.
                </div>""", unsafe_allow_html=True)
                for k, v in reasons.items():
                    bw  = int(v*100)
                    clr = COLORS["danger"] if v > 0.6 else COLORS["warn"] if v > 0.3 else COLORS["success"]
                    st.markdown(
                        f"<div style='font-size:.8rem;color:{COLORS['text_mid']};margin:2px 0'>{k}"
                        f"<span style='float:right;color:{clr};font-family:JetBrains Mono'>{v*100:.0f}%</span></div>"
                        f"<div style='background:{COLORS['panel2']};border-radius:4px;height:6px;margin-bottom:8px'>"
                        f"<div style='width:{bw}%;background:{clr};height:100%;border-radius:4px'></div></div>",
                        unsafe_allow_html=True,
                    )

    with tabs[3]:
        if imp_fig:
            st.plotly_chart(imp_fig, use_container_width=True)
            st.caption("Importance is derived from the loaded model weights. "
                       "Higher = more influential in the AFib/Normal decision.")
        else:
            st.markdown(f"""
            <div class='cs-card'>
              <div style='font-size:0.85rem; color:{COLORS["text_mid"]}; line-height:1.6;'>
                Feature importance is available when an <strong style='color:{COLORS["text"]};'>XGBoost</strong>
                or <strong style='color:{COLORS["text"]};'>CatBoost</strong> model is loaded from the sidebar.
                The HRV heuristic and Random Forest models do not produce per-feature importance scores here.
              </div>
            </div>""", unsafe_allow_html=True)

    # ── DOWNLOAD ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        "⬇  Download HRV Features (CSV)",
        data=df_feat.to_csv(index=False).encode(),
        file_name="hrv_features.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
