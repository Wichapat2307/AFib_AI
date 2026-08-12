# 🫀 AFib Detection Web App

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Data-PhysioNet-orange)

A **Streamlit web application** for detecting Atrial Fibrillation (AFib) from ECG signals. Upload an ECG file, visualize the signal, extract HRV features, and get an AFib prediction — all in the browser.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🖥️ What the App Does

Pick an input source from the sidebar — **Demo ECG**, **Upload .npy/.csv**, or **ESP WiFi 📡** — and the app will:

1. **Preprocess** the signal — bandpass filter (0.5–40 Hz) + z-score normalization
2. **Detect R-peaks** — adaptive threshold with prominence filtering
3. **Extract 24 HRV features** — time-domain, frequency-domain, and Poincaré metrics
4. **Predict AFib** — HRV-based heuristic (always available) or your trained ML model (RF / XGBoost / CatBoost / Ensemble)
5. **Visualize** results across five interactive tabs
6. *(transition demos only)* **Sliding window** — score the signal in 10 s windows with 1 s overlap and watch P(AFib) evolve over time

---

## 📊 Visualizations

| Tab              | Content                                                      |
| ---------------- | ------------------------------------------------------------ |
| 📈 ECG Signal    | Preprocessed waveform with R-peak markers                    |
| 💓 RR Tachogram  | Beat-to-beat interval over time                              |
| 🌀 Poincaré Plot | RRₙ vs RRₙ₊₁ scatter — classic AFib irregularity view        |
| 📊 HRV Features  | Radar chart + full feature table with units and descriptions |
| ⏱ Sliding Window | Per-window prediction table + CSV download (transition demos) |

**Sliding Window panel** (transition demos only): a two-row figure with the
windowed ECG stacked on top and the per-window AFib probability trace below,
with a dashed line at the Normal→AFib boundary and a 30% decision threshold.
For a 60 s sample this produces **51 overlapping windows** (10 s wide, 1 s hop).

---

## 🧠 HRV Features Extracted (24 total)

**Time domain:** mean/median RR, SDNN, RMSSD, pNN50, CV, HR stats, successive differences

**Poincaré:** SD1, SD2, SD ratio

**Frequency domain:** LF/HF ratio, LF norm, HF norm, dominant frequency

**Other:** skewness, kurtosis, IQR, irregularity score, beat count

---

## 🤖 Models

| Model           | Type                                       |
| --------------- | ------------------------------------------ |
| HRV Heuristic   | Rule-based (no weights needed)             |
| Random Forest   | Gradient boosting — HRV features           |
| XGBoost         | Gradient boosting — HRV features           |
| CatBoost        | Gradient boosting — HRV features           |
| Ensemble ⭐      | Mean of all loaded model probabilities     |

The Streamlit app uses the trained ML models (RF / XGBoost / CatBoost) and falls
back to the HRV heuristic when no weights are found. The deep models defined in
`model.py` (CNN, LSTM, RNN, CNN+LSTM) are part of the training/evaluation
pipeline but are **not** loaded by the web app.

---

| Source        | Details                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| **Demo ECG**  | 3 Normal + 3 Normal→AFib transition segments (60 s @ 128 Hz each)       |
| `.npy` upload | 1D array `(3840,)` or 2D `(N_windows, 3840)` — select window in sidebar |
| `.csv` upload | One signal per row, or a single 1D signal                               |
| **ESP WiFi 📡** | Live fetch from an ESP32/ESP8266 endpoint (raw float32, JSON, or CSV) |

> Signals should be sampled at **128 Hz** (30-second windows = 3840 samples). Other sampling rates can be set in the sidebar.

### 🛰️ ESP WiFi input

The app can pull a single ECG window from an ESP-class board over your local
network. In the sidebar, pick **ESP WiFi 📡**, enter the URL, and click anywhere
on the page to trigger a fetch. The endpoint should return one of:

- **Raw little-endian float32** binary (e.g. 3840 samples × 4 bytes = 15 360 B)
- **JSON** — a bare list, or `{"samples": [...]}`
- **CSV / text** — comma- or whitespace-separated floats on one line

Default URL: `http://192.168.4.1/ecg` (typical ESP soft-AP).

Enable **Auto-refresh** in the sidebar to re-fetch the endpoint on a timer
(requires `streamlit-autorefresh`; falls back to manual reloads if missing).

---

## 📦 Dependencies

streamlit>=1.35
numpy>=1.24
scipy>=1.11
pandas>=2.0
plotly>=5.18
scikit-learn>=1.3
xgboost>=2.0
catboost>=1.2
joblib>=1.3
requests>=2.31
streamlit-autorefresh>=1.0.3   # optional — ESP live-refresh only

The full list is in `requirements.txt`. `requests` and `streamlit-autorefresh`
are only needed for the ESP WiFi input source — everything else works without
them.

---

## ⚠️ Disclaimer

This tool is a **research prototype** and is not intended for clinical use. Do not use for medical diagnosis.

---

## 📡 Data Sources

- [MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/)
- [Long-Term Atrial Fibrillation Database](https://physionet.org/content/ltafdb/1.0.0/)
