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

Upload a raw ECG segment (`.npy` or `.csv`) or use a built-in synthetic demo, and the app will:

1. **Preprocess** the signal — bandpass filter (0.5–40 Hz) + z-score normalization
2. **Detect R-peaks** — adaptive threshold with prominence filtering
3. **Extract 24 HRV features** — time-domain, frequency-domain, and Poincaré metrics
4. **Predict AFib** — HRV-based heuristic (always available) or your trained deep model (optional)
5. **Visualize** results across four interactive tabs

---

## 📊 Visualizations

| Tab              | Content                                                      |
| ---------------- | ------------------------------------------------------------ |
| 📈 ECG Signal    | Preprocessed waveform with R-peak markers                    |
| 💓 RR Tachogram  | Beat-to-beat interval over time                              |
| 🌀 Poincaré Plot | RRₙ vs RRₙ₊₁ scatter — classic AFib irregularity view        |
| 📊 HRV Features  | Radar chart + full feature table with units and descriptions |

---

## 🧠 HRV Features Extracted (24 total)

**Time domain:** mean/median RR, SDNN, RMSSD, pNN50, CV, HR stats, successive differences

**Poincaré:** SD1, SD2, SD ratio

**Frequency domain:** LF/HF ratio, LF norm, HF norm, dominant frequency

**Other:** skewness, kurtosis, IQR, irregularity score, beat count

---

## 🤖 Models

| Model         | Type                             |
| ------------- | -------------------------------- |
| HRV Heuristic | Rule-based (no weights needed)   |
| CNN           | Deep learning — raw signal       |
| RNN           | Deep learning — raw signal       |
| CNN + LSTM    | Deep learning — raw signal       |
| XGBoost       | Gradient boosting — HRV features |
| CatBoost      | Gradient boosting — HRV features |

To use a deep model, check **"Use deep model"** in the sidebar and point it to your weights file (e.g. `models/best_model.pth`).

---

## 📂 Input Formats

| Format | Details                                                                 |
| ------ | ----------------------------------------------------------------------- |
| `.npy` | 1D array `(3840,)` or 2D `(N_windows, 3840)` — select window in sidebar |
| `.csv` | One signal per row, or a single 1D signal                               |
| Demo   | Built-in synthetic Normal or AFib ECG — no upload needed                |

> Signals should be sampled at **128 Hz** (30-second windows = 3840 samples). Other sampling rates can be set in the sidebar.

---

## 📦 Dependencies

streamlit>=1.35
numpy>=1.24
scipy>=1.11
pandas>=2.0
plotly>=5.18

---

## ⚠️ Disclaimer

This tool is a **research prototype** and is not intended for clinical use. Do not use for medical diagnosis.

---

## 📡 Data Sources

- [MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/)
- [Long-Term Atrial Fibrillation Database](https://physionet.org/content/ltafdb/1.0.0/)
