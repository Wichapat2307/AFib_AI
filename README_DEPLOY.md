# AFibAI — Deployment & Usage

This document covers how to run the AFibAI app locally for development, and how
to deploy it for real use (patient phone + clinician laptop, talking through a
cloud server).

There are three components:

| Component       | What it does                                       | Where it runs          |
|-----------------|----------------------------------------------------|------------------------|
| `server.py`     | Receives ECG chunks from the Wio, stores recordings| Your laptop (dev) or a cloud VM (prod) |
| `app.py`        | The Streamlit web app — clinician or patient view  | Same machine as `server.py` for dev; can be deployed separately |
| `live_wio/wio_ecg_streamer.ino` | Arduino sketch for the Wio Terminal | On the Wio itself |

Plus a useful testing tool: `live_wio/mock_wio.py` pretends to be a Wio so you
can test the full pipeline without owning the device yet.

---

## Local development (everything on one laptop)

### 1. Install dependencies

```bash
pip install streamlit numpy scipy plotly pandas scikit-learn xgboost catboost bcrypt flask
```

(Adjust to whatever subset of models you actually have under `models/`.)

### 2. Start the server

In one terminal:

```bash
python server.py
```

You should see:
```
[afibai-server] listening on http://0.0.0.0:5000
[afibai-server] db = C:\Users\Admin\Desktop\claude\AFib_detection\afib_history.db
[afibai-server] live buffer = 30.0s (3840 samples)
```

Sanity-check from another terminal:
```bash
curl http://127.0.0.1:5000/
```

### 3. Start the Streamlit app

In a second terminal:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### 4. Sign in

Two demo accounts are seeded automatically:

| Username  | Password    | Role      |
|-----------|-------------|-----------|
| `doctor`  | `doctor123` | clinician |
| `patient` | `patient123`| patient   |

(Plus the built-in admin: `admin` / `230709`.)

Sign in as `doctor` to see the full analysis dashboard; sign in as `patient`
(in a different browser, ideally on your phone) to see the simple button UI.

### 5. Stream fake ECG data

You don't need a real Wio to test the pipeline. In a third terminal:

```bash
python live_wio/mock_wio.py --bpm 75
```

This pushes plausible ECG samples to `server.py` every second. Switch back
to the clinician view, pick **Wio Live (Server)** from the Input Source
radio, and the live trace should start scrolling within ~2 seconds.

Useful flags:
- `--bpm 110` — faster heart rate
- `--afib` — irregular RR intervals (your classifier should flag it as AFib)
- `--duration 30` — stop after 30 seconds
- `--device wio_lab` — pretend to be a different device

### 6. Try the patient/clinician flow

1. Open `http://localhost:8501/` in your laptop's browser, sign in as `doctor`.
2. Open `http://<laptop-ip>:8501/` on your phone (replace `<laptop-ip>` with
   the result of `ipconfig` — your local WiFi IP, not `127.0.0.1`).
3. Sign in as `patient` on the phone.
4. Tap **Start Recording** on the phone. Watch the recording appear in the
   clinician's History list within seconds.
5. Tap **Alert the Doctor** on the phone. Refresh the clinician's browser —
   a toast notification appears.

---

## Network setup

For the phone to reach the laptop, both must be on the same WiFi network, and
your laptop firewall must allow inbound TCP on port 8501 (Streamlit) and 5000
(server.py). On Windows you may need to allow these through Windows Defender
Firewall the first time you run them.

---

## Deploying to the cloud

When you're ready to put this on the public internet (so the patient can be at
home and the clinician at the hospital), the architecture becomes:

```
Wio at patient's home
    │  WiFi → home router (NAT) → internet
    │  POST https://your-server.com/api/wio/upload
    ▼
Cloud VM (Render / Railway / Fly.io)
    ├── server.py       ← Flask
    └── app.py          ← Streamlit (separate service)
              ▲
              │  HTTPS, browser
              ▼
       ┌──────┴──────┐
       │             │
   Patient's     Clinician's
   phone         laptop
```

### Recommended free tier

| Service       | Use                | Free tier       |
|---------------|--------------------|-----------------|
| Render        | server.py          | 750 hrs/month   |
| Streamlit Community Cloud | app.py | Unlimited (with limits) |
| (Use SQLite on the Render VM for history — single-server fits your scale.) |

### Setting environment variables

When deploying, set on both services:

- `AFIBAI_SERVER_URL` — the public URL of `server.py`, e.g. `https://afibai-live.onrender.com`
- `AFIBAI_DEVICE_ID` — defaults to `wio_01`. Use a different one per physical Wio.

In `live_wio/config.h`, set `SERVER_URL` to the public server URL.

### HTTPS / certificates

The Wio uses `HTTPClient` which supports HTTPS via `WiFiClientSecure`. By
default it does **not** verify the server certificate, which is fine for a
demo but not for production. To enable verification, see the Wio sketch
comment header.

---

## Troubleshooting

| Symptom                              | Likely cause                                    | Fix                                  |
|--------------------------------------|--------------------------------------------------|--------------------------------------|
| `curl http://127.0.0.1:5000/` fails  | server.py not running                            | Start it in a terminal                |
| "Server unreachable" in app          | `AFIBAI_SERVER_URL` wrong, or server is down     | Check the env var; restart server     |
| Mock streams but live view is empty  | `input_mode` is not "Wio Live (Server)"          | Switch in the sidebar                 |
| Phone can't reach laptop              | Different WiFi networks, or firewall blocking    | Same SSID; allow ports 5000/8501      |
| Recordings don't persist across restarts | SQLite path wrong or DB deleted              | Check `afib_history.db` is being written |
| Wio sketch won't compile              | Wrong board selected                             | Select "Seeed SAMD (Wio Terminal)"    |
| `Could not start recording`           | server.py down, or already recording             | Restart server.py; check `/api/wio/status` |

---

## File map

```
AFib_detection/
├── app.py                  Streamlit app (clinician + patient views)
├── auth.py                 SQLite-backed user auth with roles
├── server.py               Flask live-data + recording server
├── live_wio/
│   ├── wio_ecg_streamer.ino   Arduino sketch for the Wio
│   ├── config.example.h        WiFi + server URL template (copy to config.h)
│   ├── mock_wio.py             Fake Wio for testing without hardware
│   └── README.md               Flashing instructions
├── models/                 Trained RF/XGBoost/CatBoost weights
├── samples/                Demo .npy files for the Demo ECG mode
├── users.db                (auto-created) SQLite user database
├── afib_history.db         (auto-created) SQLite recording database
└── README_DEPLOY.md        This file
```