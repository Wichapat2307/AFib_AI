# AFibAI — Deployment & Usage

This document covers how to run the AFibAI app locally for development, and how
to deploy it for real use (patient phone + clinician laptop, talking through a
cloud server).

There are three components:

| Component       | What it does                                       | Where it runs          |
|-----------------|----------------------------------------------------|------------------------|
| `server.py`     | Polls Firebase for ECG chunks, runs the live model (via `predict.py`), stores recordings, writes results back to Firebase | Your laptop (dev) or a cloud VM (prod) |
| `app.py`        | The Streamlit web app — clinician or patient view  | Same machine as `server.py` for dev; can be deployed separately |
| `live_wio/wio_ecg_streamer.ino` | Arduino sketch for the Wio Terminal | On the Wio itself |

The Wio talks **only to Firebase Realtime Database** — it PUTs ECG chunks to
`/devices/{id}/live` and reads the prediction back from `/devices/{id}/result`.
`server.py` is the only thing that talks to both Firebase and Streamlit:

```
AD8232 → Wio → Firebase (raw ECG) → server.py (predict.py: XGBoost/CatBoost/HRV)
                                        → Firebase (result) → Wio LCD
                                        → Streamlit (app.py), unchanged
```

Two useful testing tools that don't need real hardware:
- `live_wio/mock_wio.py` — POSTs straight to `server.py` (the original, pre-Firebase path — still works, `server.py` accepts both).
- `live_wio/mock_wio_firebase.py` — PUTs straight to Firebase, exercising the full Firebase → server.py → prediction → Firebase loop.

---

## Local development (everything on one laptop)

### 1. Install dependencies

```bash
pip install streamlit numpy scipy plotly pandas scikit-learn xgboost catboost bcrypt flask firebase-admin
```

(Adjust to whatever subset of models you actually have under `models/`.)

### 1b. Firebase setup (one-time)

1. In the [Firebase console](https://console.firebase.google.com/), open your
   project → **Realtime Database** and note the database URL, e.g.
   `https://afibai-45c89-default-rtdb.asia-southeast1.firebasedatabase.app/`.
2. Set the security **rules** so the Wio (which has no credentials) can only
   read/write under `/devices`, and nothing else is publicly exposed:
   ```json
   {
     "rules": {
       "devices": {
         "$device_id": {
           ".read": true,
           ".write": true
         }
       },
       ".read": false,
       ".write": false
     }
   }
   ```
3. Generate a **service account key** (Project Settings → Service Accounts →
   Generate new private key) — this is for `server.py` only, never for the
   Wio. Keep the downloaded JSON file out of git (already covered by
   `.gitignore`'s `*firebase-adminsdk*.json` pattern).
4. Set these environment variables before starting `server.py`:
   - `FIREBASE_DB_URL` — the database URL from step 1.
   - `FIREBASE_CRED_PATH` — path to the service-account JSON file (local
     dev). On a host where you can't drop a file on disk (e.g. Render), set
     `FIREBASE_CRED_JSON` to the file's raw contents instead.

   If you skip this step, `server.py` still works — it just won't start the
   Firebase polling thread, and the Wio needs to POST straight to
   `/api/wio/upload` (the original pre-Firebase path) instead.

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

You don't need a real Wio to test the pipeline. In a third terminal, either:

```bash
# Original path: straight to server.py
python live_wio/mock_wio.py --bpm 75

# Firebase path: exercises the full Wio-would-do-this loop
python live_wio/mock_wio_firebase.py --db-url https://your-project-default-rtdb.your-region.firebasedatabase.app --bpm 75
```

Both push plausible ECG samples once a second. Switch back to the clinician
view, pick **Wio Live (Server)** from the Input Source radio, and the live
trace should start scrolling within ~2 seconds — Streamlit only ever talks
to `server.py`, so it behaves identically regardless of which mock you used.

Useful flags (both scripts):
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
    │  PUT/GET https://your-project-default-rtdb...firebasedatabase.app/devices/{id}/...
    ▼
Firebase Realtime Database
    ▲                        │
    │ poll                   │ write result
    │                        ▼
Cloud VM (Render / Railway / Fly.io)
    ├── server.py       ← Flask + Firebase poller + predict.py
    └── app.py          ← Streamlit (separate service, unchanged)
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

On the Streamlit service (`app.py`):

- `AFIBAI_SERVER_URL` — the public URL of `server.py`, e.g. `https://afibai-live.onrender.com`
- `AFIBAI_DEVICE_ID` — defaults to `wio_01`. Use a different one per physical Wio.

On the `server.py` service:

- `FIREBASE_DB_URL` — your Firebase Realtime Database URL.
- `FIREBASE_CRED_JSON` — the service-account key's raw JSON contents (use
  this instead of `FIREBASE_CRED_PATH` on hosts like Render, where you can
  set env vars but can't easily drop a credentials file on disk).
- `AFIBAI_FIREBASE_DEVICES` — comma-separated device IDs to poll (default `wio_01`).
- `AFIBAI_FIREBASE_POLL_INTERVAL` — seconds between Firebase reads (default `1.0`).
- `AFIBAI_PREDICT_INTERVAL` — seconds between live predictions (default `2.0`).
- `AFIBAI_API_KEY` — if set, every `server.py` request (except `/`) must send a
  matching `X-API-Key` header. Unset by default (local/dev). **Always set this
  once `server.py` is reachable from the public internet.**

In `live_wio/config.h`, set `FIREBASE_DB_URL` (not `SERVER_URL` — the Wio no
longer talks to `server.py` directly).

**Single worker only**: `server.py`'s ring buffer and the Firebase poller/
prediction threads are process-local in-memory state. If you deploy with
`gunicorn`, use `--workers 1` — this was already a constraint for the ring
buffer before Firebase entered the picture.

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
| Wio sketch won't compile              | Wrong board selected, or missing `TFT_eSPI` library | Select "Seeed SAMD (Wio Terminal)"; install `TFT_eSPI` |
| `Could not start recording`           | server.py down, or already recording             | Restart server.py; check `/api/wio/status` |
| `server.py` never sees Firebase data  | `FIREBASE_DB_URL`/credentials not set, or RTDB rules block reads | Check server.py startup log for "Firebase bridge disabled"; verify rules JSON |
| Wio's `PUT live failed` in Serial Monitor | Wrong `FIREBASE_DB_URL`, or rules don't allow write to `/devices/*` | Double-check `config.h` and the RTDB rules |
| Wio screen stays on `--` for every field | No `/devices/{id}/result` yet — server.py hasn't predicted anything | Confirm chunks are arriving (`/api/wio/status`) and wait ~`AFIBAI_PREDICT_INTERVAL`s |

---

## File map

```
AFib_AI/
├── app.py                  Streamlit app (clinician + patient views)
├── auth.py                 SQLite-backed user auth with roles
├── server.py               Flask live-data + recording server + Firebase bridge
├── predict.py               Streamlit-free ML/HRV pipeline, used by server.py
├── firebase_bridge.py        Firebase Admin SDK wrapper, used by server.py
├── *firebase-adminsdk*.json  Service-account key (git-ignored, never commit)
├── live_wio/
│   ├── wio_ecg_streamer.ino   Arduino sketch for the Wio (talks to Firebase)
│   ├── config.example.h        WiFi + Firebase URL template (copy to config.h)
│   ├── mock_wio.py             Fake Wio, POSTs to server.py
│   ├── mock_wio_firebase.py    Fake Wio, PUTs to Firebase
│   └── README.md               Flashing instructions
├── models/                 Trained RF/XGBoost/CatBoost weights
├── samples/                Demo .npy files for the Demo ECG mode
├── users.db                (auto-created) SQLite user database
├── afib_history.db         (auto-created) SQLite recording database
└── README_DEPLOY.md        This file
```