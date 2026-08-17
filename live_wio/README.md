# live_wio — Wio Terminal ECG streamer

Arduino sketch for the Seeed Studio **Wio Terminal** that streams ECG samples
to Firebase Realtime Database, and reads the live AFib prediction back from
Firebase to show on the Wio's own screen. The Wio doesn't talk to `server.py`
directly at all — `server.py` (running elsewhere) polls Firebase for the ECG
chunks, runs the model, and writes the result back to Firebase.

```
AD8232 → Wio (this sketch) → Firebase → server.py (model) → Firebase → Wio LCD
```

## What you need

- 1× **Seeed Wio Terminal** (~$35)
- 1× **Grove – ECG** sensor (AD8232-based, ~$20) with the 3-lead snap cable
- USB-C cable (to flash)
- A computer with the **Arduino IDE** (or PlatformIO)

## Wiring

The Grove port on the **bottom** of the Wio Terminal is the analog Grove port.
Plug the ECG sensor's Grove connector straight in. No wiring, no soldering.

Snap the three electrode leads onto the patient:
- **RA** (red, right arm) → right side of the chest, below the clavicle
- **LA** (yellow, left arm) → left side of the chest, below the clavicle
- **RL** (green, right leg / reference) → right lower rib cage or hip

The Wio's ADC reads from analog pin `A4`, which is what the bottom Grove port
maps to on this board.

## Flashing

### 1. Install board support

In Arduino IDE:

1. **File → Preferences → Additional Board URLs**, add:
   ```
   https://raw.githubusercontent.com/Seeed-Studio/Seeed_Platform/master/package_seeeduino_boards_index.json
   ```
2. **Tools → Board → Boards Manager**, search "Wio Terminal", install the
   Seeed SAMD package.

### 2. Install libraries

The sketch uses `WiFi` and `HTTPClient` (built in to the Wio Terminal board
core) plus **`TFT_eSPI`** (a.k.a. `Seeed_Arduino_LCD`) for the on-screen
status display — install it via **Tools → Manage Libraries → search
"TFT_eSPI"**.

### 3. Configure

Copy `config.example.h` to `config.h` (same folder as the `.ino`):

```bash
cp config.example.h config.h
```

Edit `config.h`:
- `WIFI_SSID` — your 2.4 GHz WiFi network name
- `WIFI_PASS` — your WiFi password
- `FIREBASE_DB_URL` — your Firebase Realtime Database URL, e.g.
  ```
  https://your-project-default-rtdb.your-region.firebasedatabase.app
  ```
  Find it in the Firebase console under **Realtime Database**. This isn't a
  secret — the Wio has no credentials at all; instead, the database's
  security rules are scoped so only `/devices/*` is publicly readable/
  writable (see the "Firebase setup" section in `README_DEPLOY.md` for the
  exact rules JSON — set these up before flashing, or the Wio's writes will
  be rejected).

> Note: `config.h` should be git-ignored — never commit WiFi passwords.

### 4. Upload

1. Connect the Wio via USB-C.
2. **Tools → Board → Seeed SAMD (Wio Terminal)**
3. **Tools → Port → (the COM port that appears)**
4. Click **Upload**.
5. Open **Serial Monitor** at 115200 baud. You should see:
   ```
   [wio] AFibAI ECG streamer starting…
   [wio] connecting to your-wifi-ssid ......
   [wio] WiFi OK, IP = 192.168.x.x
   [wio] device_id = wio_01
   [wio] firebase  = https://your-project-default-rtdb.your-region.firebasedatabase.app
   [wio] fs        = 128
   ```
   The LCD should immediately show the card layout (Heart Rate, R-R Interval,
   Signal Quality, AFib Status, ECG trace, Recording Status), starting on
   `--` placeholders until the first result comes back from Firebase.

If you see `[wio] WiFi FAILED`, double-check the SSID and password (and that
your router is 2.4 GHz — 5 GHz networks won't work, the ESP32 in the Wio only
sees 2.4).

## Verifying it works

With the sketch running and `server.py` running somewhere with
`FIREBASE_DB_URL` + credentials set (see `README_DEPLOY.md`'s Firebase setup
section):

```bash
curl http://127.0.0.1:5000/api/wio/status?device_id=wio_01
```

You should see:
```json
{
  "connected": true,
  "last_chunk_age_s": 0.3,
  "total_samples_received": 1280,
  ...
}
```

This confirms `server.py`'s Firebase poller is picking up what the Wio wrote
to `/devices/wio_01/live`. Then open the Streamlit app, switch to **Wio Live
(Server)** mode, and the ECG trace should start scrolling within a couple of
seconds — Streamlit still only ever talks to `server.py`, unchanged.

On the Wio's own screen, within a few seconds of `server.py`'s prediction
loop running (`AFIBAI_PREDICT_INTERVAL`, default 2s), the Heart Rate, R-R
Interval, Signal Quality and AFib Status cards should fill in with real
values instead of `--`.

## Sample rate

The sketch samples at **128 Hz** by default, matching `FS = 128` in `app.py`.
If you change one, change the other (and the `fs` field the Wio sends), or
the analysis window in the app will be mis-sized.

## Testing without a Wio

If you don't have the hardware yet, run `mock_wio_firebase.py` from your
laptop — it exercises the exact path the real Wio now uses (Firebase, not a
direct POST to `server.py`):

```bash
python live_wio/mock_wio_firebase.py --db-url https://your-project-default-rtdb.your-region.firebasedatabase.app --bpm 75
```

(`mock_wio.py`, which POSTs straight to `server.py`, still works too — useful
if you want to test `server.py`/Streamlit without involving Firebase at all.)

## Troubleshooting

| Symptom                       | Fix                                                     |
|-------------------------------|---------------------------------------------------------|
| Upload fails                  | Wrong COM port, or board not in "Seeed SAMD" mode        |
| `[wio] WiFi FAILED`           | 2.4 GHz only; double-check SSID/password                |
| `[wio] PUT live failed: ...`  | Firebase unreachable — check `FIREBASE_DB_URL` and WiFi |
| `[wio] PUT live → HTTP 401/403` | RTDB rules don't allow public write to `/devices/*` — check the rules JSON |
| Screen stays on `--`          | `server.py` hasn't written a result yet — confirm it's running with Firebase configured, and that chunks are arriving (`/api/wio/status`) |
| ECG looks flat in app         | Leads off (electrode detached); re-attach and wait      |
| ECG is noisy                  | Move away from power supplies; clean electrode contact  |

## Hardware reference

- [Wio Terminal wiki](https://wiki.seeedstudio.com/Wio-Terminal-Getting-Started/)
- [Grove – ECG (AD8232) wiki](https://wiki.seeedstudio.com/Grove-ECG/)
- [Seeed AD8232 Arduino library](https://github.com/Seeed-Studio/Seeed_AD8232) (not required by this sketch, but useful for reference)