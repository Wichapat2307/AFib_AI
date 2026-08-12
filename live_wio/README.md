# live_wio — Wio Terminal ECG streamer

Arduino sketch for the Seeed Studio **Wio Terminal** that streams ECG samples
to the AFibAI server.

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

The sketch uses only built-in libraries (`WiFi`, `HTTPClient`). No extra
installs needed.

### 3. Configure

Copy `config.example.h` to `config.h` (same folder as the `.ino`):

```bash
cp config.example.h config.h
```

Edit `config.h`:
- `WIFI_SSID` — your 2.4 GHz WiFi network name
- `WIFI_PASS` — your WiFi password
- `SERVER_URL` — your `server.py` URL. For local dev:
  ```
  http://192.168.1.42:5000
  ```
  Replace `192.168.1.42` with your laptop's actual IP on the WiFi network.
  Run `ipconfig` (Windows) or `ifconfig` (mac/Linux) to find it.

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
   [wio] server    = http://192.168.1.42:5000
   [wio] fs        = 128
   ```

If you see `[wio] WiFi FAILED`, double-check the SSID and password (and that
your router is 2.4 GHz — 5 GHz networks won't work, the ESP32 in the Wio only
sees 2.4).

## Verifying it works

With the sketch running and `python server.py` running on your laptop:

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

Then open the Streamlit app, switch to **Wio Live (Server)** mode, and the
ECG trace should start scrolling within a couple of seconds.

## Optional: live trace on the Wio's own screen

The sketch includes commented-out code (search for `setupLcd` and
`drawStatus`) that draws a status bar and scrolling trace on the Wio's 2.4"
screen. Uncomment the two call sites in `setup()` and `loop()` and the
`TFT_eSPI` library usage at the bottom to enable it. You'll need to install
`TFT_eSPI` from the Arduino library manager.

## Sample rate

The sketch samples at **128 Hz** by default, matching `FS = 128` in `app.py`.
If you change one, change the other (and the `fs` field the Wio sends), or
the analysis window in the app will be mis-sized.

## Testing without a Wio

If you don't have the hardware yet, run `mock_wio.py` from your laptop:

```bash
python live_wio/mock_wio.py --bpm 75
```

This pushes ECG-shaped samples to the server every second, so you can test
the full pipeline (server → Streamlit → model → recording history) without
any hardware.

## Troubleshooting

| Symptom                       | Fix                                                     |
|-------------------------------|---------------------------------------------------------|
| Upload fails                  | Wrong COM port, or board not in "Seeed SAMD" mode        |
| `[wio] WiFi FAILED`           | 2.4 GHz only; double-check SSID/password                |
| `[wio] POST failed: ...`      | Server unreachable — check `SERVER_URL` and firewall    |
| ECG looks flat in app         | Leads off (electrode detached); re-attach and wait      |
| ECG is noisy                  | Move away from power supplies; clean electrode contact  |

## Hardware reference

- [Wio Terminal wiki](https://wiki.seeedstudio.com/Wio-Terminal-Getting-Started/)
- [Grove – ECG (AD8232) wiki](https://wiki.seeedstudio.com/Grove-ECG/)
- [Seeed AD8232 Arduino library](https://github.com/Seeed-Studio/Seeed_AD8232) (not required by this sketch, but useful for reference)