// config.example.h — copy this to config.h and fill in your values.
//
// config.h is git-ignored (don't commit real WiFi passwords).

#ifndef AFIB_AI_CONFIG_H
#define AFIB_AI_CONFIG_H

// ── WiFi credentials ──────────────────────────────────────────────────────
// The Wio needs to join a 2.4 GHz network with internet access so it can
// reach Firebase. 5 GHz networks won't work — the ESP32 in the Wio only
// supports 2.4 GHz.
#define WIFI_SSID  "your-wifi-ssid"
#define WIFI_PASS  "your-wifi-password"

// ── Firebase Realtime Database ───────────────────────────────────────────
// The Wio talks *only* to Firebase now (not to server.py directly) — it PUTs
// ECG chunks to /devices/{DEVICE_ID}/live and reads the AFib result back
// from /devices/{DEVICE_ID}/result. server.py runs its own background poller
// that reads/writes the same paths.
//
// This is not a secret — it's just your project's database URL. The Wio has
// no service-account credentials; the Realtime Database rules instead allow
// public read/write scoped to /devices/* only (see README_DEPLOY.md for the
// exact rules JSON to paste into the Firebase console). Find your URL in the
// Firebase console under Realtime Database.
#define FIREBASE_DB_URL  "https://your-project-default-rtdb.your-region.firebasedatabase.app"

// ── Device ID ─────────────────────────────────────────────────────────────
// If you ever run more than one Wio, give each one a unique ID so the
// server (and Firebase paths) keep their data separate.
#define DEVICE_ID   "wio_01"

#endif