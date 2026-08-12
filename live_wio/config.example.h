// config.example.h — copy this to config.h and fill in your values.
//
// config.h is git-ignored (don't commit real WiFi passwords).

#ifndef AFIB_AI_CONFIG_H
#define AFIB_AI_CONFIG_H

// ── WiFi credentials ──────────────────────────────────────────────────────
// The Wio needs to join a 2.4 GHz network with internet access so it can
// reach SERVER_URL. 5 GHz networks won't work — the ESP32 in the Wio only
// supports 2.4 GHz.
#define WIFI_SSID  "your-wifi-ssid"
#define WIFI_PASS  "your-wifi-password"

// ── Server URL ────────────────────────────────────────────────────────────
// Where should the Wio POST its ECG chunks?
//
//   Local dev  (laptop running python server.py on the same WiFi):
//     "http://192.168.1.42:5000"      ← replace 192.168.1.42 with your
//                                       laptop's actual IP (run `ipconfig`
//                                       on Windows or `ifconfig` on mac).
//
//   Deployed    (Render / Railway / etc.):
//     "https://afibai-live.example.com"
//
#define SERVER_URL  "http://192.168.1.42:5000"

// ── Device ID ─────────────────────────────────────────────────────────────
// If you ever run more than one Wio, give each one a unique ID so the
// server can keep their data separate.
#define DEVICE_ID   "wio_01"

#endif