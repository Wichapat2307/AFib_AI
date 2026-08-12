/**
 * wio_ecg_streamer.ino — AFibAI live ECG streamer
 * =================================================
 *
 * Runs on a Seeed Studio Wio Terminal with the Grove – ECG sensor (AD8232).
 * Samples the analog ECG pin at ~128 Hz and POSTs 1-second chunks to the
 * AFibAI server as JSON.
 *
 * ─── Wiring ────────────────────────────────────────────────────────────────
 *  Wio Terminal Grove port (the bottom one, marked "GROVE / D38") → AD8232
 *  AD8232 leads (RA / LA / RL) → patient via the snap electrodes.
 *
 * ─── Setup ────────────────────────────────────────────────────────────────
 *  1. Copy config.example.h to config.h (same folder).
 *  2. Edit config.h with your WiFi SSID/password and the server URL.
 *  3. Open this sketch in Arduino IDE.
 *  4. Board:  "Seeed SAMD (Wio Terminal)"
 *     Port:   (the one that appears after plugging in Wio via USB-C)
 *  5. Upload.
 *
 * ─── Protocol ─────────────────────────────────────────────────────────────
 *  Every 1 second the Wio POSTs to SERVER_URL/api/wio/upload :
 *      { "device_id": "wio_01",
 *        "fs": 128,
 *        "samples": [float, float, ...]   // exactly 128 samples per chunk
 *      }
 *
 *  See mock_wio.py for a Python equivalent you can run without the device.
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "config.h"

// ── AD8232 connections on the Wio Terminal ────────────────────────────────
// The Grove port at the bottom of the Wio Terminal maps to:
//   A4  (analog)   → AD8232 OUT
//   D38 (digital)  → AD8232 LO+ (used to detect "leads off")
// For now we ignore the leads-off pin (the server treats flat-line as
// "leads off" anyway, since real AFib patients don't have a literal 0.0 ECG).
const int ECG_PIN = A4;

// ── Sampling settings ─────────────────────────────────────────────────────
static const int   FS                = 128;       // Hz
static const int   CHUNK_SAMPLES     = FS;        // exactly 1 second per chunk
static const float CHUNK_PERIOD_MS   = 1000.0f;   // wall-clock between chunks

// ── State ────────────────────────────────────────────────────────────────
static float     chunkBuf[CHUNK_SAMPLES];
static int       chunkIdx           = 0;
static unsigned long lastChunkSentAt = 0;

// Wio LCD refresh (heart icon + WiFi signal). Optional — comment out the
// body of drawStatus() to disable.
static unsigned long lastLcdUpdateAt  = 0;

// ─────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);  // give the serial monitor time to attach

    // If you want the live trace on the Wio screen, uncomment these:
    // setupLcd();

    Serial.println();
    Serial.println("[wio] AFibAI ECG streamer starting…");

    pinMode(ECG_PIN, INPUT);

    connectWifi();

    Serial.print("[wio] device_id = "); Serial.println(DEVICE_ID);
    Serial.print("[wio] server    = "); Serial.println(SERVER_URL);
    Serial.print("[wio] fs        = "); Serial.println(FS);
}

// ─────────────────────────────────────────────────────────────────────────
// Loop
// ─────────────────────────────────────────────────────────────────────────
void loop() {
    const unsigned long now = millis();

    // 1. Sample at ~FS Hz. We don't use delay() because millis()-based
    //    pacing drifts less and stays accurate if a sample is slow.
    const unsigned long sampleIntervalUs = 1000000UL / FS;
    static unsigned long nextSampleAt = 0;
    if (micros() >= nextSampleAt) {
        nextSampleAt = micros() + sampleIntervalUs;
        // read the pin and scale the 0..1023 ADC into roughly -1..+1 V units.
        // The Wio is 3.3V; AD8232 output is centered around Vcc/2, so we
        // subtract the mid-scale and divide. This is the same convention
        // used in app.py's preprocess().
        int raw = analogRead(ECG_PIN);
        float v = (raw - 512.0f) / 512.0f;
        chunkBuf[chunkIdx++] = v;
    }

    // 2. Send a chunk every CHUNK_PERIOD_MS ms.
    if (chunkIdx >= CHUNK_SAMPLES && (now - lastChunkSentAt) >= (unsigned long)CHUNK_PERIOD_MS) {
        sendChunk(chunkBuf, CHUNK_SAMPLES);
        chunkIdx = 0;
        lastChunkSentAt = now;
    }

    // 3. Cheap LCD refresh — runs at most ~5 Hz.
    // if (now - lastLcdUpdateAt > 200UL) {
    //     drawStatus();
    //     lastLcdUpdateAt = now;
    // }

    // 4. Watchdog: if WiFi drops, try to reconnect.
    static unsigned long lastWifiCheckAt = 0;
    if (now - lastWifiCheckAt > 15000UL) {
        lastWifiCheckAt = now;
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[wio] WiFi lost — reconnecting");
            connectWifi();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// WiFi
// ─────────────────────────────────────────────────────────────────────────
void connectWifi() {
    Serial.printf("[wio] connecting to %s ", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[wio] WiFi OK, IP = %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[wio] WiFi FAILED — will retry in loop.");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// HTTP POST
// ─────────────────────────────────────────────────────────────────────────
// Manually build the JSON to avoid pulling in ArduinoJson — keeps the sketch
// lean. Each chunk is exactly 128 samples; encoding "1.234567" as ~8 chars
// gives ~1 KB per chunk, well within HTTPClient's default buffer.
static char jsonOut[CHUNK_SAMPLES * 10 + 256];

void sendChunk(const float* samples, int n) {
    if (WiFi.status() != WL_CONNECTED) return;

    // Build JSON body. We keep it compact: no whitespace, fixed decimals.
    // Using dtostrf to format each float.
    int pos = 0;
    pos += snprintf(jsonOut + pos, sizeof(jsonOut) - pos,
                    "{\"device_id\":\"%s\",\"fs\":%d,\"samples\":[",
                    DEVICE_ID, FS);
    for (int i = 0; i < n; i++) {
        char tmp[16];
        dtostrf(samples[i], 1, 4, tmp);   // 4 decimal places is plenty
        int written = snprintf(jsonOut + pos, sizeof(jsonOut) - pos,
                               "%s%s", (i == 0 ? "" : ","), tmp);
        if (written < 0 || written >= (int)(sizeof(jsonOut) - pos)) {
            Serial.println("[wio] JSON buffer overflow — skipping rest of chunk");
            break;
        }
        pos += written;
    }
    if (pos < (int)sizeof(jsonOut) - 2) {
        jsonOut[pos++] = ']';
        jsonOut[pos++] = '}';
        jsonOut[pos]   = '\0';
    }

    HTTPClient http;
    String url = String(SERVER_URL) + "/api/wio/upload";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST((uint8_t*)jsonOut, pos);
    if (code > 0) {
        if (code != 200) {
            Serial.printf("[wio] POST → HTTP %d : %s\n", code, http.getString().c_str());
        }
    } else {
        Serial.printf("[wio] POST failed: %s\n", http.errorToString(code).c_str());
    }
    http.end();
}

// ─────────────────────────────────────────────────────────────────────────
// Optional Wio-screen status display (commented out by default — enable
// in setup() and loop() if you want the live trace on the device).
// ─────────────────────────────────────────────────────────────────────────
//
// #include <TFT_eSPI.h>
// static TFT_eSprite sprite = TFT_eSprite(&M5.Lcd);
//
// void setupLcd() {
//     M5.Lcd.fillScreen(TFT_BLACK);
//     M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
//     M5.Lcd.setTextSize(2);
//     M5.Lcd.setCursor(10, 10);
//     M5.Lcd.println("AFibAI Wio");
// }
//
// static int traceIdx = 0;
// static const int TRACE_W = 320;
// void drawStatus() {
//     M5.Lcd.fillRect(0, 0, 320, 24, TFT_BLACK);
//     M5.Lcd.setTextSize(2);
//     M5.Lcd.setTextColor(TFT_GREEN, TFT_BLACK);
//     M5.Lcd.setCursor(4, 4);
//     M5.Lcd.printf("AFibAI  ");
//     M5.Lcd.setTextColor(WiFi.status() == WL_CONNECTED ? TFT_GREEN : TFT_RED, TFT_BLACK);
//     M5.Lcd.printf(WiFi.status() == WL_CONNECTED ? "ONLINE" : "OFFLINE");
//
//     // scrolling trace
//     int h = 60;
//     int y0 = 40;
//     int x = (traceIdx++) % TRACE_W;
//     int v = (int)((chunkBuf[chunkIdx > 0 ? chunkIdx - 1 : 0] + 1.0f) * (h / 2));
//     M5.Lcd.drawPixel(x, y0 + v, TFT_WHITE);
// }