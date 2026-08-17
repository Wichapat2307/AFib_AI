/**
 * wio_ecg_streamer.ino — AFibAI live ECG streamer (Firebase edition)
 * =====================================================================
 *
 * Runs on a Seeed Studio Wio Terminal with the Grove – ECG sensor (AD8232).
 * Samples the analog ECG pin at ~128 Hz and pushes 1-second chunks straight
 * to Firebase Realtime Database — server.py (running elsewhere) polls
 * Firebase for those chunks, runs the AFib model on them, and writes a
 * result back to Firebase, which this sketch reads back and shows on the
 * built-in LCD.
 *
 *      AD8232 → Wio (this sketch) → Firebase → server.py (model) → Firebase → Wio LCD
 *
 * The Wio no longer talks to server.py directly at all — Firebase is its
 * only network peer, which means it no longer needs to be on the same WiFi
 * network as whatever machine is running server.py.
 *
 * ─── Wiring ────────────────────────────────────────────────────────────────
 *  Wio Terminal Grove port (the bottom one, marked "GROVE / D38") → AD8232
 *  AD8232 leads (RA / LA / RL) → patient via the snap electrodes.
 *
 * ─── Setup ────────────────────────────────────────────────────────────────
 *  1. Copy config.example.h to config.h (same folder).
 *  2. Edit config.h with your WiFi SSID/password and your Firebase project's
 *     Realtime Database URL (FIREBASE_DB_URL). See README_DEPLOY.md for the
 *     RTDB security rules that need to be set so this sketch can read/write
 *     without a service-account credential.
 *  3. Open this sketch in Arduino IDE.
 *  4. Board:  "Seeed SAMD (Wio Terminal)"
 *     Port:   (the one that appears after plugging in Wio via USB-C)
 *  5. Library dependency: TFT_eSPI (Seeed_Arduino_LCD), for the on-device
 *     status screen. WiFi/HTTPClient come with the Wio Terminal board core.
 *  6. Upload.
 *
 * ─── Protocol ─────────────────────────────────────────────────────────────
 *  Every 1 second the Wio PUTs to {FIREBASE_DB_URL}/devices/{DEVICE_ID}/live.json :
 *      { "device_id": "wio_01", "fs": 128, "seq": 42, "ts": 123456,
 *        "samples": [float, float, ...]   // exactly 128 samples per chunk
 *      }
 *  `seq` increments every chunk so server.py's poller can tell new data
 *  apart from a chunk it already ingested.
 *
 *  Every ~1.5 seconds the Wio GETs {FIREBASE_DB_URL}/devices/{DEVICE_ID}/result.json,
 *  which server.py keeps updated with the live prediction:
 *      { "label": "Normal"|"AFib", "prob": 0.0-1.0, "hr": float, "rr_ms": float,
 *        "quality": "GOOD"|"WEAK"|"POOR", "recording_active": bool,
 *        "recording_elapsed_s": float, "updated_at": iso8601, "seq": int }
 *
 *  See mock_wio_firebase.py for a Python equivalent you can run without the
 *  device, to test server.py's Firebase bridge end-to-end.
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <TFT_eSPI.h>
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
static const unsigned long RESULT_FETCH_PERIOD_MS = 1500UL;  // Wio ← Firebase cadence
static const unsigned long LCD_REFRESH_PERIOD_MS  = 200UL;   // dynamic-field redraw cadence

// ── State ────────────────────────────────────────────────────────────────
static float     chunkBuf[CHUNK_SAMPLES];
static int       chunkIdx           = 0;
static unsigned long lastChunkSentAt = 0;
static unsigned long lastResultFetchAt = 0;
static unsigned long lastLcdUpdateAt   = 0;
static uint32_t   chunkSeq           = 0;

// Latest values read back from Firebase (/devices/{id}/result), used to
// drive the LCD. "haveResult" stays false until the first successful fetch.
static bool   haveResult          = false;
static String latestLabel         = "--";
static String latestQuality       = "--";
static float  latestHr            = 0.0f;
static float  latestRR            = 0.0f;
static bool   latestRecording     = false;
static float  latestElapsedS      = 0.0f;

// ─────────────────────────────────────────────────────────────────────────
// LCD — card layout matching the AFibAI Wio mockup
// ─────────────────────────────────────────────────────────────────────────
TFT_eSPI tft = TFT_eSPI();

// Wio Terminal LCD is 320x240. Rotation 3 puts the USB-C port to the right
// in landscape, matching the mockup's orientation — flip to 1 if your unit
// comes out upside-down.
static const int SCREEN_W = 320;
static const int SCREEN_H = 240;

// Palette (RGB565), tuned to roughly match the mockup's dark card UI.
uint16_t COL_BG, COL_CARD, COL_CARD_BORDER, COL_TITLE_BLUE, COL_TEXT_DIM,
         COL_TEXT_BRIGHT, COL_GOOD, COL_WEAK, COL_POOR, COL_RED, COL_ECG_TRACE;

// Live ECG trace panel geometry — samples scroll left as new ones arrive.
static const int ECG_PANEL_X = 168, ECG_PANEL_Y = 40, ECG_PANEL_W = 140, ECG_PANEL_H = 84;
static int ecgTraceCol = 0;
static int ecgLastY = -1;

// Card geometry (2x2 stat cards on the left, ECG + recording panel on the right).
static const int CARD_X = 12,  CARD_W = 148, CARD_H = 44, CARD_GAP = 6;
static const int CARD1_Y = 40, CARD2_Y = CARD1_Y + CARD_H + CARD_GAP,
                  CARD3_Y = CARD2_Y + CARD_H + CARD_GAP, CARD4_Y = CARD3_Y + CARD_H + CARD_GAP;
static const int REC_PANEL_X = ECG_PANEL_X, REC_PANEL_Y = 132,
                  REC_PANEL_W = ECG_PANEL_W, REC_PANEL_H = 92;

void initPalette() {
    COL_BG          = tft.color565(10, 14, 20);
    COL_CARD        = tft.color565(18, 24, 32);
    COL_CARD_BORDER = tft.color565(40, 48, 58);
    COL_TITLE_BLUE  = tft.color565(60, 140, 255);
    COL_TEXT_DIM    = tft.color565(140, 150, 160);
    COL_TEXT_BRIGHT = TFT_WHITE;
    COL_GOOD        = tft.color565(60, 200, 100);
    COL_WEAK        = tft.color565(230, 180, 40);
    COL_POOR        = tft.color565(220, 70, 70);
    COL_RED         = tft.color565(220, 60, 60);
    COL_ECG_TRACE   = tft.color565(60, 220, 100);
}

// Cheap heart-icon primitive (two circles + a triangle) — good enough at
// small sizes, no custom font/bitmap needed.
void drawHeart(int cx, int cy, int r, uint16_t color) {
    tft.fillCircle(cx - r / 2, cy - r / 4, r / 2, color);
    tft.fillCircle(cx + r / 2, cy - r / 4, r / 2, color);
    tft.fillTriangle(cx - r, cy - r / 4, cx + r, cy - r / 4, cx, cy + r, color);
}

void drawCardFrame(int x, int y, int w, int h, const char* label) {
    tft.fillRoundRect(x, y, w, h, 6, COL_CARD);
    tft.drawRoundRect(x, y, w, h, 6, COL_CARD_BORDER);
    tft.setTextColor(COL_TEXT_DIM, COL_CARD);
    tft.setTextFont(1);
    tft.setTextSize(1);
    tft.setCursor(x + 8, y + 5);
    tft.print(label);
}

void drawStaticLayout() {
    tft.fillScreen(COL_BG);

    // Title bar.
    tft.setTextColor(COL_TITLE_BLUE, COL_BG);
    tft.setTextFont(4);
    tft.setCursor(10, 6);
    tft.print("AFibAI");
    drawHeart(102, 16, 6, COL_RED);
    tft.setTextColor(COL_TEXT_DIM, COL_BG);
    tft.setTextFont(2);
    tft.setCursor(10, 26);
    tft.print("ECG Monitoring");

    // Stat cards (values themselves are drawn/refreshed by updateDynamicFields()).
    drawCardFrame(CARD_X, CARD1_Y, CARD_W, CARD_H, "Heart Rate");
    drawCardFrame(CARD_X, CARD2_Y, CARD_W, CARD_H, "R-R Interval");
    drawCardFrame(CARD_X, CARD3_Y, CARD_W, CARD_H, "Signal Quality");
    drawCardFrame(CARD_X, CARD4_Y, CARD_W, CARD_H, "AFib Status");

    // ECG trace panel.
    tft.fillRoundRect(ECG_PANEL_X, ECG_PANEL_Y, ECG_PANEL_W, ECG_PANEL_H, 6, TFT_BLACK);
    tft.drawRoundRect(ECG_PANEL_X, ECG_PANEL_Y, ECG_PANEL_W, ECG_PANEL_H, 6, COL_CARD_BORDER);
    tft.setTextColor(COL_TEXT_DIM, TFT_BLACK);
    tft.setTextFont(1);
    tft.setCursor(ECG_PANEL_X + 6, ECG_PANEL_Y + 4);
    tft.print("ECG Signal");

    // Recording status panel.
    drawCardFrame(REC_PANEL_X, REC_PANEL_Y, REC_PANEL_W, REC_PANEL_H, "Recording Status");

    tft.setTextColor(COL_TEXT_DIM, COL_BG);
    tft.setTextFont(1);
    tft.setCursor(10, SCREEN_H - 14);
    tft.print("connecting...");
}

// Redraws just the bits that change — called after each Firebase result
// fetch (and, for connection/timer, on the faster LCD_REFRESH cadence).
void drawStatCard(int y, const char* value, const char* unit, uint16_t valueColor) {
    // Clear the value area only (right ~2/3 of the card) to avoid full flicker.
    tft.fillRect(CARD_X + 6, y + 16, CARD_W - 12, CARD_H - 20, COL_CARD);
    tft.setTextColor(valueColor, COL_CARD);
    tft.setTextFont(4);
    tft.setCursor(CARD_X + 8, y + 16);
    tft.print(value);
    if (unit[0] != '\0') {
        tft.setTextFont(2);
        tft.setTextColor(COL_TEXT_DIM, COL_CARD);
        tft.print(" ");
        tft.print(unit);
    }
}

uint16_t qualityColor(const String& q) {
    if (q == "GOOD") return COL_GOOD;
    if (q == "WEAK") return COL_WEAK;
    if (q == "POOR") return COL_POOR;
    return COL_TEXT_DIM;
}

void updateDynamicFields() {
    char buf[24];

    // Heart rate.
    if (haveResult && latestHr > 0) {
        snprintf(buf, sizeof(buf), "%d", (int)roundf(latestHr));
        drawStatCard(CARD1_Y, buf, "BPM", COL_TEXT_BRIGHT);
    } else {
        drawStatCard(CARD1_Y, "--", "BPM", COL_TEXT_DIM);
    }

    // R-R interval.
    if (haveResult && latestRR > 0) {
        snprintf(buf, sizeof(buf), "%d", (int)roundf(latestRR));
        drawStatCard(CARD2_Y, buf, "ms", COL_TEXT_BRIGHT);
    } else {
        drawStatCard(CARD2_Y, "--", "ms", COL_TEXT_DIM);
    }

    // Signal quality.
    drawStatCard(CARD3_Y, haveResult ? latestQuality.c_str() : "--", "",
                 haveResult ? qualityColor(latestQuality) : COL_TEXT_DIM);

    // AFib status.
    const char* afibText = "--";
    uint16_t afibColor = COL_TEXT_DIM;
    if (haveResult) {
        if (latestLabel == "AFib") { afibText = "POSSIBLE AFib"; afibColor = COL_POOR; }
        else if (latestLabel == "Normal") { afibText = "NOT DETECTED"; afibColor = COL_GOOD; }
    }
    tft.fillRect(CARD_X + 6, CARD4_Y + 16, CARD_W - 12, CARD_H - 20, COL_CARD);
    tft.setTextColor(afibColor, COL_CARD);
    tft.setTextFont(2);
    tft.setCursor(CARD_X + 8, CARD4_Y + 20);
    tft.print(afibText);

    // Recording status + timer.
    tft.fillRect(REC_PANEL_X + 6, REC_PANEL_Y + 18, REC_PANEL_W - 12, 20, COL_CARD);
    tft.setTextFont(2);
    tft.setCursor(REC_PANEL_X + 8, REC_PANEL_Y + 20);
    if (latestRecording) {
        tft.setTextColor(COL_RED, COL_CARD);
        tft.print("RECORDING");
        tft.fillCircle(REC_PANEL_X + REC_PANEL_W - 18, REC_PANEL_Y + 28, 8, COL_RED);
    } else {
        tft.setTextColor(COL_TITLE_BLUE, COL_CARD);
        tft.print("READY");
        tft.drawCircle(REC_PANEL_X + REC_PANEL_W - 18, REC_PANEL_Y + 28, 8, COL_TEXT_DIM);
    }
    int elapsed = (int)latestElapsedS;
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d", elapsed / 3600, (elapsed / 60) % 60, elapsed % 60);
    tft.fillRect(REC_PANEL_X + 6, REC_PANEL_Y + REC_PANEL_H - 20, REC_PANEL_W - 12, 16, COL_CARD);
    tft.setTextColor(COL_TEXT_BRIGHT, COL_CARD);
    tft.setTextFont(2);
    tft.setCursor(REC_PANEL_X + 8, REC_PANEL_Y + REC_PANEL_H - 18);
    tft.print(buf);

    // Bottom-left connection line.
    tft.fillRect(0, SCREEN_H - 16, 200, 16, COL_BG);
    tft.setTextColor(WiFi.status() == WL_CONNECTED ? COL_GOOD : COL_POOR, COL_BG);
    tft.setTextFont(1);
    tft.setCursor(10, SCREEN_H - 14);
    tft.print(WiFi.status() == WL_CONNECTED ? "CONNECTED" : "OFFLINE");
    if (haveResult) {
        tft.setTextColor(COL_TEXT_DIM, COL_BG);
        tft.print("  seq ");
        tft.print((int)chunkSeq);
    }
}

// Live-scrolling mini ECG trace, fed one (decimated) sample at a time as the
// ADC is read, independent of the once-a-second Firebase chunk cadence.
void pushEcgSample(float v) {
    int innerX = ECG_PANEL_X + 4, innerY = ECG_PANEL_Y + 18;
    int innerW = ECG_PANEL_W - 8, innerH = ECG_PANEL_H - 24;
    int x = innerX + ecgTraceCol;

    // Wipe a thin vertical strip ahead of the trace so it looks like a
    // scrolling scope rather than leaving old traces behind.
    tft.drawFastVLine(x, innerY, innerH, TFT_BLACK);
    int wipeAheadX = innerX + ((ecgTraceCol + 3) % innerW);
    tft.drawFastVLine(wipeAheadX, innerY, innerH, TFT_BLACK);

    int y = innerY + innerH / 2 - (int)(v * (innerH * 0.45f));
    if (y < innerY) y = innerY;
    if (y > innerY + innerH - 1) y = innerY + innerH - 1;

    if (ecgLastY >= 0 && ecgTraceCol > 0) {
        tft.drawLine(x - 1, ecgLastY, x, y, COL_ECG_TRACE);
    } else {
        tft.drawPixel(x, y, COL_ECG_TRACE);
    }
    ecgLastY = y;
    ecgTraceCol = (ecgTraceCol + 1) % innerW;
}

// ─────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(200);  // give the serial monitor time to attach

    Serial.println();
    Serial.println("[wio] AFibAI ECG streamer starting…");

    pinMode(ECG_PIN, INPUT);

    tft.init();
    tft.setRotation(3);
    initPalette();
    drawStaticLayout();

    connectWifi();

    Serial.print("[wio] device_id = "); Serial.println(DEVICE_ID);
    Serial.print("[wio] firebase  = "); Serial.println(FIREBASE_DB_URL);
    Serial.print("[wio] fs        = "); Serial.println(FS);

    updateDynamicFields();
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
    static int sampleCounter = 0;
    if (micros() >= nextSampleAt) {
        nextSampleAt = micros() + sampleIntervalUs;
        // read the pin and scale the 0..1023 ADC into roughly -1..+1 V units.
        // The Wio is 3.3V; AD8232 output is centered around Vcc/2, so we
        // subtract the mid-scale and divide. This is the same convention
        // used in predict.py's preprocess().
        int raw = analogRead(ECG_PIN);
        float v = (raw - 512.0f) / 512.0f;
        chunkBuf[chunkIdx++] = v;

        // Feed the on-screen live trace at a decimated ~32 Hz so SPI draw
        // calls don't compete with 128 Hz sampling.
        if ((sampleCounter++ % 4) == 0) {
            pushEcgSample(v);
        }
    }

    // 2. Send a chunk every CHUNK_PERIOD_MS ms.
    if (chunkIdx >= CHUNK_SAMPLES && (now - lastChunkSentAt) >= (unsigned long)CHUNK_PERIOD_MS) {
        sendChunk(chunkBuf, CHUNK_SAMPLES);
        chunkIdx = 0;
        lastChunkSentAt = now;
    }

    // 3. Poll Firebase for the latest prediction result.
    if (now - lastResultFetchAt >= RESULT_FETCH_PERIOD_MS) {
        lastResultFetchAt = now;
        fetchResult();
    }

    // 4. Redraw the dynamic LCD fields (connection dot, timer, etc.) at a
    //    steady cadence, independent of when new data actually arrives.
    if (now - lastLcdUpdateAt >= LCD_REFRESH_PERIOD_MS) {
        lastLcdUpdateAt = now;
        updateDynamicFields();
    }

    // 5. Watchdog: if WiFi drops, try to reconnect.
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
// Firebase — send ECG chunk
// ─────────────────────────────────────────────────────────────────────────
// Manually build the JSON to avoid pulling in ArduinoJson — keeps the sketch
// lean. Each chunk is exactly 128 samples; encoding "1.234567" as ~8 chars
// gives ~1 KB per chunk, well within HTTPClient's default buffer.
static char jsonOut[CHUNK_SAMPLES * 10 + 300];

void sendChunk(const float* samples, int n) {
    if (WiFi.status() != WL_CONNECTED) return;
    chunkSeq++;

    // Build JSON body. We keep it compact: no whitespace, fixed decimals.
    int pos = 0;
    pos += snprintf(jsonOut + pos, sizeof(jsonOut) - pos,
                    "{\"device_id\":\"%s\",\"fs\":%d,\"seq\":%lu,\"ts\":%lu,\"samples\":[",
                    DEVICE_ID, FS, (unsigned long)chunkSeq, (unsigned long)millis());
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

    // PUT overwrites /devices/{id}/live in place each chunk (we don't want
    // Firebase push() history here — server.py only ever needs the latest
    // chunk). ?print=silent tells Firebase not to echo the written data back,
    // which keeps the response small.
    HTTPClient http;
    String url = String(FIREBASE_DB_URL) + "/devices/" + DEVICE_ID + "/live.json?print=silent";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    int code = http.PUT((uint8_t*)jsonOut, pos);
    if (code > 0) {
        if (code != 200 && code != 204) {
            Serial.printf("[wio] PUT live → HTTP %d : %s\n", code, http.getString().c_str());
        }
    } else {
        Serial.printf("[wio] PUT live failed: %s\n", http.errorToString(code).c_str());
    }
    http.end();
}

// ─────────────────────────────────────────────────────────────────────────
// Firebase — fetch prediction result
// ─────────────────────────────────────────────────────────────────────────
// Lightweight manual JSON field extraction — the result object is flat (no
// nesting/arrays), so this stays simple without pulling in ArduinoJson.

bool jsonFindRaw(const String& json, const char* key, String& out) {
    String needle = String("\"") + key + "\":";
    int idx = json.indexOf(needle);
    if (idx < 0) return false;
    int start = idx + needle.length();
    int len = json.length();
    if (start < len && json[start] == '"') {
        int end = start + 1;
        while (end < len && json[end] != '"') end++;
        out = json.substring(start, end + 1);  // keeps the quotes
        return true;
    }
    int end = start;
    while (end < len && json[end] != ',' && json[end] != '}') end++;
    out = json.substring(start, end);
    out.trim();
    return true;
}

bool jsonGetString(const String& json, const char* key, String& out) {
    String raw;
    if (!jsonFindRaw(json, key, raw)) return false;
    if (raw.length() >= 2 && raw[0] == '"' && raw[raw.length() - 1] == '"') {
        out = raw.substring(1, raw.length() - 1);
    } else {
        out = raw;
    }
    return true;
}

bool jsonGetFloat(const String& json, const char* key, float& out) {
    String raw;
    if (!jsonFindRaw(json, key, raw)) return false;
    out = raw.toFloat();
    return true;
}

bool jsonGetBool(const String& json, const char* key, bool& out) {
    String raw;
    if (!jsonFindRaw(json, key, raw)) return false;
    out = (raw == "true");
    return true;
}

void fetchResult() {
    if (WiFi.status() != WL_CONNECTED) return;

    HTTPClient http;
    String url = String(FIREBASE_DB_URL) + "/devices/" + DEVICE_ID + "/result.json";
    http.begin(url);
    int code = http.GET();
    if (code == 200) {
        String payload = http.getString();
        // Firebase returns the literal string "null" for a path with no data.
        if (payload.length() > 4 && payload != "null") {
            String s; float v; bool b;
            if (jsonGetString(payload, "label", s))   latestLabel = s;
            if (jsonGetString(payload, "quality", s)) latestQuality = s;
            if (jsonGetFloat(payload, "hr", v))        latestHr = v;
            if (jsonGetFloat(payload, "rr_ms", v))     latestRR = v;
            if (jsonGetBool(payload, "recording_active", b)) latestRecording = b;
            if (jsonGetFloat(payload, "recording_elapsed_s", v)) latestElapsedS = v;
            haveResult = true;
        }
    } else if (code > 0) {
        Serial.printf("[wio] fetchResult → HTTP %d\n", code);
    } else {
        Serial.printf("[wio] fetchResult failed: %s\n", http.errorToString(code).c_str());
    }
    http.end();
}
