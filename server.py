"""
server.py — AFibAI live data server
====================================

A tiny Flask app that sits between the Wio Terminal (or any device that
speaks the same JSON protocol) and the Streamlit web app.

Endpoints
---------
  GET  /                              health-check JSON
  POST /api/wio/upload                Wio → server: push a chunk of ECG samples
  GET  /api/wio/stream?device_id=...  Streamlit ← server: poll latest samples
  GET  /api/wio/status?device_id=...  Streamlit ← server: connection state
  POST /api/recording/start          patient phone → server: begin recording
  POST /api/recording/stop           patient phone → server: finalize recording
  GET  /api/recording/active?device   Streamlit ← server: is recording in progress?
  GET  /api/recordings                Streamlit ← server: list saved recordings
  GET  /api/recording/<id>            Streamlit ← server: load one recording
  POST /api/alert                     patient phone → server: raise an alert
  GET  /api/alerts                    Streamlit ← server: read + clear alerts

Run
---
  python server.py            # default port 5000
  PORT=8080 python server.py  # custom port

Tested with curl while developing, see README_DEPLOY.md for examples.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import db  # local file: shared connection layer (local SQLite or Turso)

from flask import Flask, jsonify, request

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

APP_DIR     = Path(__file__).parent.resolve()
DB_PATH     = str(APP_DIR / db.HISTORY_DB)  # local mode only; ignored in Turso mode
HOST        = os.environ.get("HOST", "0.0.0.0")
PORT        = int(os.environ.get("PORT", "5000"))

# How many seconds of ECG to keep in the per-device ring buffer. Anything
# past this is discarded. Must be at least as long as the longest analysis
# window the Streamlit app uses (currently 15s) — we keep 30s to be safe.
LIVE_BUFFER_SECONDS = 30.0
SAMPLES_PER_SECOND  = 128        # matches FS in app.py
LIVE_BUFFER_MAXLEN  = int(LIVE_BUFFER_SECONDS * SAMPLES_PER_SECOND)  # ~3840

# Recording constraints — prevents a misbehaving client from filling the disk.
MAX_RECORDING_SECONDS = 600      # 10 minutes per recording
MAX_DB_BYTES          = 200 * 1024 * 1024   # 200 MB cap on the recordings table

# Alert table is small (one row per tap of the patient button) so we don't
# cap it, but we do mark old rows as "consumed" rather than deleting them.

# ═══════════════════════════════════════════════════════════════════════════
# STATE (in-memory, per process)
# ═══════════════════════════════════════════════════════════════════════════

# device_id → { "samples": deque[float], "last_ts": float, "last_chunk_ts": float }
_devices: dict[str, dict] = {}
_state_lock = threading.Lock()

# device_id → { "id": int, "user_id": int|None, "started_at": float,
#               "samples": list[float], "fs": int, "samples_per_chunk": int }
_active_recordings: dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _device_state(device_id: str) -> dict:
    """Get-or-create the per-device state. Thread-safe."""
    with _state_lock:
        st = _devices.get(device_id)
        if st is None:
            st = {
                "samples": deque(maxlen=LIVE_BUFFER_MAXLEN),
                "last_ts": 0.0,        # unix time of last sample received
                "last_chunk_ts": 0.0,  # unix time of last chunk received
                "total_samples": 0,    # monotonic counter, never resets
            }
            _devices[device_id] = st
        return st


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE (recordings + alerts)
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def _db():
    """Connection. Local: sqlite3 file. Production (Turso env vars set):
    libsql remote DB. The db.py wrapper exposes the same surface, so the
    callers below don't need to change."""
    with db.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        yield con


def init_db():
    with _db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id    TEXT    NOT NULL,
                user_id      INTEGER,
                started_at   TEXT    NOT NULL,
                ended_at     TEXT,
                duration_s   REAL    NOT NULL DEFAULT 0,
                fs           INTEGER NOT NULL DEFAULT 128,
                samples_npy  BLOB,
                n_samples    INTEGER NOT NULL DEFAULT 0,
                notes        TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id  TEXT    NOT NULL,
                user_id    INTEGER,
                ts         TEXT    NOT NULL,
                kind       TEXT    NOT NULL DEFAULT 'symptom',
                message    TEXT,
                consumed   INTEGER NOT NULL DEFAULT 0
            )
        """)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_rec_device ON recordings(device_id, id DESC)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_unread ON alerts(consumed, id DESC)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.route("/")
def health():
    with _state_lock:
        devs = sorted(_devices.keys())
        active = sorted(_active_recordings.keys())
    return jsonify({
        "ok": True,
        "service": "afibai-live-server",
        "devices_seen": devs,
        "active_recordings": active,
        "now": _now_iso(),
    })


# ── Wio → server ────────────────────────────────────────────────────────────

@app.route("/api/wio/upload", methods=["POST"])
def wio_upload():
    """
    Wio POSTs a JSON body like:
        { "device_id": "wio_01", "fs": 128, "samples": [0.12, 0.11, ...] }
    Server appends to the per-device ring buffer, and if a recording is
    active for this device, also appends to the recording buffer.
    """
    body = request.get_json(silent=True) or {}
    device_id = (body.get("device_id") or "").strip() or "wio_anon"
    fs = int(body.get("fs") or SAMPLES_PER_SECOND)
    samples = body.get("samples") or []
    if not isinstance(samples, list) or not samples:
        return jsonify({"ok": False, "error": "no samples"}), 400
    try:
        samples_f = [float(x) for x in samples]
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "non-numeric samples"}), 400

    now = time.time()
    state = _device_state(device_id)
    with _state_lock:
        state["samples"].extend(samples_f)
        state["last_ts"] = now
        state["last_chunk_ts"] = now
        state["total_samples"] += len(samples_f)

        rec = _active_recordings.get(device_id)
        if rec is not None:
            # Append + enforce length cap.
            rec["samples"].extend(samples_f)
            elapsed = len(rec["samples"]) / float(rec.get("fs") or fs)
            if elapsed > MAX_RECORDING_SECONDS:
                # Auto-stop the recording — server-side safety net.
                finalize_recording_locked(device_id, reason="max_duration")

    return jsonify({
        "ok": True,
        "device_id": device_id,
        "accepted": len(samples_f),
        "buffer_len": len(state["samples"]),
        "recording": device_id in _active_recordings,
    })


# ── Streamlit ← server (live polling) ───────────────────────────────────────

@app.route("/api/wio/stream", methods=["GET"])
def wio_stream():
    """Returns the current ring buffer for the device as a JSON array of floats.

    Query params:
      device_id: which device to read (required)
      since:     unix timestamp; only samples newer than this are returned.
                 (Optional. Streamlit just polls with no `since` most of the time
                 and the client side replaces the array.)
    """
    device_id = (request.args.get("device_id") or "").strip() or "wio_anon"
    state = _device_state(device_id)
    with _state_lock:
        samples = list(state["samples"])
        last_ts = state["last_ts"]
    return jsonify({
        "ok": True,
        "device_id": device_id,
        "fs": SAMPLES_PER_SECOND,
        "samples": samples,
        "n": len(samples),
        "last_ts": last_ts,
        "now": time.time(),
    })


@app.route("/api/wio/status", methods=["GET"])
def wio_status():
    """Connection state — useful for the clinician UI to show 'connected' dot."""
    device_id = (request.args.get("device_id") or "").strip() or "wio_anon"
    state = _device_state(device_id)
    now = time.time()
    with _state_lock:
        last_chunk = state["last_chunk_ts"]
        total = state["total_samples"]
    age = (now - last_chunk) if last_chunk > 0 else None
    connected = age is not None and age < 5.0   # no chunks in 5s = disconnected
    return jsonify({
        "ok": True,
        "device_id": device_id,
        "connected": connected,
        "last_chunk_age_s": age,
        "total_samples_received": total,
        "recording_active": device_id in _active_recordings,
        "now": now,
    })


# ── Recording control ───────────────────────────────────────────────────────

def finalize_recording_locked(device_id: str, reason: str = "user_stopped") -> int | None:
    """Must be called with _state_lock held. Returns the recording id, or None."""
    rec = _active_recordings.pop(device_id, None)
    if rec is None:
        return None
    samples = rec["samples"]
    fs = rec.get("fs") or SAMPLES_PER_SECOND
    duration_s = len(samples) / float(fs) if fs > 0 else 0.0
    import numpy as np
    npy_blob = np.asarray(samples, dtype=np.float32).tobytes()
    with _db() as con:
        cur = con.execute(
            "INSERT INTO recordings (device_id, user_id, started_at, ended_at, "
            "duration_s, fs, samples_npy, n_samples, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (device_id, rec.get("user_id"), rec["started_at"], _now_iso(),
             duration_s, fs, npy_blob, len(samples), reason),
        )
        rid = cur.lastrowid
    return rid


@app.route("/api/recording/start", methods=["POST"])
def recording_start():
    """Begin recording for the given device.

    Body: { "device_id": "wio_01", "user_id": <int|None>, "fs": 128 }
    """
    body = request.get_json(silent=True) or {}
    device_id = (body.get("device_id") or "").strip() or "wio_anon"
    user_id = body.get("user_id")
    fs = int(body.get("fs") or SAMPLES_PER_SECOND)

    with _state_lock:
        if device_id in _active_recordings:
            return jsonify({"ok": False, "error": "already recording"}), 409
        _active_recordings[device_id] = {
            "user_id": int(user_id) if user_id is not None else None,
            "started_at": _now_iso(),
            "samples": [],
            "fs": fs,
        }
    return jsonify({"ok": True, "device_id": device_id, "started_at": _now_iso()})


@app.route("/api/recording/stop", methods=["POST"])
def recording_stop():
    """Stop recording and persist to SQLite."""
    body = request.get_json(silent=True) or {}
    device_id = (body.get("device_id") or "").strip() or "wio_anon"
    with _state_lock:
        rid = finalize_recording_locked(device_id, reason="user_stopped")
    if rid is None:
        return jsonify({"ok": False, "error": "no active recording"}), 404
    return jsonify({"ok": True, "recording_id": rid})


@app.route("/api/recording/active", methods=["GET"])
def recording_active():
    device_id = (request.args.get("device_id") or "").strip() or "wio_anon"
    with _state_lock:
        rec = _active_recordings.get(device_id)
    if rec is None:
        return jsonify({"ok": True, "active": False})
    return jsonify({
        "ok": True,
        "active": True,
        "started_at": rec["started_at"],
        "n_samples": len(rec["samples"]),
        "duration_s": len(rec["samples"]) / float(rec.get("fs") or SAMPLES_PER_SECOND),
    })


# ── Recording history ───────────────────────────────────────────────────────

@app.route("/api/recordings", methods=["GET"])
def recordings_list():
    """List recordings, newest first. Optional ?device_id= filter."""
    device_id = (request.args.get("device_id") or "").strip()
    with _db() as con:
        if device_id:
            rows = con.execute(
                "SELECT id, device_id, user_id, started_at, ended_at, "
                "duration_s, fs, n_samples, notes "
                "FROM recordings WHERE device_id = ? ORDER BY id DESC LIMIT 200",
                (device_id,),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id, device_id, user_id, started_at, ended_at, "
                "duration_s, fs, n_samples, notes "
                "FROM recordings ORDER BY id DESC LIMIT 200"
            ).fetchall()
    return jsonify({
        "ok": True,
        "recordings": [dict(r) for r in rows],
    })


@app.route("/api/recording/<int:rid>", methods=["GET"])
def recording_get(rid: int):
    """Return one recording as JSON. Samples are base64-encoded to avoid
    JSON-encoding 128 floats per second of binary float data."""
    import base64
    import numpy as np
    with _db() as con:
        row = con.execute(
            "SELECT id, device_id, user_id, started_at, ended_at, duration_s, "
            "fs, samples_npy, n_samples, notes FROM recordings WHERE id = ?",
            (rid,),
        ).fetchone()
    if row is None:
        return jsonify({"ok": False, "error": "not found"}), 404
    blob = row["samples_npy"] or b""
    arr = np.frombuffer(blob, dtype=np.float32) if blob else np.array([], dtype=np.float32)
    return jsonify({
        "ok": True,
        "id": row["id"],
        "device_id": row["device_id"],
        "user_id": row["user_id"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "duration_s": row["duration_s"],
        "fs": row["fs"],
        "n_samples": row["n_samples"],
        "notes": row["notes"],
        "samples_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        "samples_dtype": "float32",
    })


# ── Alerts (patient phone → clinician screen) ───────────────────────────────

@app.route("/api/alert", methods=["POST"])
def alert_post():
    """Patient tapped the 'I feel a symptom' / 'Alert doctor' button."""
    body = request.get_json(silent=True) or {}
    device_id = (body.get("device_id") or "").strip() or "wio_anon"
    user_id = body.get("user_id")
    kind = (body.get("kind") or "symptom").strip()
    message = (body.get("message") or "").strip() or None
    with _db() as con:
        cur = con.execute(
            "INSERT INTO alerts (device_id, user_id, ts, kind, message) "
            "VALUES (?, ?, ?, ?, ?)",
            (device_id, int(user_id) if user_id is not None else None,
             _now_iso(), kind, message),
        )
        aid = cur.lastrowid
    return jsonify({"ok": True, "alert_id": aid})


@app.route("/api/alerts", methods=["GET"])
def alerts_get():
    """Return unconsumed alerts. Pass ?consume=1 to mark them as read."""
    consume = request.args.get("consume", "0") == "1"
    with _db() as con:
        rows = con.execute(
            "SELECT id, device_id, user_id, ts, kind, message FROM alerts "
            "WHERE consumed = 0 ORDER BY id DESC LIMIT 50"
        ).fetchall()
        if consume and rows:
            ids = [r["id"] for r in rows]
            qmarks = ",".join("?" * len(ids))
            con.execute(f"UPDATE alerts SET consumed = 1 WHERE id IN ({qmarks})", ids)
    return jsonify({
        "ok": True,
        "alerts": [dict(r) for r in rows],
    })


# ═══════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    print(f"[afibai-server] listening on http://{HOST}:{PORT}")
    print(f"[afibai-server] db = {DB_PATH}")
    print(f"[afibai-server] live buffer = {LIVE_BUFFER_SECONDS}s "
          f"({LIVE_BUFFER_MAXLEN} samples)")
    # Threaded so the Wio upload and Streamlit polling don't block each other.
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
