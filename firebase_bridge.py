"""
firebase_bridge.py — thin Firebase Realtime Database wrapper
==============================================================

Used by server.py to read live ECG chunks the Wio pushes to Firebase, and to
write back the live prediction result the Wio polls for its screen.

Credentials (checked in this order):
  1. FIREBASE_CRED_PATH  — path to a service-account JSON key file.
  2. FIREBASE_CRED_JSON  — the key file's contents, as a raw JSON string
                            (for hosts like Render where you set env vars
                            instead of dropping files on disk).
  3. Auto-detect          — first *firebase-adminsdk*.json file found next
                            to this module (local dev convenience).

Database URL: FIREBASE_DB_URL env var, e.g.
  https://afibai-45c89-default-rtdb.asia-southeast1.firebasedatabase.app/

If FIREBASE_DB_URL isn't set, enabled() returns False and server.py simply
doesn't start the Firebase polling thread — the existing /api/wio/upload
HTTP path keeps working unconditionally either way.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("afibai.firebase")

APP_DIR = Path(__file__).parent.resolve()

_app = None
_db_ref_cache: dict[str, object] = {}


def enabled() -> bool:
    return bool(os.environ.get("FIREBASE_DB_URL"))


def _find_cred_file() -> Path | None:
    matches = sorted(APP_DIR.glob("*firebase-adminsdk*.json"))
    return matches[0] if matches else None


def _init():
    global _app
    if _app is not None:
        return _app

    import firebase_admin
    from firebase_admin import credentials

    db_url = os.environ.get("FIREBASE_DB_URL")
    if not db_url:
        raise RuntimeError("FIREBASE_DB_URL is not set")

    cred_path = os.environ.get("FIREBASE_CRED_PATH")
    cred_json = os.environ.get("FIREBASE_CRED_JSON")

    if cred_path:
        cred = credentials.Certificate(cred_path)
    elif cred_json:
        cred = credentials.Certificate(json.loads(cred_json))
    else:
        found = _find_cred_file()
        if found is None:
            raise RuntimeError(
                "No Firebase credentials found — set FIREBASE_CRED_PATH, "
                "FIREBASE_CRED_JSON, or place a *firebase-adminsdk*.json "
                "key file next to firebase_bridge.py"
            )
        cred = credentials.Certificate(str(found))

    _app = firebase_admin.initialize_app(cred, {"databaseURL": db_url})
    log.info("Firebase initialized (db=%s)", db_url)
    return _app


def _ref(path: str):
    from firebase_admin import db as fb_db
    _init()
    if path not in _db_ref_cache:
        _db_ref_cache[path] = fb_db.reference(path)
    return _db_ref_cache[path]


def read_live_chunk(device_id: str) -> dict | None:
    """Read /devices/{device_id}/live. Returns None on any error or if the
    node is empty/malformed — callers should treat that as "nothing new"."""
    try:
        data = _ref(f"/devices/{device_id}/live").get()
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:
        log.warning("read_live_chunk(%s) failed: %s", device_id, e)
        return None


def write_result(device_id: str, result: dict) -> None:
    try:
        _ref(f"/devices/{device_id}/result").set(result)
    except Exception as e:
        log.warning("write_result(%s) failed: %s", device_id, e)
