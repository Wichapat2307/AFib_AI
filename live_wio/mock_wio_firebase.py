"""
mock_wio_firebase.py — fake Wio Terminal that pushes straight to Firebase
===========================================================================

Same synthetic ECG generator as mock_wio.py, but instead of POSTing chunks to
server.py's /api/wio/upload, it PUTs them directly to the Firebase Realtime
Database at /devices/{device_id}/live — exactly what the Firebase-flashed
wio_ecg_streamer.ino now does. Use this to test the whole
Firebase → server.py → prediction → Firebase loop without real hardware.

server.py must be running with FIREBASE_DB_URL (and Firebase credentials) set
so its background poller picks up what this script writes.

Usage
-----
  python mock_wio_firebase.py --db-url https://your-project-default-rtdb.region.firebasedatabase.app
  python mock_wio_firebase.py --db-url ... --bpm 110
  python mock_wio_firebase.py --db-url ... --afib
  python mock_wio_firebase.py --db-url ... --device wio_lab --duration 60

--db-url can also be set via the FIREBASE_DB_URL environment variable.

Requires the Realtime Database rules to allow public write to /devices/* (see
README_DEPLOY.md) — this script has no service-account credentials, same as
the real Wio.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

from mock_wio import ecg_chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=os.environ.get("FIREBASE_DB_URL", ""),
                    help="Firebase RTDB URL, e.g. "
                         "https://your-project-default-rtdb.region.firebasedatabase.app "
                         "(or set FIREBASE_DB_URL)")
    ap.add_argument("--device", default="wio_01")
    ap.add_argument("--fs", type=int, default=128)
    ap.add_argument("--bpm", type=float, default=72.0)
    ap.add_argument("--afib", action="store_true",
                    help="Generate AFib-like RR irregularity.")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Stop after N seconds (0 = run forever).")
    args = ap.parse_args()

    db_url = args.db_url.rstrip("/")
    if not db_url:
        ap.error("--db-url is required (or set FIREBASE_DB_URL)")

    live_url = f"{db_url}/devices/{args.device}/live.json"

    print(f"[mock_wio_firebase] db_url  = {db_url}")
    print(f"[mock_wio_firebase] device  = {args.device}")
    print(f"[mock_wio_firebase] fs      = {args.fs} Hz")
    print(f"[mock_wio_firebase] bpm     = {args.bpm}")
    print(f"[mock_wio_firebase] afib    = {args.afib}")
    if args.duration:
        print(f"[mock_wio_firebase] duration = {args.duration}s")
    print("[mock_wio_firebase] Ctrl-C to stop.\n")

    start = time.time()
    t_offset = 0.0
    seq = 0
    chunks_sent = 0

    try:
        while True:
            chunk_start = time.time()
            samples = ecg_chunk(args.fs, args.bpm, args.afib, t_offset)
            t_offset += len(samples) / args.fs
            seq += 1
            body = json.dumps({
                "device_id": args.device,
                "fs": args.fs,
                "samples": samples,
                "seq": seq,
                "ts": int(time.time() * 1000),
            }).encode("utf-8")
            req = urllib.request.Request(
                live_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp.read()
            except urllib.error.URLError as e:
                print(f"[mock_wio_firebase] write failed: {e.reason}")
            chunks_sent += 1
            if chunks_sent % 5 == 0:
                elapsed = time.time() - start
                print(f"[mock_wio_firebase] {chunks_sent} chunks sent (seq={seq}), "
                      f"{elapsed:.1f}s elapsed")

            # Pace at 1 chunk/sec, matching the real Wio's cadence.
            elapsed_chunk = time.time() - chunk_start
            time.sleep(max(0.0, 1.0 - elapsed_chunk))

            if args.duration and (time.time() - start) >= args.duration:
                print(f"[mock_wio_firebase] reached --duration={args.duration}s, stopping")
                break
    except KeyboardInterrupt:
        print("\n[mock_wio_firebase] interrupted")


if __name__ == "__main__":
    main()
