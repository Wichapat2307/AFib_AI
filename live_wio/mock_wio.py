"""
mock_wio.py — fake Wio Terminal for testing the AFibAI server without hardware
===============================================================================

Generates plausible-looking ECG samples (a heart-rate signal with adjustable
rhythm regularity) and POSTs them to the server every second, exactly like
the real wio_ecg_streamer.ino sketch.

Usage
-----
  # Make sure server.py is running first.
  python mock_wio.py              # default: 70 BPM, normal sinus rhythm
  python mock_wio.py --bpm 110    # faster
  python mock_wio.py --afib       # introduce RR-interval irregularity
  python mock_wio.py --device wio_lab
  python mock_wio.py --server http://192.168.1.42:5000
  python mock_wio.py --duration 60 # stop after 60s (Ctrl-C otherwise)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
import urllib.error


def ecg_chunk(fs: int, bpm: float, afib: bool, t_offset: float) -> list[float]:
    """Generate `fs` samples of a plausible ECG-shaped waveform.

    The waveform is a simple sum of:
      - a P wave (small bump, ~0.10 s wide)
      - a QRS complex (sharp R-spike ~0.08 s wide)  ← the dominant feature
      - a T wave (broader bump, ~0.16 s wide)
    Each beat occupies 60/bpm seconds; in AFib, RR intervals are randomized.
    """
    out = [0.0] * fs
    beat_period = 60.0 / bpm

    # If AFib, generate per-beat RR intervals (in seconds) with high variance.
    rr_intervals: list[float] = []
    if afib:
        rr_intervals = [beat_period * random.gauss(1.0, 0.18)
                        for _ in range(int(fs / (beat_period * fs)) + 4)]
        rr_intervals = [max(0.3, min(1.4, r)) for r in rr_intervals]
    rr_idx = 0
    next_beat_at = 0.0  # seconds within this chunk where the next beat starts

    def gauss_bump(x: float, mu: float, sigma: float, amp: float) -> float:
        return amp * math.exp(-((x - mu) ** 2) / (2 * sigma * sigma))

    t = t_offset
    for i in range(fs):
        # Have we passed the next beat? If so, advance.
        if t >= next_beat_at:
            out[i] = 0.0
            # Compute when the next beat should start
            if afib:
                rr = rr_intervals[rr_idx % len(rr_intervals)]
                rr_idx += 1
            else:
                # tiny natural RR variability (normal sinus arrhythmia)
                rr = beat_period * (1.0 + random.gauss(0.0, 0.04))
            next_beat_at = t + rr
            beat_local_t = 0.0
        else:
            beat_local_t = t - (next_beat_at - (rr if afib else beat_period))
            if beat_local_t < 0:
                beat_local_t = 0.0

        # Waveform components, parameterized in seconds within a beat
        p  = gauss_bump(beat_local_t, 0.16, 0.025, 0.10)
        q  = gauss_bump(beat_local_t, 0.42, 0.010, -0.18)
        r  = gauss_bump(beat_local_t, 0.44, 0.012,  1.10)
        s  = gauss_bump(beat_local_t, 0.47, 0.014, -0.30)
        t_wave = gauss_bump(beat_local_t, 0.62, 0.040, 0.22)

        out[i] = p + q + r + s + t_wave

        # Tiny noise floor (mains + ADC)
        out[i] += random.gauss(0, 0.012)
        t += 1.0 / fs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:5000")
    ap.add_argument("--device", default="wio_mock")
    ap.add_argument("--fs", type=int, default=128)
    ap.add_argument("--bpm", type=float, default=72.0)
    ap.add_argument("--afib", action="store_true",
                    help="Generate AFib-like RR irregularity.")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Stop after N seconds (0 = run forever).")
    args = ap.parse_args()

    print(f"[mock_wio] server  = {args.server}")
    print(f"[mock_wio] device  = {args.device}")
    print(f"[mock_wio] fs      = {args.fs} Hz")
    print(f"[mock_wio] bpm     = {args.bpm}")
    print(f"[mock_wio] afib    = {args.afib}")
    if args.duration:
        print(f"[mock_wio] duration = {args.duration}s")
    print("[mock_wio] Ctrl-C to stop.\n")

    start = time.time()
    t_offset = 0.0
    chunks_sent = 0

    try:
        while True:
            chunk_start = time.time()
            samples = ecg_chunk(args.fs, args.bpm, args.afib, t_offset)
            t_offset += len(samples) / args.fs
            body = json.dumps({
                "device_id": args.device,
                "fs": args.fs,
                "samples": samples,
            }).encode("utf-8")
            req = urllib.request.Request(
                args.server + "/api/wio/upload",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp.read()
            except urllib.error.URLError as e:
                print(f"[mock_wio] upload failed: {e.reason}")
            chunks_sent += 1
            if chunks_sent % 5 == 0:
                elapsed = time.time() - start
                print(f"[mock_wio] {chunks_sent} chunks sent, "
                      f"{elapsed:.1f}s elapsed")

            # Pace at 1 chunk/sec.
            elapsed_chunk = time.time() - chunk_start
            sleep_for = max(0.0, 1.0 - elapsed_chunk)
            time.sleep(sleep_for)

            if args.duration and (time.time() - start) >= args.duration:
                print(f"[mock_wio] reached --duration={args.duration}s, stopping")
                break
    except KeyboardInterrupt:
        print("\n[mock_wio] interrupted")


if __name__ == "__main__":
    main()