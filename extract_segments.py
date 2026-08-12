"""
extract_samples.py
==================
Extracts demo segments from MIT-BIH AFib Database.

- 3 Normal segments
- 3 AFib segments that transition from Normal → AFib
  so the sliding window demo can show the detection working.

Uses the second ECG lead (LEAD_INDEX = 1), auto-detects and corrects
inverted leads (same logic as colour.py's is_inverted()), and labels
regions using the same rhythm-label sets as colour.py so "Normal" and
"AFib" are classified identically across both scripts.

Output: samples/normal_1.npy ... normal_3.npy
        samples/afib_1.npy   ... afib_3.npy
All resampled to 128 Hz, 30 seconds each.
"""

import wfdb
import numpy as np
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd

# ============================================================
# CONFIG
# ============================================================

AFDB_DIR     = Path(r"C:\Users\Admin\Desktop\AFib_detection\mit-bih-atrial-fibrillation-database-1.0.0\files")
OUTPUT_DIR   = Path("samples")
TARGET_FS    = 128
SEG_SECONDS  = 30
TRANS_SECONDS = 30   # for transition segments: extract 30s spanning the Normal→AFib boundary

# MIT-BIH AFib records have 2 ECG leads (columns in p_signal).
# LEAD_INDEX = 0 -> first lead (ECG1)
# LEAD_INDEX = 1 -> second lead (ECG2)
LEAD_INDEX = 1

OUTPUT_DIR.mkdir(exist_ok=True)

# Records to try (in order of preference)
RECORDS = ["04015","04043","04048","04126","04746","04908","05091","05121","06426","06453"]

# ============================================================
# RHYTHM LABELS  (kept in sync with colour.py)
# ============================================================

AFIB_LABELS = {"(AFIB", "AFIB"}
NORMAL_LABELS = {"(N", "N", "(NSR", "NSR"}
EXCLUDED_RHYTHMS = {
    "(AFL", "AFL", "(VT", "VT", "(SVTA", "SVTA",
    "(B", "B", "(SBR", "SBR", "(T", "T",
    "(IVR", "IVR", "(AB", "AB",
    "MISSB", "PSE", "MB", "M"
}

# ============================================================
# INVERSION DETECTION  (same approach as colour.py's is_inverted())
# ============================================================

def is_inverted(signal, fs, check_seconds=30):
    n      = min(len(signal), int(check_seconds * fs))
    sample = signal[:n]
    pos    = np.percentile(sample, 98)
    neg    = abs(np.percentile(sample, 2))
    return neg > pos

# ============================================================
# HELPERS
# ============================================================

def load_record(record_id, lead_index=LEAD_INDEX):
    path   = str(AFDB_DIR / record_id)
    record = wfdb.rdrecord(path)
    ann    = wfdb.rdann(path, "atr")

    n_leads = record.p_signal.shape[1]
    if lead_index >= n_leads:
        print(f"  {record_id}: only {n_leads} lead(s) available, "
              f"falling back to lead 0 instead of {lead_index}")
        lead_index = 0

    signal = record.p_signal[:, lead_index].astype(np.float32)
    fs     = int(record.fs)

    # Auto-detect and correct an inverted lead, same rule as colour.py
    if is_inverted(signal, fs):
        signal = -signal
        print(f"  {record_id} lead {lead_index}: INVERTED — flipped")
    else:
        print(f"  {record_id} lead {lead_index}: normal polarity")

    return signal, fs, ann

def resample_to_128(seg, fs):
    if fs == TARGET_FS:
        return seg.astype(np.float32)
    g   = gcd(TARGET_FS, fs)
    out = resample_poly(seg, TARGET_FS // g, fs // g)
    return out.astype(np.float32)

def get_rhythms(ann, signal_len):
    """Returns list of (start_sample, end_sample, rhythm_str)."""
    regions = []
    samples = ann.sample
    aux     = ann.aux_note
    for i in range(len(samples) - 1):
        start  = samples[i]
        end    = min(samples[i+1], signal_len)
        rhythm = aux[i].replace("\x00","").strip()
        regions.append((start, end, rhythm))
    # last region
    if len(samples) > 0:
        regions.append((samples[-1], signal_len,
                        aux[-1].replace("\x00","").strip()))
    return regions

def is_afib(r):
    return any(x in r for x in AFIB_LABELS)

def is_normal(r):
    return any(x in r for x in NORMAL_LABELS)

def rr_regularity_ok(seg, fs, cv_threshold=0.12, min_peaks=5):
    """
    Quick sanity check that a candidate 'Normal' segment is actually
    rhythmically regular, using the same style of R-peak detection as
    the app's own detect_rpeaks(). Some MIT-BIH regions are annotated
    'N' but still contain enough artifacts/ectopic beats that the
    HRV features look irregular and get misclassified as AFib by the
    app. This filters those out before they're ever saved.
    """
    from scipy.signal import find_peaks
    sig = seg.astype(np.float32) - np.mean(seg)
    if np.abs(sig.min()) > np.abs(sig.max()):
        sig = -sig
    sig_max = float(np.max(sig)) if np.max(sig) > 0 else 1.0
    thr = max(0.3, sig_max * 0.3)
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.3 * fs))
    if len(peaks) < min_peaks:
        return False
    rr = np.diff(peaks) / fs * 1000
    rr = rr[(rr > 250) & (rr < 2000)]
    if len(rr) < 4:
        return False
    cv = float(np.std(rr) / np.mean(rr))
    return cv <= cv_threshold

# ============================================================
# EXTRACT PURE NORMAL SEGMENTS
# ============================================================

print(f"\n=== Extracting Normal segments (lead index {LEAD_INDEX}) ===")
normal_segs = []

# How many valid, rhythmically-regular candidate regions to skip, per slot,
# before accepting one. Slot indices are 0-based: 0 -> normal_1,
# 1 -> normal_2, 2 -> normal_3. Bump a slot's number any time you want
# that particular sample swapped out for a different region on the next run.
NORMAL_SKIP = {0: 0, 1: 1, 2: 1}
skip_used = {0: 0, 1: 0, 2: 0}

for rec_id in RECORDS:
    if len(normal_segs) >= 3:
        break
    try:
        signal, fs, ann = load_record(rec_id)
    except Exception as e:
        print(f"  {rec_id} SKIP: {e}")
        continue

    needed = SEG_SECONDS * fs
    regions = get_rhythms(ann, len(signal))

    for start, end, rhythm in regions:
        if len(normal_segs) >= 3:
            break
        if not is_normal(rhythm):
            continue
        if (end - start) < needed * 1.5:
            continue

        # Take a segment from the middle of the normal region
        mid   = (start + end) // 2
        s_start = mid - needed // 2
        s_end   = s_start + needed
        if s_start < 0 or s_end > len(signal):
            s_start = start + needed // 4
            s_end   = s_start + needed

        seg = signal[s_start:s_end]
        if len(seg) < needed:
            continue

        # Reject segments that aren't actually rhythmically regular,
        # even if the database annotates the region as Normal.
        if not rr_regularity_ok(seg, fs):
            print(f"  (rejecting irregular 'Normal' candidate from {rec_id} "
                  f"@ {s_start/fs:.1f}s — RR variability too high)")
            continue

        slot = len(normal_segs)
        skip_target = NORMAL_SKIP.get(slot, 0)
        if skip_used[slot] < skip_target:
            skip_used[slot] += 1
            print(f"  (skipping clean candidate for Normal #{slot+1} from "
                  f"{rec_id} @ {s_start/fs:.1f}s — using a later one instead)")
            continue

        seg_128 = resample_to_128(seg, fs)
        normal_segs.append({
            "data":   seg_128,
            "record": rec_id,
            "time":   f"{s_start/fs:.1f}s",
            "sample": f"{s_start:,}",
            "label":  "Normal",
        })
        print(f"  Normal #{len(normal_segs)} from {rec_id} @ {s_start/fs:.1f}s")
        if len(normal_segs) >= 3:
            break

# ============================================================
# EXTRACT TRANSITION SEGMENTS (Normal → AFib)
# ============================================================
# For each transition segment we take a window that starts in
# Normal rhythm and crosses into AFib, so the sliding window
# in the app will first show Normal then flip to AFib.

print("\n=== Extracting Normal→AFib transition segments ===")
afib_segs = []

for rec_id in RECORDS:
    if len(afib_segs) >= 3:
        break
    try:
        signal, fs, ann = load_record(rec_id)
    except Exception as e:
        print(f"  {rec_id} SKIP: {e}")
        continue

    needed  = SEG_SECONDS * fs
    regions = get_rhythms(ann, len(signal))

    for i in range(len(regions) - 1):
        s1, e1, r1 = regions[i]
        s2, e2, r2 = regions[i+1]

        # We want Normal → AFib boundary
        if not (is_normal(r1) and is_afib(r2)):
            continue

        boundary = e1   # = s2 = start of AFib region

        # Extract a window spanning TRANS_SECONDS total, split 1/3 before
        # the boundary (Normal) and 2/3 after it (AFib).
        # This means the sliding window will initially detect Normal
        # then detect AFib as it moves into the AFib region
        pre_secs  = TRANS_SECONDS // 3
        post_secs = TRANS_SECONDS - pre_secs
        total_secs = pre_secs + post_secs

        seg_start = boundary - pre_secs * fs
        seg_end   = boundary + post_secs * fs

        if seg_start < 0 or seg_end > len(signal):
            continue

        seg     = signal[seg_start:seg_end]
        seg_128 = resample_to_128(seg, fs)

        afib_segs.append({
            "data":     seg_128,
            "record":   rec_id,
            "time":     f"{seg_start/fs:.1f}s",
            "sample":   f"{seg_start:,}",
            "boundary": f"{boundary/fs:.1f}s",
            "total_s":  total_secs,
            "label":    "AFib",
        })
        print(f"  AFib #{len(afib_segs)} from {rec_id} "
              f"| boundary @ {boundary/fs:.1f}s "
              f"| window {seg_start/fs:.1f}s–{seg_end/fs:.1f}s "
              f"| {pre_secs}s Normal + {post_secs}s AFib")

        if len(afib_segs) >= 3:
            break

# ============================================================
# SAVE
# ============================================================

print("\n=== Saving files ===")

for i, seg in enumerate(normal_segs):
    path = OUTPUT_DIR / f"normal_{i+1}.npy"
    np.save(path, seg["data"])
    print(f"  {path.name}  shape={seg['data'].shape}  "
          f"duration={len(seg['data'])/TARGET_FS:.1f}s  "
          f"record={seg['record']}  time={seg['time']}  "
          f"label={seg['label']}")

for i, seg in enumerate(afib_segs):
    path = OUTPUT_DIR / f"afib_{i+1}.npy"
    np.save(path, seg["data"])
    print(f"  {path.name}  shape={seg['data'].shape}  "
          f"duration={len(seg['data'])/TARGET_FS:.1f}s  "
          f"record={seg['record']}  time={seg['time']}  "
          f"N→AFib boundary @ {seg['boundary']}  label={seg['label']}")

# ============================================================
# VERIFY
# ============================================================

print("\n=== Verification ===")
for f in sorted(OUTPUT_DIR.glob("*.npy")):
    arr = np.load(f)
    print(f"  {f.name:<15} shape={arr.shape}  "
          f"duration={len(arr)/TARGET_FS:.1f}s @ {TARGET_FS}Hz")

print("\nDone! Copy the samples/ folder to your Streamlit app directory.")
print("The AFib segments start with Normal rhythm then transition to AFib,")
print("so the sliding window will show Normal→AFib detection in real time.")