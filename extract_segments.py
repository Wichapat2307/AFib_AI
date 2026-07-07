"""
Extract 3 AFib and 3 Normal demo segments
from MIT-BIH Atrial Fibrillation Database.

Output:
samples/
├── afib_1.npy
├── afib_2.npy
├── afib_3.npy
├── normal_1.npy
├── normal_2.npy
└── normal_3.npy
"""

import wfdb
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

RECORD = "04043"

AFDB_DIR = (
    Path("mit-bih-atrial-fibrillation-database-1.0.0")
    / "files"
)

SEGMENT_SECONDS = 30
NUM_AFIB = 3
NUM_NORMAL = 3

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("samples")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD RECORD
# ============================================================

record_path = str(AFDB_DIR / RECORD)

print(f"\nLoading record {RECORD}...")
print("Path:", record_path)

record = wfdb.rdrecord(record_path)

signal = record.p_signal[:, 0]  # Use Lead 1
fs = int(record.fs)

print(f"Sampling Rate : {fs} Hz")
print(f"Signal Length : {len(signal):,} samples")
print(f"Duration      : {len(signal)/(fs*3600):.2f} hours")

needed = SEGMENT_SECONDS * fs

# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print("\nLoading annotations...")

ann = wfdb.rdann(record_path, "atr")

samples = ann.sample
aux = ann.aux_note

print(f"Found {len(samples)} rhythm annotations")

# ============================================================
# FIND AFIB + NORMAL REGIONS
# ============================================================

afib_regions = []
normal_regions = []

for i in range(len(samples) - 1):

    start = samples[i]
    end = samples[i + 1]

    rhythm = aux[i].replace("\x00", "").strip()

    # AFIB
    if "(AFIB" in rhythm or rhythm == "AFIB":
        afib_regions.append((start, end))

    # NORMAL
    elif (
        "(N" in rhythm
        or rhythm == "N"
        or "(NSR" in rhythm
        or rhythm == "NSR"
    ):
        normal_regions.append((start, end))

print(f"AFib Regions   : {len(afib_regions)}")
print(f"Normal Regions : {len(normal_regions)}")

# ============================================================
# EXTRACT AFIB SEGMENTS
# ============================================================

print("\nExtracting AFib segments...")

saved = 0

for start, end in afib_regions:

    if (end - start) < needed:
        continue

    segment = signal[start:start + needed]

    np.save(
        OUTPUT_DIR / f"afib_{saved + 1}.npy",
        segment.astype(np.float32)
    )

    print(
        f"Saved AFib #{saved + 1}"
        f" | Sample: {start:,}"
        f" | Time: {start/fs:.1f}s"
    )

    saved += 1

    if saved >= NUM_AFIB:
        break

# ============================================================
# EXTRACT NORMAL SEGMENTS
# ============================================================

print("\nExtracting Normal segments...")

saved = 0

for start, end in normal_regions:

    if (end - start) < needed:
        continue

    segment = signal[start:start + needed]

    np.save(
        OUTPUT_DIR / f"normal_{saved + 1}.npy",
        segment.astype(np.float32)
    )

    print(
        f"Saved Normal #{saved + 1}"
        f" | Sample: {start:,}"
        f" | Time: {start/fs:.1f}s"
    )

    saved += 1

    if saved >= NUM_NORMAL:
        break

# ============================================================
# VERIFY OUTPUT FILES
# ============================================================

print("\nVerification:")

for file in sorted(OUTPUT_DIR.glob("*.npy")):

    arr = np.load(file)

    print(
        f"{file.name:<15}"
        f" shape={arr.shape}"
        f" duration={len(arr)/fs:.1f}s"
    )

print("\nDone!")