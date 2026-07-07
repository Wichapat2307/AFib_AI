"""
Extract 3 AFib and 3 Normal demo segments
from MIT-BIH AF Database annotations.
"""

import wfdb
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

RECORD = "04043"
AFDB_DIR = "MIT-BIH Atrial Fibrillation Database V1.0.0"

FS = 128
SEGMENT_SECONDS = 11
NUM_SAMPLES = 3

# ============================================================
# OUTPUT
# ============================================================

Path("samples").mkdir(exist_ok=True)

# ============================================================
# LOAD ECG
# ============================================================

record_path = f"{AFDB_DIR}/{RECORD}"

record = wfdb.rdrecord(record_path)
signal = record.p_signal[:, 0]

# ============================================================
# LOAD RHYTHM ANNOTATIONS
# ============================================================

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

    rhythm = aux[i].strip()

    if "(AFIB" in rhythm:
        afib_regions.append((start, end))

    elif "(N" in rhythm:
        normal_regions.append((start, end))

print("AFib regions:", len(afib_regions))
print("Normal regions:", len(normal_regions))

# ============================================================
# EXTRACT AFIB SEGMENTS
# ============================================================

needed = SEGMENT_SECONDS * FS

saved = 0

for start, end in afib_regions:

    if (end - start) < needed:
        continue

    segment = signal[start:start + needed]

    np.save(
        f"samples/afib_{saved + 1}.npy",
        segment
    )

    # UPDATED: Now tells the Record Name, Sample Index, and Time
    print(
        f"Saved AFib #{saved+1} "
        f"| Record: {RECORD} "
        f"| Sample: {start:,} "
        f"| Time: {start/FS:.1f}s"
    )

    saved += 1

    if saved >= NUM_SAMPLES:
        break

# ============================================================
# EXTRACT NORMAL SEGMENTS
# ============================================================

saved = 0

for start, end in normal_regions:

    if (end - start) < needed:
        continue

    segment = signal[start:start + needed]

    np.save(
        f"samples/normal_{saved + 1}.npy",
        segment
    )

    # UPDATED: Now tells the Record Name, Sample Index, and Time
    print(
        f"Saved Normal #{saved+1} "
        f"| Record: {RECORD} "
        f"| Sample: {start:,} "
        f"| Time: {start/FS:.1f}s"
    )

    saved += 1

    if saved >= NUM_SAMPLES:
        break

# ============================================================
# DONE
# ============================================================

print("\nFinished.\n")
print("Generated:")
print("samples/afib_1.npy")
print("samples/afib_2.npy")
print("samples/afib_3.npy")
print("samples/normal_1.npy")
print("samples/normal_2.npy")
print("samples/normal_3.npy")