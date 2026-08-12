"""
regen_normals.py
================
Regenerate only `samples/normal_2.npy` and `samples/normal_3.npy` from
MIT-BIH AF Database, picking the cleanest Normal regions we can find.

`normal_1` is left untouched (user reports it shows "Normal" correctly).

Search strategy:
  - For each record in MIT-BIH AF, walk the rhythm annotations.
  - For every region labelled Normal/NSR, extract a 30s @ 128Hz window
    from the middle of the region.
  - Score each candidate with both:
        (1) RR-interval regularity (CV of RR < 0.06 — tighter than the
            old 0.12 threshold).
        (2) The trained RandomForest model. Only accept candidates whose
            RF AFib probability is < 0.20 — well below the model's 0.22
            threshold.
  - Pick the candidate with the LOWEST RF AFib probability for each
    slot.

Output: overwrites samples/normal_2.npy and samples/normal_3.npy only.
"""

import wfdb
import numpy as np
from pathlib import Path
from scipy.signal import resample_poly, find_peaks
from math import gcd
import joblib

# ── Config ──────────────────────────────────────────────────────────────────
AFDB_DIR     = Path("mit-bih-atrial-fibrillation-database-1.0.0/files")
OUTPUT_DIR   = Path("samples")
TARGET_FS    = 128
SEG_SECONDS  = 30
RF_PATH      = "models/rf.pkl"

# Same label sets as extract_segments.py
NORMAL_LABELS = {"(N", "N", "(NSR", "NSR"}
EXCLUDED_RHYTHMS = {
    "(AFL", "AFL", "(VT", "VT", "(SVTA", "SVTA",
    "(B", "B", "(SBR", "SBR", "(T", "T",
    "(IVR", "IVR", "(AB", "AB",
    "MISSB", "PSE", "MB", "M"
}

# Records known to have substantial Normal rhythm
RECORDS = ["04015","04043","04048","04126","04746","04908","05091",
           "05121","06426","06453","04936","07162","08215"]

LEAD_INDEX = 1


# ── Helpers (mirrors extract_segments.py) ───────────────────────────────────

def is_inverted(signal, fs, check_seconds=30):
    n = min(len(signal), int(check_seconds * fs))
    sample = signal[:n]
    pos = np.percentile(sample, 98)
    neg = abs(np.percentile(sample, 2))
    return neg > pos

def load_record(record_id, lead_index=LEAD_INDEX):
    path = str(AFDB_DIR / record_id)
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    n_leads = record.p_signal.shape[1]
    if lead_index >= n_leads:
        lead_index = 0
    signal = record.p_signal[:, lead_index].astype(np.float32)
    fs = int(record.fs)
    if is_inverted(signal, fs):
        signal = -signal
    return signal, fs, ann

def resample_to_128(seg, fs):
    if fs == TARGET_FS:
        return seg.astype(np.float32)
    g = gcd(TARGET_FS, fs)
    return resample_poly(seg, TARGET_FS // g, fs // g).astype(np.float32)

def get_rhythms(ann, signal_len):
    regions = []
    samples = ann.sample
    aux = ann.aux_note
    for i in range(len(samples) - 1):
        regions.append((samples[i], min(samples[i+1], signal_len),
                        aux[i].replace("\x00","").strip()))
    if len(samples) > 0:
        regions.append((samples[-1], signal_len,
                        aux[-1].replace("\x00","").strip()))
    return regions

def is_normal(r):
    return any(x in r for x in NORMAL_LABELS)

def rr_cv(seg, fs):
    sig = seg.astype(np.float32) - np.mean(seg)
    if np.abs(sig.min()) > np.abs(sig.max()):
        sig = -sig
    sig_max = float(np.max(sig)) if np.max(sig) > 0 else 1.0
    thr = max(0.3, sig_max * 0.3)
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.3 * fs))
    if len(peaks) < 5:
        return None, 0
    rr = np.diff(peaks) / fs * 1000
    rr = rr[(rr > 250) & (rr < 2000)]
    if len(rr) < 4:
        return None, len(peaks)
    return float(np.std(rr) / np.mean(rr)), len(peaks)


# ── Feature extraction (mirrors app.py extract_hrv) ────────────────────────
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d

FEATURE_NAMES = [
    "mean_rr","median_rr","sdnn","rmssd","pnn50","cv_rr",
    "mean_hr","std_hr","min_hr","max_hr",
    "mean_diff","max_diff","sd1","sd2","sd_ratio",
    "skewness","kurtosis","iqr","irr_score",
    "lf_hf","lf_norm","hf_norm","dominant_freq","n_beats",
]

def bandpass(signal, fs=TARGET_FS, low=0.5, high=40.0):
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype="band")
    if len(signal) <= 3 * max(len(a), len(b)):
        return signal
    return filtfilt(b, a, signal)

def detect_rpeaks(signal, fs=TARGET_FS):
    sig = bandpass(signal, fs)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    if np.abs(sig.min()) > np.abs(sig.max()):
        sig = -sig
    sig_max = float(np.max(sig))
    thr = max(0.3, sig_max * 0.3)
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.3*fs),
                          prominence=sig_max*0.2)
    if len(peaks) < 3:
        peaks, _ = find_peaks(sig, height=max(0.2, sig_max*0.2),
                              distance=int(0.3*fs))
    return peaks

def extract_hrv(signal, fs=TARGET_FS):
    peaks = detect_rpeaks(signal, fs)
    rr = np.diff(peaks) / fs * 1000
    rr = rr[(rr > 250) & (rr < 2000)]
    if len(rr) < 4:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    mean_rr=float(np.mean(rr)); median_rr=float(np.median(rr))
    sdnn=float(np.std(rr)); rmssd=float(np.sqrt(np.mean(np.diff(rr)**2)))
    pnn50=float(np.mean(np.abs(np.diff(rr))>50))
    cv_rr=sdnn/mean_rr if mean_rr>0 else 0.0
    hr=60000.0/rr
    mean_hr=float(np.mean(hr)); std_hr=float(np.std(hr))
    min_hr=float(np.min(hr)); max_hr=float(np.max(hr))
    diff_rr=np.diff(rr)
    mean_diff=float(np.mean(np.abs(diff_rr)))
    max_diff=float(np.max(np.abs(diff_rr))) if len(diff_rr)>0 else 0.0
    sd1=float(np.sqrt(0.5)*np.std(diff_rr))
    sd2_sq=max(0, 2*sdnn**2 - 0.5*np.std(diff_rr)**2)
    sd2=float(np.sqrt(sd2_sq))
    sd_ratio=sd1/(sd2+1e-8)
    sk=float(skew(rr)); ku=float(kurtosis(rr))
    iqr_val=float(np.percentile(rr,75)-np.percentile(rr,25))
    irr=float(np.sum(np.abs(diff_rr)/(rr[:-1]+1e-9)>0.10)/max(len(diff_rr),1))
    try:
        t_rr=np.cumsum(rr)/1000.0
        t_uni=np.arange(t_rr[0], t_rr[-1], 0.25)
        if len(t_uni)>8:
            rr_uni=interp1d(t_rr, rr, kind="linear",
                             bounds_error=False, fill_value="extrapolate")(t_uni)
            rr_uni-=np.mean(rr_uni)
            freqs, psd = welch(rr_uni, fs=4.0, nperseg=min(len(rr_uni),64))
            pos = freqs > 0; freqs, psd = freqs[pos], psd[pos]
            lf=np.sum(psd[(freqs>=0.04)&(freqs<0.15)])
            hf=np.sum(psd[(freqs>=0.15)&(freqs<0.40)])
            tot=lf+hf+1e-9
            lf_hf=lf/(hf+1e-9); lf_norm=lf/tot; hf_norm=hf/tot
            dom_freq=float(freqs[np.argmax(psd)])
        else:
            lf_hf=lf_norm=hf_norm=dom_freq=0.0
    except Exception:
        lf_hf=lf_norm=hf_norm=dom_freq=0.0
    return np.array([mean_rr,median_rr,sdnn,rmssd,pnn50,cv_rr,
                     mean_hr,std_hr,min_hr,max_hr,
                     mean_diff,max_diff,sd1,sd2,sd_ratio,
                     sk,ku,iqr_val,irr,
                     lf_hf,lf_norm,hf_norm,dom_freq,float(len(peaks))],
                    dtype=np.float32)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading RF model…")
    bundle = joblib.load(RF_PATH)
    clf = bundle["model"]
    imp = bundle.get("imputer")
    thr_rf = float(bundle.get("threshold", 0.5))
    print(f"  RF threshold = {thr_rf}")

    print("\nScanning MIT-BIH AF records for cleanest Normal segments…\n")
    candidates = []  # (score, rec_id, t_start_fs, t_end_fs, prob_afib, cv_rr)

    for rec_id in RECORDS:
        try:
            signal, fs, ann = load_record(rec_id)
        except Exception as e:
            print(f"  {rec_id}: SKIP ({e})")
            continue
        regions = get_rhythms(ann, len(signal))
        needed = SEG_SECONDS * fs

        for start, end, rhythm in regions:
            if not is_normal(rhythm):
                continue
            if (end - start) < needed * 1.5:
                continue

            # Try the middle of the region (cleanest part)
            mid = (start + end) // 2
            s_start = mid - needed // 2
            s_end   = s_start + needed
            if s_start < 0 or s_end > len(signal):
                s_start = start + needed // 4
                s_end   = s_start + needed
            if s_end > len(signal) or s_start < 0:
                continue

            seg = signal[s_start:s_end]
            cv, npeaks = rr_cv(seg, fs)
            if cv is None or cv > 0.06:
                continue

            seg_128 = resample_to_128(seg, fs)
            feats = extract_hrv(seg_128).reshape(1, -1)
            x = imp.transform(feats) if imp is not None else feats
            prob = float(clf.predict_proba(x)[0][1])

            if prob >= 0.20:
                continue  # not confidently Normal under RF

            # Lower CV + lower RF prob = better candidate
            score = prob + cv
            candidates.append((score, rec_id, s_start, s_end, fs,
                               prob, cv, seg_128))
            print(f"  {rec_id} @ {s_start/fs:7.1f}s  "
                  f"cv={cv:.4f}  RF_prob_afib={prob:.4f}  npeaks={npeaks}")

    if not candidates:
        print("\nERROR: No clean Normal candidate found. Aborting.")
        return

    candidates.sort(key=lambda x: x[0])
    print(f"\nFound {len(candidates)} candidates. Picking best 2.\n")

    for slot, idx in enumerate([1, 2]):  # slots 1 and 2 → normal_2, normal_3
        c = candidates[idx]
        _, rec_id, s_start, s_end, fs, prob, cv, seg_128 = c
        out_path = OUTPUT_DIR / f"normal_{slot+2}.npy"
        np.save(out_path, seg_128)
        print(f"  Saved {out_path.name}  record={rec_id}  "
              f"t={s_start/fs:.1f}s–{s_end/fs:.1f}s  "
              f"cv={cv:.4f}  RF_prob_afib={prob:.4f}  "
              f"shape={seg_128.shape}")

    # Sanity check: re-predict all three normals
    print("\nFinal sanity check on the three normals:")
    for name in ["normal_1", "normal_2", "normal_3"]:
        sig = np.load(OUTPUT_DIR / f"{name}.npy")
        feats = extract_hrv(sig).reshape(1, -1)
        x = imp.transform(feats) if imp is not None else feats
        prob = float(clf.predict_proba(x)[0][1])
        verdict = "Normal" if prob < thr_rf else "AFib"
        print(f"  {name}: RF_prob_afib={prob:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
