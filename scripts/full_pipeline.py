"""
full_pipeline.py
----------------
The bigger upgrade. Needs your raw EDF files (the data/ folder you already used).

Why this should help much more than tuning the model:
  Your project used ONE channel (FP1-F7) and three amplitude features.
  Seizures often start in other parts of the head and change the RHYTHM of
  the EEG (frequency), not only its size. Here we:
    * use all 18 standard bipolar channels shared by these recordings,
    * band-pass filter 0.5-40 Hz (removes slow drift and 60 Hz mains noise),
    * per channel compute: log line length, log power, relative power in the
      delta / theta / alpha / beta / gamma bands, and spectral entropy,
    * summarise each feature across channels with median, max and spread.
      This keeps the model small (24 features) and works the same for every
      patient, no matter where the seizure starts. Small = less overfitting.
    * add the same neighbouring-window context as improved_model.py.

Evaluation is IDENTICAL to improved_model.py (unseen-patient testing,
nested threshold choice, leave-one-patient-out, shuffled-label check),
plus seizure-level results: how many seizures were caught and how many
false alarms per hour.

Run from the repository folder (the data/ folder must be there too):
    pip install mne scikit-learn pandas numpy scipy
    python scripts/full_pipeline.py

Adding more patients to RECORDINGS (copy seizure times from each patient's
chbXX-summary.txt) is the single most useful thing you can do next.
"""

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

import improved_model as im
from final_subset_metadata import recordings as RECORDINGS   # your own list

mne.set_log_level("ERROR")

WINDOW_SECONDS = 4
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def pick_channels(raw):
    """CHB-MIT sometimes has duplicates like 'T8-P8-0'; take the first match of each name."""
    chosen = []
    for want in CHANNELS:
        hits = [c for c in raw.ch_names if c.upper() == want or c.upper().startswith(want + "-")]
        if not hits:
            raise ValueError(f"{want} missing in {raw.filenames[0]}")
        chosen.append(hits[0])
    return chosen


def window_features(x, fs):
    """x: channels x samples. Returns dict of channel-summarised features."""
    per_ch = {}
    per_ch["log_line_length"] = np.log(np.sum(np.abs(np.diff(x, axis=1)), axis=1) + 1e-12)
    per_ch["log_power"] = np.log(np.mean(x ** 2, axis=1) + 1e-20)
    f, pxx = welch(x, fs=fs, nperseg=fs * 2, axis=1)
    keep = (f >= 0.5) & (f <= 40)
    total = pxx[:, keep].sum(axis=1) + 1e-30
    for name, (lo, hi) in BANDS.items():
        band = (f >= lo) & (f < hi)
        per_ch[f"rel_{name}"] = pxx[:, band].sum(axis=1) / total
    p = pxx[:, keep] / total[:, None]
    per_ch["spectral_entropy"] = -(p * np.log(p + 1e-30)).sum(axis=1) / np.log(keep.sum())

    out = {}
    for name, v in per_ch.items():
        out[f"{name}_median"] = np.median(v)
        out[f"{name}_max"] = np.max(v)
        out[f"{name}_std"] = np.std(v)
    return out


def extract():
    rows = []
    for rec in RECORDINGS:
        print("reading", rec["file"])
        raw = mne.io.read_raw_edf(rec["file"], preload=True)
        raw.pick(pick_channels(raw))
        raw.filter(0.5, 40.0, verbose="ERROR")
        fs = int(raw.info["sfreq"])
        data = raw.get_data()
        n_sec = data.shape[1] // fs
        s0, s1 = rec["seizure_start"], rec["seizure_end"]
        for start in range(0, n_sec - WINDOW_SECONDS + 1, WINDOW_SECONDS):
            end = start + WINDOW_SECONDS
            if end <= s0 or start >= s1:
                label = 0
            elif start >= s0 and end <= s1:
                label = 1
            else:
                continue                      # same rule as your project
            feats = window_features(data[:, start * fs:end * fs], fs)
            rows.append(dict(patient_id=rec["patient_id"], recording_id=rec["recording_id"],
                             split=rec["split"], start_time=start, end_time=end,
                             label=label, **feats))
    df = pd.DataFrame(rows)
    df.to_csv("results/full_features.csv", index=False)
    return df


def add_context(df, base_cols):
    df = df.sort_values(["recording_id", "start_time"]).reset_index(drop=True)
    for c in base_cols:
        df[f"{c}_ctx5"] = (df.groupby("recording_id")[c]
                             .transform(lambda s: s.rolling(5, center=True, min_periods=1).mean()))
    return df


def event_level(test, prob, threshold):
    """Seizure caught = at least 2 consecutive flagged windows inside it.
    False alarm = a run of >=2 flagged windows outside any seizure."""
    test = test.assign(flag=(prob >= threshold).astype(int))
    caught, alarms, hours = 0, 0, 0.0
    for _, r in test.groupby("recording_id"):
        r = r.sort_values("start_time")
        hours += len(r) * WINDOW_SECONDS / 3600
        run = r.flag.rolling(2).sum().fillna(0) >= 2
        if (run & (r.label == 1)).any():
            caught += 1
        starts = run & ~run.shift(1, fill_value=False)
        alarms += int((starts & (r.label == 0)).sum())
    return caught, test.recording_id.nunique(), alarms / max(hours, 1e-9)


def main():
    try:
        df = pd.read_csv("results/full_features.csv")
        print("using existing results/full_features.csv (delete it to re-extract)")
    except FileNotFoundError:
        df = extract()

    base = [c for c in df.columns if c.endswith(("_median", "_max", "_std"))]
    ctx_base = ["log_line_length_median", "log_line_length_max", "log_power_median",
                "rel_delta_median", "rel_theta_median", "spectral_entropy_median"]
    df = add_context(df, ctx_base)
    im.FEATURES = base + [f"{c}_ctx5" for c in ctx_base]
    print(f"{len(im.FEATURES)} features, {len(df)} windows")

    print("\nA) ORIGINAL SPLIT (test chb08)  - previous best F1 = 0.217")
    for name, mk in im.MODELS.items():
        r, train_auc = im.original_split(df, mk)
        te, va, tr = (df[df.split == s] for s in ("test", "validation", "train"))
        m = mk().fit(tr[im.FEATURES], tr.label)
        caught, n, fa = event_level(te, im.predict(m, te), r["threshold"])
        print(f"{name:20s} P={r['Precision']:.3f} R={r['Recall']:.3f} F1={r['F1']:.3f} "
              f"AUC={r['AUC']:.3f} (train AUC {train_auc:.3f}) | seizures caught {caught}/{n}, "
              f"false alarms/hour {fa:.2f}")

    print("\nB) LEAVE-ONE-PATIENT-OUT")
    for name, mk in im.MODELS.items():
        res = im.lopo(df, mk)
        print(f"\n{name}\n", res[["AUC", "Precision", "Recall", "F1", "TP", "FP", "FN", "train_AUC"]].round(3))
        print(f"MEAN AUC={res.AUC.mean():.3f} F1={res.F1.mean():.3f}")
        res.round(4).to_csv(f"results/full_lopo_{name.replace(' ', '_').lower()}.csv")

    print("\nC) SHUFFLED-LABEL CHECK (should be ~0.5)")
    rng = np.random.default_rng(im.RNG)
    df["shuffled"] = df.groupby("patient_id")["label"].transform(lambda s: rng.permutation(s.values))
    for name, mk in im.MODELS.items():
        print(f"{name:20s} mean AUC = {im.lopo(df, mk, label_col='shuffled').AUC.mean():.3f}")


if __name__ == "__main__":
    main()
