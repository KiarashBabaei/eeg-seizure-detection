"""
improved_model.py
-----------------
Improved, honest seizure detector built ONLY from final_features.csv
(the same single channel FP1-F7 and the same 3 features you already have).

What changed compared with final_logistic_regression.py, and why:
  1. Log-transform of RMS and line length. EEG amplitudes are heavily skewed;
     on a log scale seizure and non-seizure separate much better.
  2. Variance is dropped. It is almost exactly RMS squared, so it adds no
     information and only makes the coefficients unstable (your negative
     variance coefficient comes from this).
  3. Line-length / RMS ratio: a cheap "how fast is the signal wiggling" feature.
  4. Temporal context: the average of the 5 and 9 neighbouring windows.
     A seizure lasts tens of seconds, a single noisy window does not.
  5. Decision threshold chosen on held-out patients, not the default 0.5.
  6. Light smoothing of the predicted probability over 3 neighbouring windows.
  7. Two models: regularised Logistic Regression and a small, regularised
     gradient-boosted tree model (max_depth=3).

How we guarantee the result is not fake and not overfitting:
  * Every evaluation is on a patient the model has NEVER seen.
  * The threshold is picked using other patients only (nested evaluation).
  * Leave-one-patient-out: each of the 5 patients takes a turn as the test
    patient, so the result does not depend on one lucky split.
  * Train-vs-test gap is printed (a big gap = overfitting).
  * Label-shuffle sanity test: labels are randomly shuffled and the whole
    procedure is repeated. If the pipeline were leaking information it would
    still score well; a real model must collapse to near-chance.

Note: context and smoothing use windows a few seconds BEFORE and AFTER each
window. That is fine for this offline study but would need to be made
past-only for real-time use.

Run from the repository folder:  python scripts/improved_model.py
"""

import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/final_features.csv"
RNG = 42
THRESHOLDS = np.linspace(0.05, 0.99, 95)


# ---------------------------------------------------------------- features
def build_features(df):
    df = df.sort_values(["recording_id", "start_time"]).reset_index(drop=True)
    df["log_rms"] = np.log(df["rms"])
    df["log_line_length"] = np.log(df["line_length"])
    df["ll_over_rms"] = df["log_line_length"] - df["log_rms"]
    for col in ["log_rms", "log_line_length"]:
        for k in (5, 9):
            df[f"{col}_ctx{k}"] = (df.groupby("recording_id")[col]
                                     .transform(lambda s: s.rolling(k, center=True, min_periods=1).mean()))
    return df


FEATURES = ["log_rms", "log_line_length", "ll_over_rms",
            "log_rms_ctx5", "log_line_length_ctx5",
            "log_rms_ctx9", "log_line_length_ctx9"]

MODELS = {
    "Logistic Regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", C=0.1,
                                  max_iter=2000, random_state=RNG))]),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
        class_weight="balanced", random_state=RNG),
}


def smooth(prob, recording_ids, k=3):
    return (pd.Series(prob).groupby(np.asarray(recording_ids))
              .transform(lambda s: s.rolling(k, center=True, min_periods=1).mean())
              .values)


def predict(model, part):
    return smooth(model.predict_proba(part[FEATURES])[:, 1], part["recording_id"])


def best_threshold(y, p):
    scores = [f1_score(y, p >= t, zero_division=0) for t in THRESHOLDS]
    return THRESHOLDS[int(np.argmax(scores))]


def scores(y, p, t):
    yhat = p >= t
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p), Precision=precision_score(y, yhat, zero_division=0),
                Recall=recall_score(y, yhat, zero_division=0), F1=f1_score(y, yhat, zero_division=0),
                TP=tp, FP=fp, FN=fn, TN=tn, threshold=t)


# ---------------------------------------------------------- 1. original split
def original_split(df, make_model):
    """Same split as your project: train chb01/03/05, validate chb07, test chb08."""
    tr, va, te = (df[df.split == s] for s in ("train", "validation", "test"))
    m = make_model().fit(tr[FEATURES], tr.label)
    t = best_threshold(va.label, predict(m, va))      # threshold from validation only
    return scores(te.label, predict(m, te), t), roc_auc_score(tr.label, predict(m, tr))


# ------------------------------------------- 2. leave-one-patient-out (nested)
def lopo(df, make_model, label_col="label"):
    rows = []
    patients = sorted(df.patient_id.unique())
    for test_p in patients:
        outer = df[df.patient_id != test_p]
        test = df[df.patient_id == test_p]
        # inner loop: out-of-patient probabilities on the training patients -> threshold
        oof_y, oof_p = [], []
        for inner_p in sorted(outer.patient_id.unique()):
            a, b = outer[outer.patient_id != inner_p], outer[outer.patient_id == inner_p]
            m = make_model().fit(a[FEATURES], a[label_col])
            oof_y.append(b[label_col].values); oof_p.append(predict(m, b))
        t = best_threshold(np.concatenate(oof_y), np.concatenate(oof_p))
        m = make_model().fit(outer[FEATURES], outer[label_col])
        r = scores(test[label_col], predict(m, test), t)
        r["train_AUC"] = roc_auc_score(outer[label_col], predict(m, outer))
        r["patient"] = test_p
        rows.append(r)
    return pd.DataFrame(rows).set_index("patient")


def main():
    df = build_features(pd.read_csv(CSV))
    pd.set_option("display.width", 140)

    print("=" * 70)
    print("A) ORIGINAL SPLIT (train chb01/03/05, threshold on chb07, test chb08)")
    print("   Your previous Logistic Regression: P=0.126  R=0.775  F1=0.217")
    print("=" * 70)
    for name, mk in MODELS.items():
        r, train_auc = original_split(df, mk)
        print(f"{name:20s} P={r['Precision']:.3f}  R={r['Recall']:.3f}  F1={r['F1']:.3f}  "
              f"AUC={r['AUC']:.3f}  (train AUC {train_auc:.3f})  "
              f"TP={r['TP']} FP={r['FP']} FN={r['FN']}")

    print("\n" + "=" * 70)
    print("B) LEAVE-ONE-PATIENT-OUT (each patient is the unseen test patient once)")
    print("=" * 70)
    summary = []
    # your original model, same protocol, for a fair comparison
    orig_rows = []
    for p in sorted(df.patient_id.unique()):
        tr, te = df[df.patient_id != p], df[df.patient_id == p]
        cols = ["variance", "rms", "line_length"]
        m = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RNG))]).fit(tr[cols], tr.label)
        r = scores(te.label, m.predict_proba(te[cols])[:, 1], 0.5)
        r["patient"] = p
        orig_rows.append(r)
    orig = pd.DataFrame(orig_rows).set_index("patient")
    print("\nYOUR ORIGINAL MODEL (3 raw features, threshold 0.5)")
    print(orig[["AUC", "Precision", "Recall", "F1", "TP", "FP", "FN"]].round(3))
    print(f"MEAN  AUC={orig.AUC.mean():.3f}  F1={orig.F1.mean():.3f}")
    summary.append(dict(model="Original LR", mean_AUC=orig.AUC.mean(), mean_F1=orig.F1.mean(),
                        mean_precision=orig.Precision.mean(), mean_recall=orig.Recall.mean()))
    for name, mk in MODELS.items():
        res = lopo(df, mk)
        print(f"\n{name}")
        print(res[["AUC", "Precision", "Recall", "F1", "TP", "FP", "FN", "train_AUC"]].round(3))
        print(f"MEAN  AUC={res.AUC.mean():.3f}  F1={res.F1.mean():.3f}  "
              f"(train AUC {res.train_AUC.mean():.3f})")
        summary.append(dict(model=name, mean_AUC=res.AUC.mean(), mean_F1=res.F1.mean(),
                            mean_precision=res.Precision.mean(), mean_recall=res.Recall.mean()))
        res.round(4).to_csv(f"results/lopo_{name.replace(' ', '_').lower()}.csv")

    print("\n" + "=" * 70)
    print("C) SANITY CHECK: shuffled labels (should collapse to AUC ~0.5)")
    print("=" * 70)
    rng = np.random.default_rng(RNG)
    shuffled = df.copy()
    shuffled["shuffled"] = shuffled.groupby("patient_id")["label"].transform(
        lambda s: rng.permutation(s.values))
    for name, mk in MODELS.items():
        res = lopo(shuffled, mk, label_col="shuffled")
        print(f"{name:20s} mean AUC with random labels = {res.AUC.mean():.3f}")

    pd.DataFrame(summary).round(4).to_csv("results/improved_summary.csv", index=False)
    print("\nSaved: results/improved_summary.csv, results/lopo_*.csv")


if __name__ == "__main__":
    main()
