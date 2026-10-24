"""Technical Validation #11 - Controls for the decoding benchmark.

Small, targeted controls that bound how much of the within-dataset TS+LR
accuracy of 10_classification_benchmark.py could come from something other
than sensorimotor activity. Same data preparation, band, epoch selection and
classifier as script 10 (see its docstring); only the factor named below is
changed in each control. Evoked responses are not subtracted in either script.

Conditions (movement vs rest, leave-one-run-out, TS+LR, per participant and limb):
  - standard: script 10 as published (movement 0.5 to 2.5 s, rest -2.0 to 0.0 s,
    all 32 channels, ~18 Hz line notched). Recomputed here as the reference; it
    must reproduce 10_classification_results.csv.
  - late window: movement 1.0 to 2.5 s, rest -2.0 to -0.5 s (1.5 s each). Moves
    the movement window further from the cue-evoked response and the rest
    window away from the cue onset, where a zero-phase filter can leak the
    evoked response backwards.
  - occipital only: O1, Oz, O2 and POz. Estimates what is decodable from the
    visual sites alone (cue identity in Dataset_A; no visual cue in B/C).
  - no notch: bad channels interpolated but the ~18 Hz line left in.
  - matched trial count (Dataset_A only): 3 of the 4 runs and 10 movement cues
    per run (5 left + 5 right where available) with their paired rest epochs,
    i.e. 60 epochs and 3 folds as in Dataset_B/C; mean over N_DRAWS random draws.

Usage:
    py -3.12 11_classification_controls.py [--subjects sub-01 ...] [--n-jobs 1]

Writes:
    derivatives/validation/11_classification_controls_results.csv  (participant x condition)
    derivatives/validation/11_classification_controls_summary.csv  (group level, paired vs standard)
    derivatives/validation/11_classification_controls_cache/<sub>.csv (per-participant cache;
        delete it to recompute after changing the pipeline)
"""
import argparse
import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import cues_only
from common import (BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS,
                    clean_raw, recording_bads)

mne.set_log_level("ERROR")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]
OCCIPITAL = ["O1", "Oz", "O2", "POz"]
ALL = list(range(len(EEG_CHANNELS)))
OCC = [EEG_CHANNELS.index(c) for c in OCCIPITAL]
BAND = (8.0, 30.0)
DECIM = 5  # 500 Hz -> 100 Hz
WINDOWS = {"standard": ((0.5, 2.5), (-2.0, 0.0)), "late": ((1.0, 2.5), (-2.0, -0.5))}
MIN_FOLLOW_S = 2.5
LIMB = {"lower": "foot", "upper": "hand"}
N_DRAWS = 20
MATCH_RUNS, MATCH_CUES = 3, 10  # Dataset_B/C design
CACHE_DIR = OUT_DIR / "11_classification_controls_cache"


def interpolate_only(raw, ds, sub_id, task, run):
    """Step 1 of common.clean_raw without the ~18 Hz notch."""
    bads = recording_bads(ds, sub_id, task, run)
    if bads:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                        match_case=False, on_missing="raise")
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw


def cut(data, onsets, sfreq, move_win, rest_win):
    """Epochs as in script 10: (X, y (1=movement, 0=rest), index of the cue)."""
    n_win = int(round((move_win[1] - move_win[0]) * sfreq))
    X, y, cue = [], [], []
    for i, onset in enumerate(onsets):
        starts = [int(round((onset + t0) * sfreq)) for t0 in (move_win[0], rest_win[0])]
        if min(starts) < 0 or max(starts) + n_win > data.shape[1]:
            continue  # keep only complete movement/rest pairs
        for label, a in zip((1, 0), starts):
            X.append(data[:, a:a + n_win:DECIM])
            y.append(label)
            cue.append(i)
    return np.array(X), np.array(y), np.array(cue)


def load_recording(ds, sub_id, task, run):
    """Return {condition: (X, y, cue index)} and the side of each selected cue."""
    stem = f"{sub_id}_task-{task}_run-{run:02d}"
    eeg_dir = BIDS_ROOT / ds / sub_id / "eeg"
    raw = mne.io.read_raw_brainvision(eeg_dir / f"{stem}_eeg.vhdr", preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    sfreq = raw.info["sfreq"]

    ev = cues_only(pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t"))
    next_onset = ev["onset"].shift(-1)
    prev_type = ev["trial_type"].shift(1)
    ok = (ev.trial_type.isin(["left", "right"])
          & ((next_onset - ev.onset) >= MIN_FOLLOW_S)
          & (prev_type == "rest"))
    onsets = ev.loc[ok, "onset"].to_numpy()
    side = ev.loc[ok, "trial_type"].to_numpy()

    no_notch = interpolate_only(raw.copy(), ds, sub_id, task, run).filter(*BAND, verbose=False).get_data()
    clean = clean_raw(raw, ds, sub_id, task, run).filter(*BAND, verbose=False).get_data()
    out = {name: cut(clean, onsets, sfreq, *wins) for name, wins in WINDOWS.items()}
    out["no notch"] = cut(no_notch, onsets, sfreq, *WINDOWS["standard"])
    return out, side


def loro(C, y, g):
    """Leave-one-run-out TS+LR on precomputed covariances: (accuracy, AUC)."""
    s = np.empty(len(y))
    for test_run in np.unique(g):
        tr, te = g != test_run, g == test_run
        ts = TangentSpace(metric="riemann").fit(C[tr])
        clf = LogisticRegression(max_iter=1000).fit(ts.transform(C[tr]), y[tr])
        s[te] = clf.predict_proba(ts.transform(C[te]))[:, 1]
    return balanced_accuracy_score(y, (s > 0.5).astype(int)), roc_auc_score(y, s)


def matched_draw(y, g, cue, side, rng):
    """Indices of one Dataset_B/C-sized subsample of a Dataset_A limb."""
    keep = []
    for run in rng.choice(np.unique(g), MATCH_RUNS, replace=False):
        cues = np.unique(cue[g == run])
        chosen = []
        for s in ("left", "right"):
            pool = [c for c in cues if side[run][c] == s]
            chosen += list(rng.permutation(pool)[:MATCH_CUES // 2])
        rest = [c for c in rng.permutation(cues) if c not in chosen]
        chosen += rest[:MATCH_CUES - len(chosen)]
        keep.append(np.flatnonzero((g == run) & np.isin(cue, chosen)))
    return np.concatenate(keep)


def run_subject(sub_id):
    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")  # also inside joblib workers
    rng = np.random.default_rng(int(sub_id[-2:]))
    rows = []

    def add(ds, task, condition, n, acc, auc):
        rows.append(dict(subject=sub_id, dataset=ds, limb=LIMB[task], condition=condition,
                         n_epochs=n, accuracy=acc, auc=auc))

    for ds, n_runs in DATASETS.items():
        for task in TASKS:
            parts, sides = {}, {}
            for run in range(1, n_runs + 1):
                rec, sides[run] = load_recording(ds, sub_id, task, run)
                for name, (X, y, cue) in rec.items():
                    parts.setdefault(name, []).append((X, y, np.full(len(y), run), cue))
            data = {name: [np.concatenate(z) for z in zip(*v)] for name, v in parts.items()}

            X, y, g, cue = data["standard"]
            C = Covariances("oas").fit_transform(X)
            add(ds, task, "standard", len(y), *loro(C, y, g))
            add(ds, task, "occipital only", len(y), *loro(Covariances("oas").fit_transform(X[:, OCC]), y, g))
            for name in ("late", "no notch"):
                Xc, yc, gc, _ = data[name]
                add(ds, task, name if name == "no notch" else "late window", len(yc),
                    *loro(Covariances("oas").fit_transform(Xc), yc, gc))
            if ds == "Dataset_A":
                draws = []
                for _ in range(N_DRAWS):
                    idx = matched_draw(y, g, cue, sides, rng)
                    draws.append((len(idx), *loro(C[idx], y[idx], g[idx])))
                n, acc, auc = np.mean(draws, axis=0)
                add(ds, task, "matched trial count", int(round(n)), acc, auc)
    print(f"{sub_id} done", flush=True)
    return rows


def run_subject_cached(sub_id):
    """Per-participant results are cached, so an interrupted run resumes."""
    path = CACHE_DIR / f"{sub_id}.csv"
    if not path.exists():
        pd.DataFrame(run_subject(sub_id)).to_csv(path, index=False)
    return pd.read_csv(path).to_dict("records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    suffix = "_partial" if args.subjects else ""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = Parallel(n_jobs=args.n_jobs)(delayed(run_subject_cached)(s) for s in subjects)
    res = pd.DataFrame([r for rows in all_rows for r in rows])
    res.to_csv(OUT_DIR / f"11_classification_controls_results{suffix}.csv", index=False)

    summ = []
    for (ds, limb, cond), d in res.groupby(["dataset", "limb", "condition"], sort=False):
        a = d.set_index("subject").accuracy
        b = res.query("dataset == @ds and limb == @limb and condition == 'standard'").set_index("subject").accuracy
        a, b = a.align(b, join="inner")
        diff = a - b
        testable = len(a) > 5
        summ.append(dict(
            dataset=ds, limb=limb, condition=cond, n_subjects=len(a), n_epochs=d.n_epochs.mean(),
            accuracy_mean=a.mean(), accuracy_sd=a.std(), auc_mean=d.auc.mean(),
            p_vs_chance=wilcoxon(a - 0.5).pvalue if testable else np.nan,
            difference_vs_standard=diff.mean(),
            p_vs_standard=wilcoxon(diff).pvalue if testable and (diff != 0).any() else np.nan))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT_DIR / f"11_classification_controls_summary{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    show = summ.copy()
    for col in ("p_vs_chance", "p_vs_standard"):
        show[col] = show[col].map(lambda v: f"{v:.2g}")
    print("\n" + show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
