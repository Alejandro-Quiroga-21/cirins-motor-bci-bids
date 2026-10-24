"""Technical Validation #10 - Baseline decoding benchmark, with controls.

Shows that the data support their intended use - detecting executed movements
for a rehabilitation BCI - with standard, reproducible pipelines. It is a
baseline, not an attempt to maximise accuracy. The analysis windows and band
are standard choices that were fixed before looking at any decoding result
(they match the ERD/ERS windows of 05_erd_ers_check.py); nothing was tuned on
the accuracies.

Data: the same preparation as the neurophysiological checks (common.clean_raw:
bad channels interpolated, ~18 Hz line notched out), native reference,
band-pass 8-30 Hz (mu + beta, zero-phase FIR), decimated to 100 Hz.

Epochs (2 s each, equal length), per movement cue preceded by a rest cue and
followed by at least 2.5 s before the next event (drops the few spurious short
re-triggers, see flagged_events.tsv):
  - movement: 0.5 to 2.5 s after the movement cue (skips the cue-evoked
    VEP/AEP);
  - rest: the 2 s preceding the same movement cue (-2.0 to 0.0 s), i.e. the end
    of the rest interval - away from the previous trial's post-movement beta
    rebound and, in Dataset_C, from the mirror-animation feedback - giving
    balanced classes. Note: with a zero-phase filter, up to ~0.8 s of the
    cue-evoked response can leak backwards into the end of the rest window; the
    sensorimotor-channel control (below) excludes the occipital/frontal sites
    where the evoked responses are largest.

Pipelines:
  - CSP+LDA: 6 CSP components (Ledoit-Wolf regularised; rank set to the number
    of channels minus the interpolated ones), log-variance, LDA;
  - TS+LR: OAS covariance, Riemannian tangent space, logistic regression.

Analyses (per participant, one limb at a time):
  1. within: movement vs rest, leave-one-run-out cross-validation (4 folds in
     Dataset_A, 3 in Dataset_B/C), out-of-fold predictions pooled. All fitting
     (CSP, tangent-space reference, classifier) uses training runs only.
  2. transfer: trained on all Dataset_A epochs of the limb (left and right
     movements pooled as "movement"), tested on Dataset_B and on Dataset_C
     (dominant-limb movements) - a BCI calibrated once and then used in open-
     and closed-loop operation within the same session (same cap placement), so
     this is cross-task, not cross-session, transfer. Variants:
       - TS+LR / CSP+LDA without adaptation, tested on all runs;
       - TS+LR re-centred, transductive: each dataset's covariances re-centred
         to the identity by their own Riemannian mean (unsupervised; Zanini et
         al., 2018), the test mean estimated from all test epochs;
       - TS+LR re-centred, online-feasible: the test mean estimated from the
         unlabeled epochs of run 1 only, tested on runs 2-3 (compared with the
         within-dataset and unadapted results on the same runs 2-3).

Controls:
  - sensorimotor channels only (FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2,
    CP6): repeats the within and transfer analyses without the frontal
    (ocular), occipital (visual) and temporal (muscle-prone) sites, to rule out
    that decoding relies on differences in visual stimulation between the
    movement and rest cues (Dataset_A) or on artifacts;
  - permutation test (TS+LR, within, both channel sets): the movement/rest labels of
    every (movement, rest) epoch pair are swapped with probability 0.5, N_PERM times
    (preserving the pairs, the run structure and the class balance); in each
    permutation the classifier is refitted on the permuted training labels of
    every leave-one-run-out fold and scored against the permuted labels, so the
    null distribution is that of the same statistic on the permuted data set; p =
    (1 + #null >= observed) / (1 + N_PERM). Covariances and tangent-space
    projections are label-independent, so they are computed once per fold and
    only the classifier is refitted for each permutation. The random generator
    is seeded with the participant number, so the permutations are reproducible.

Metrics: balanced accuracy and ROC AUC. Chance: permutation p per participant
(within analyses); binomial 95% threshold for the number of test epochs
(Combrisson & Jerbi, 2015) as a second, less conservative criterion; group
level, two-sided Wilcoxon signed-rank test of accuracy against 0.5.

Usage:
    py -3.12 10_classification_benchmark.py [--subjects sub-01 ...] [--n-jobs 2] [--n-perm 1000]

Writes:
    derivatives/validation/10_classification_results.csv     (participant x condition)
    derivatives/validation/10_classification_summary.csv
    derivatives/validation/10_classification_comparisons.csv (paired comparisons)
    derivatives/validation/10_classification_cache/<sub>.csv (per-participant cache;
        delete it to recompute after changing the pipeline)
"""
import argparse
import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import fractional_matrix_power
from scipy.stats import binom, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline

from common import cues_only
from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS, clean_raw, recording_bads

mne.set_log_level("ERROR")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]
SENSORIMOTOR = ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"]
CHANNEL_SETS = {"all 32": list(range(len(EEG_CHANNELS))),
                "sensorimotor": [EEG_CHANNELS.index(c) for c in SENSORIMOTOR]}
BAND = (8.0, 30.0)
DECIM = 5  # 500 Hz -> 100 Hz
MOVE_WIN = (0.5, 2.5)
REST_WIN = (-2.0, 0.0)
MIN_FOLLOW_S = 2.5
LIMB = {"lower": "foot", "upper": "hand"}
CACHE_DIR = OUT_DIR / "10_classification_cache"


def load_epochs(ds, sub_id, task, run):
    """Return (X (n, 32, n_times), y (1=movement, 0=rest)) for one recording."""
    stem = f"{sub_id}_task-{task}_run-{run:02d}"
    eeg_dir = BIDS_ROOT / ds / sub_id / "eeg"
    raw = mne.io.read_raw_brainvision(eeg_dir / f"{stem}_eeg.vhdr", preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    clean_raw(raw, ds, sub_id, task, run)
    raw.filter(*BAND, verbose=False)
    sfreq = raw.info["sfreq"]

    ev = cues_only(pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t"))
    next_onset = ev["onset"].shift(-1)
    prev_type = ev["trial_type"].shift(1)
    ok = (ev.trial_type.isin(["left", "right"])
          & ((next_onset - ev.onset) >= MIN_FOLLOW_S)
          & (prev_type == "rest"))
    onsets = ev.loc[ok, "onset"].to_numpy()

    data = raw.get_data()
    n_win = int(round((MOVE_WIN[1] - MOVE_WIN[0]) * sfreq))
    assert n_win == int(round((REST_WIN[1] - REST_WIN[0]) * sfreq))
    X, y = [], []
    for onset in onsets:
        starts = [int(round((onset + t0) * sfreq)) for t0 in (MOVE_WIN[0], REST_WIN[0])]
        if min(starts) < 0 or max(starts) + n_win > data.shape[1]:
            continue  # keep only complete movement/rest pairs
        for label, a in zip((1, 0), starts):
            X.append(data[:, a:a + n_win:DECIM])
            y.append(label)
    return np.array(X), np.array(y)


def csp_lda(rank):
    """rank is given explicitly: after interpolation of bad channels the 32-channel
    data are rank-deficient, and letting MNE estimate the rank numerically made the
    CSP result of those participants depend on the computing environment."""
    return make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True, rank={"eeg": rank}),
                         LinearDiscriminantAnalysis())


def lr():
    return LogisticRegression(max_iter=1000)


def recentre(C, ref):
    """Re-centre covariances C to the identity using the Riemannian mean of ref."""
    isq = fractional_matrix_power(mean_riemann(ref), -0.5).real
    return np.array([isq @ c @ isq for c in C])


def metrics(y, score, threshold):
    return balanced_accuracy_score(y, (score > threshold).astype(int)), roc_auc_score(y, score)


def within_ts_lr(C, y, g, n_perm, rng):
    """Leave-one-run-out TS+LR on precomputed covariances, with a pairwise
    label-permutation null. Returns (oof probabilities, null accuracies).

    Epochs come in (movement, rest) pairs taken around the same cue (load_epochs),
    so the two epochs of a pair are not independent. The null keeps that
    structure: in each permutation, the two labels of every pair are swapped
    with probability 0.5, which preserves the pairs, the run structure and the
    class balance of every run."""
    assert (y[0::2] == 1).all() and (y[1::2] == 0).all() and (g[0::2] == g[1::2]).all()
    folds = []
    for test_run in np.unique(g):
        tr, te = g != test_run, g == test_run
        ts = TangentSpace(metric="riemann").fit(C[tr])  # label-independent
        folds.append((tr, te, ts.transform(C[tr]), ts.transform(C[te])))

    def oof(labels):
        s = np.empty(len(labels))
        for tr, te, Ztr, Zte in folds:
            s[te] = lr().fit(Ztr, labels[tr]).predict_proba(Zte)[:, 1]
        return s

    obs = oof(y)
    null = []
    for _ in range(n_perm):
        swap = rng.random(len(y) // 2) < 0.5
        yp = y.copy()
        yp[0::2][swap], yp[1::2][swap] = 0, 1
        # the statistic of the permuted data set: refitted on, and scored against, yp
        null.append(balanced_accuracy_score(yp, (oof(yp) > 0.5).astype(int)))
    return obs, np.array(null)


def run_subject(sub_id, n_perm):
    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")  # also inside joblib workers
    rng = np.random.default_rng(int(sub_id[-2:]))  # seed = participant number
    n_bad = len(recording_bads("Dataset_A", sub_id, "lower", 1))  # per participant
    data = {}
    for ds, n_runs in DATASETS.items():
        for task in TASKS:
            Xs, ys, gs = [], [], []
            for run in range(1, n_runs + 1):
                X, y = load_epochs(ds, sub_id, task, run)
                Xs.append(X), ys.append(y), gs.append(np.full(len(y), run))
            data[(ds, task)] = (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs))

    rows = []

    def add(**kw):
        rows.append(dict(subject=sub_id, **kw))

    for chset, ch in CHANNEL_SETS.items():
        # interpolated channels are combinations of the other channels: they reduce
        # the rank of the full montage only (the 11-channel subset stays full rank)
        rank = len(ch) - n_bad if chset == "all 32" else len(ch)
        cov = {k: Covariances("oas").fit_transform(X[:, ch]) for k, (X, _, _) in data.items()}

        # 1. within dataset
        for (ds, task), (X, y, g) in data.items():
            s = cross_val_predict(csp_lda(rank), X[:, ch], y, groups=g, cv=LeaveOneGroupOut(),
                                  method="decision_function")
            acc, auc = metrics(y, s, 0.0)
            add(analysis="within", channels=chset, train=ds, test=ds, test_runs="all",
                limb=LIMB[task], pipeline="CSP+LDA", n_test=len(y), accuracy=acc, auc=auc)

            s, null = within_ts_lr(cov[(ds, task)], y, g, n_perm, rng)
            acc, auc = metrics(y, s, 0.5)
            perm = dict(p_perm=(1 + (null >= acc).sum()) / (1 + len(null)),
                        perm_threshold=np.quantile(null, 0.95)) if len(null) else {}
            add(analysis="within", channels=chset, train=ds, test=ds, test_runs="all",
                limb=LIMB[task], pipeline="TS+LR", n_test=len(y), accuracy=acc, auc=auc, **perm)
            if ds != "Dataset_A":  # same test runs as the online-feasible transfer
                m = g >= 2
                acc, auc = metrics(y[m], s[m], 0.5)
                add(analysis="within", channels=chset, train=ds, test=ds, test_runs="2-3",
                    limb=LIMB[task], pipeline="TS+LR", n_test=int(m.sum()), accuracy=acc, auc=auc)

        # 2. transfer Dataset_A -> Dataset_B / Dataset_C
        for task in TASKS:
            Xa, ya, _ = data[("Dataset_A", task)]
            Ca = cov[("Dataset_A", task)]
            csp = csp_lda(rank).fit(Xa[:, ch], ya)
            ts = TangentSpace(metric="riemann").fit(Ca)
            clf = lr().fit(ts.transform(Ca), ya)
            Ca_rc = recentre(Ca, Ca)
            ts_rc = TangentSpace(metric="riemann").fit(Ca_rc)
            clf_rc = lr().fit(ts_rc.transform(Ca_rc), ya)
            for test in ("Dataset_B", "Dataset_C"):
                Xt, yt, gt = data[(test, task)]
                Ct = cov[(test, task)]
                base = dict(analysis="transfer", channels=chset, train="Dataset_A", test=test,
                            limb=LIMB[task])
                acc, auc = metrics(yt, csp.decision_function(Xt[:, ch]), 0.0)
                add(**base, test_runs="all", pipeline="CSP+LDA", n_test=len(yt), accuracy=acc, auc=auc)
                s = clf.predict_proba(ts.transform(Ct))[:, 1]
                acc, auc = metrics(yt, s, 0.5)
                add(**base, test_runs="all", pipeline="TS+LR", n_test=len(yt), accuracy=acc, auc=auc)
                m = gt >= 2
                acc, auc = metrics(yt[m], s[m], 0.5)
                add(**base, test_runs="2-3", pipeline="TS+LR", n_test=int(m.sum()), accuracy=acc, auc=auc)
                s = clf_rc.predict_proba(ts_rc.transform(recentre(Ct, Ct)))[:, 1]
                acc, auc = metrics(yt, s, 0.5)
                add(**base, test_runs="all", pipeline="TS+LR recentred (transductive)",
                    n_test=len(yt), accuracy=acc, auc=auc)
                s = clf_rc.predict_proba(ts_rc.transform(recentre(Ct[m], Ct[gt == 1])))[:, 1]
                acc, auc = metrics(yt[m], s, 0.5)
                add(**base, test_runs="2-3", pipeline="TS+LR recentred (run-1 estimate)",
                    n_test=int(m.sum()), accuracy=acc, auc=auc)
    print(f"{sub_id} done", flush=True)
    return rows


def run_subject_cached(sub_id, n_perm):
    """Per-participant results are cached, so an interrupted run resumes."""
    path = CACHE_DIR / f"{sub_id}.csv"
    if not path.exists():
        pd.DataFrame(run_subject(sub_id, n_perm)).to_csv(path, index=False)
    return pd.read_csv(path).to_dict("records")


def paired(res, label, a_sel, b_sel):
    out = []
    for test in ("Dataset_A", "Dataset_B", "Dataset_C"):
        for limb in ("foot", "hand"):
            a = res.query(a_sel + " and test == @test and limb == @limb").set_index("subject").accuracy
            b = res.query(b_sel + " and test == @test and limb == @limb").set_index("subject").accuracy
            a, b = a.align(b, join="inner")
            if not len(a):
                continue
            d = a - b
            out.append(dict(comparison=label, test=test, limb=limb, n=len(d),
                            mean_a=a.mean(), mean_b=b.mean(), mean_difference=d.mean(),
                            pct_subjects_a_better=100 * (d > 0).mean(),
                            p=wilcoxon(d).pvalue if (d != 0).any() else np.nan))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()
    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    suffix = "_partial" if args.subjects else ""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = Parallel(n_jobs=args.n_jobs)(delayed(run_subject_cached)(s, args.n_perm) for s in subjects)
    res = pd.DataFrame([r for rows in all_rows for r in rows])
    res["binomial_threshold"] = [binom.ppf(0.95, n, 0.5) / n for n in res.n_test]
    res["above_binomial"] = res.accuracy > res.binomial_threshold
    res.to_csv(OUT_DIR / f"10_classification_results{suffix}.csv", index=False)

    keys = ["analysis", "channels", "train", "test", "test_runs", "limb", "pipeline"]
    summ = []
    for k, d in res.groupby(keys, sort=False):
        row = dict(zip(keys, k), n_subjects=len(d),
                   accuracy_mean=d.accuracy.mean(), accuracy_sd=d.accuracy.std(),
                   auc_mean=d.auc.mean(), pct_above_binomial=100 * d.above_binomial.mean(),
                   p_vs_chance=wilcoxon(d.accuracy - 0.5).pvalue if len(d) > 5 else np.nan)
        if "p_perm" in d and d.p_perm.notna().any():
            row["pct_perm_p_lt_05"] = 100 * (d.p_perm < 0.05).mean()
            row["median_perm_threshold"] = d.perm_threshold.median()
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT_DIR / f"10_classification_summary{suffix}.csv", index=False)
    show = summ.copy()
    show["p_vs_chance"] = show.p_vs_chance.map(lambda v: f"{v:.1e}")
    pd.set_option("display.width", 250)
    print("\n" + show.round(3).to_string(index=False))

    comp = []
    comp += paired(res, "transfer re-centred (run-1 estimate) vs within, runs 2-3",
                   "analysis == 'transfer' and channels == 'all 32' and pipeline == 'TS+LR recentred (run-1 estimate)'",
                   "analysis == 'within' and channels == 'all 32' and pipeline == 'TS+LR' and test_runs == '2-3'")
    comp += paired(res, "re-centring effect (run-1 estimate vs none), runs 2-3",
                   "analysis == 'transfer' and channels == 'all 32' and pipeline == 'TS+LR recentred (run-1 estimate)'",
                   "analysis == 'transfer' and channels == 'all 32' and pipeline == 'TS+LR' and test_runs == '2-3'")
    comp += paired(res, "sensorimotor vs all 32 channels, within TS+LR",
                   "analysis == 'within' and channels == 'sensorimotor' and pipeline == 'TS+LR' and test_runs == 'all'",
                   "analysis == 'within' and channels == 'all 32' and pipeline == 'TS+LR' and test_runs == 'all'")
    comp += paired(res, "sensorimotor vs all 32 channels, transfer TS+LR (no re-centring)",
                   "analysis == 'transfer' and channels == 'sensorimotor' and pipeline == 'TS+LR' and test_runs == 'all'",
                   "analysis == 'transfer' and channels == 'all 32' and pipeline == 'TS+LR' and test_runs == 'all'")
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT_DIR / f"10_classification_comparisons{suffix}.csv", index=False)
    show = comp.copy()
    show["p"] = show.p.map(lambda v: f"{v:.2g}")
    print("\nPaired comparisons (accuracy, a minus b):")
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
