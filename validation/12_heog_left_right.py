"""Technical Validation #12 - Horizontal eye movements after left and right cues (Dataset_A).

In Dataset_A the movement cues are left- and right-pointing arrows. If participants
shifted their gaze towards the arrow and held it there, the movement epochs of
left and right cues would differ in horizontal EOG, a non-motor signal that could
help a classifier. There is no EOG channel, so the bipolar F7 - F8 derivation
(frontolateral sites next to the outer canthi; independent of the CPz reference)
is used as a proxy of horizontal EOG: a gaze shift to the left makes it more
positive or negative, consistently, and a sustained shift gives a DC offset.

Method: per recording, F7 - F8 is low-pass filtered at 15 Hz (no high-pass, so
that sustained offsets are kept), epoched from -0.5 to 2.5 s around every left and
right movement cue and baseline-corrected on -0.5 to 0 s. Epochs whose peak-to-peak
amplitude exceeds 200 uV are dropped. For each participant and limb, the mean
F7 - F8 over the movement window used by the decoding benchmark (0.5-2.5 s) and
over the early window (0-0.5 s) is averaged across left cues and across right
cues; the left-minus-right difference is tested across participants with a
two-sided Wilcoxon signed-rank test.

Usage:
    py -3.12 12_heog_left_right.py

Writes:
    derivatives/validation/12_heog_left_right_subjects.csv
    derivatives/validation/12_heog_left_right_summary.csv
"""
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, cues_only

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

DS = "Dataset_A"
LIMB = {"lower": "foot", "upper": "hand"}
TMIN, TMAX = -0.5, 2.5
WINDOWS = {"early_0_0.5": (0.0, 0.5), "movement_0.5_2.5": (0.5, 2.5)}
LOWPASS = 15.0
REJECT_UV = 200.0


def recording_means(sub_id, task, run):
    stem = f"{sub_id}_task-{task}_run-{run:02d}"
    eeg_dir = BIDS_ROOT / DS / sub_id / "eeg"
    raw = mne.io.read_raw_brainvision(eeg_dir / f"{stem}_eeg.vhdr", preload=True)
    raw.pick(["F7", "F8"])
    sfreq = raw.info["sfreq"]
    heog = raw.get_data(units="uV")
    heog = mne.filter.filter_data(heog[0] - heog[1], sfreq, None, LOWPASS)
    ev = cues_only(pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t"))
    out = {"left": {w: [] for w in WINDOWS}, "right": {w: [] for w in WINDOWS}}
    n_rej = 0
    for r in ev[ev.trial_type.isin(["left", "right"])].itertuples():
        a = int(r.sample) + int(round(TMIN * sfreq))
        b = int(r.sample) + int(round(TMAX * sfreq))
        if a < 0 or b > len(heog):
            continue
        seg = heog[a:b] - heog[a:int(r.sample)].mean()
        if np.ptp(seg) > REJECT_UV:
            n_rej += 1
            continue
        t = np.arange(len(seg)) / sfreq + TMIN
        for w, (t0, t1) in WINDOWS.items():
            out[r.trial_type][w].append(seg[(t >= t0) & (t < t1)].mean())
    return out, n_rej


def main():
    rows = []
    for i in range(1, N_SUBJECTS + 1):
        sub_id = f"sub-{i:02d}"
        for task in TASKS:
            acc = {"left": {w: [] for w in WINDOWS}, "right": {w: [] for w in WINDOWS}}
            n_rej = 0
            for run in range(1, DATASETS[DS] + 1):
                out, nr = recording_means(sub_id, task, run)
                n_rej += nr
                for side in acc:
                    for w in WINDOWS:
                        acc[side][w] += out[side][w]
            row = dict(subject=sub_id, limb=LIMB[task], n_left=len(acc["left"]["early_0_0.5"]),
                       n_right=len(acc["right"]["early_0_0.5"]), n_rejected=n_rej)
            for w in WINDOWS:
                row[f"left_{w}_uV"] = np.mean(acc["left"][w])
                row[f"right_{w}_uV"] = np.mean(acc["right"][w])
                row[f"diff_{w}_uV"] = row[f"left_{w}_uV"] - row[f"right_{w}_uV"]
            rows.append(row)
        print(f"{sub_id} done", flush=True)
    subj = pd.DataFrame(rows)
    subj.to_csv(OUT_DIR / "12_heog_left_right_subjects.csv", index=False)

    summ = []
    for limb, d in subj.groupby("limb"):
        for w in WINDOWS:
            x = d[f"diff_{w}_uV"]
            summ.append(dict(limb=limb, window=w, n=len(x), mean_diff_uV=x.mean(), sd_diff_uV=x.std(),
                             median_diff_uV=x.median(), p_wilcoxon=wilcoxon(x).pvalue,
                             pct_left_more_positive=100 * (x > 0).mean(),
                             epochs_rejected_pct=100 * d.n_rejected.sum()
                             / (d.n_rejected.sum() + d.n_left.sum() + d.n_right.sum())))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT_DIR / "12_heog_left_right_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
