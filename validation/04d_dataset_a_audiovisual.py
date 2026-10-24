"""Technical Validation #4d - Auditory response to the movement cues of Dataset_A.

In Dataset_A every cue is an image on the screen, but the movement cues (left and
right arrows) were also accompanied by the same 1,000 Hz, 100 ms beep used in
Datasets B and C (Task_1.py calls winsound.Beep(1000, 100) at every movement cue;
rest cues are silent). This script checks that the beep is visible in the EEG and
quantifies it, using the processing of 04_evoked_potential_check.py --clean for the
auditory ROI (bad channels interpolated, ~18 Hz line notched, linked mastoids,
0.5-30 Hz, -0.2 to 0.6 s epochs, 150 uV rejection, ROI Fz and Cz):

  - grand-average waveforms for Dataset_A movement cues (arrow + beep) and rest cues
    (image only), with the negative and positive peaks of each;
  - per participant, the mean amplitude within +-25 ms of the grand-average N1 and P2
    latencies of the movement cues, for movement and rest cues, compared with a
    two-sided Wilcoxon signed-rank test.

Usage:
    py -3.12 04d_dataset_a_audiovisual.py

Writes:
    derivatives/validation/04d_dataset_a_audiovisual_subjects.csv
    derivatives/validation/04d_dataset_a_audiovisual_grand_average.csv
    derivatives/validation/04d_dataset_a_audiovisual_summary.csv
    derivatives/validation/04d_dataset_a_audiovisual.png
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import DATASETS, TASKS, N_SUBJECTS, OUT_DIR

spec = importlib.util.spec_from_file_location("ep", Path(__file__).with_name("04_evoked_potential_check.py"))
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)

DS = "Dataset_A"
ROI = ["Fz", "Cz"]
CONDITIONS = {"movement": ("left", "right"), "rest": ("rest",)}
HALF_WINDOW = 0.025


def subject_wave(sub_id, cond):
    ep.STIM_TRIAL_TYPES[DS] = CONDITIONS[cond]
    waves, times = [], None
    for task in TASKS:
        for run in range(1, DATASETS[DS] + 1):
            w, t, _ = ep.load_epoch_waveforms(DS, sub_id, task, run, ROI, "linked_mastoids", clean=True)
            if w is not None:
                waves.append(w)
                times = t
    stacked = np.vstack(waves)
    return stacked.mean(axis=0), times, stacked.shape[0]


def main():
    subj = {c: [] for c in CONDITIONS}
    n_ep = {c: [] for c in CONDITIONS}
    times = None
    subjects = [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    for sub_id in subjects:
        for c in CONDITIONS:
            w, times, n = subject_wave(sub_id, c)
            subj[c].append(w)
            n_ep[c].append(n)
        print(f"{sub_id} done", flush=True)
    ga = {c: np.mean(subj[c], axis=0) for c in CONDITIONS}
    pd.DataFrame({"time_s": times, **{f"{c}_grand_avg_uv": ga[c] for c in CONDITIONS}}).to_csv(
        OUT_DIR / "04d_dataset_a_audiovisual_grand_average.csv", index=False)

    peaks = {c: ep.pick_peaks(ga[c], times) for c in CONDITIONS}
    rows = []
    for k, sub_id in enumerate(subjects):
        row = dict(subject=sub_id, n_epochs_movement=n_ep["movement"][k], n_epochs_rest=n_ep["rest"][k])
        for comp, key in (("N1", "neg_peak_latency_ms"), ("P2", "pos_peak_latency_ms")):
            t0 = peaks["movement"][key] / 1000
            m = (times >= t0 - HALF_WINDOW) & (times <= t0 + HALF_WINDOW)
            for c in CONDITIONS:
                row[f"{comp}_{c}_uv"] = subj[c][k][m].mean()
        rows.append(row)
    sdf = pd.DataFrame(rows)
    sdf.to_csv(OUT_DIR / "04d_dataset_a_audiovisual_subjects.csv", index=False)

    summ = []
    for comp, key in (("N1", "neg_peak_latency_ms"), ("P2", "pos_peak_latency_ms")):
        d = sdf[f"{comp}_movement_uv"] - sdf[f"{comp}_rest_uv"]
        summ.append(dict(component=comp, window_center_ms=peaks["movement"][key],
                         movement_mean_uv=sdf[f"{comp}_movement_uv"].mean(),
                         rest_mean_uv=sdf[f"{comp}_rest_uv"].mean(),
                         diff_mean_uv=d.mean(), diff_sd_uv=d.std(), p_wilcoxon=wilcoxon(d).pvalue,
                         pct_participants_same_sign=100 * ((d < 0) if comp == "N1" else (d > 0)).mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT_DIR / "04d_dataset_a_audiovisual_summary.csv", index=False)
    for c in CONDITIONS:
        print(c, peaks[c])
    print(summ.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for c, style in (("movement", "-"), ("rest", "--")):
        ax.plot(times * 1000, ga[c], style, color="k", label=f"{c} cues")
    ax.axvline(0, color="grey", lw=0.6)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("Time from cue onset (ms)")
    ax.set_ylabel("Fz/Cz (µV, linked mastoids)")
    ax.set_title("Dataset A: movement cues (arrow + beep) vs rest cues (image only)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04d_dataset_a_audiovisual.png", dpi=150)


if __name__ == "__main__":
    main()
