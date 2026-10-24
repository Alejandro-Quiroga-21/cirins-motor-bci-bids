"""Technical Validation #5d - Latency of the beta-power increase after movement cues.

Reuses the per-subject ERD% maps of 05_erd_ers_check.py (05_erd_ers_tfr_clean.npz,
bad channels interpolated and ~18 Hz line notched out) - does not re-process any
recording. For every dataset and limb, the 13-30 Hz time course at the expected
site (contralateral C3/C4 for the hand, Cz for the foot; left and right cues pooled
per subject as in 05b) is taken, and two descriptive latencies are read relative to
movement-cue onset:
  - beta minimum: time of the lowest beta power between 0 and 4 s;
  - return to baseline: first time after that minimum at which beta power is above
    the pre-cue baseline (0 %).
Both are given for the grand average and per participant (median and IQR), next to
the median measured movement-cue duration (09_cue_timing_summary.csv).

They describe the cue-locked EEG only: movement onset and termination were not
recorded, and the Morlet wavelets (n_cycles = f / 2) smear beta power by roughly
+/- 0.1-0.2 s.

Usage:
    py -3.12 05d_beta_rebound_latency.py

Writes:
    derivatives/validation/05d_beta_rebound_latency.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_DIR

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("fig05b", HERE / "05b_erd_ers_figure.py")
fig05b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig05b)

LIMB = {"lower": "foot", "upper": "hand"}
MIN_SEARCH = (0.0, 4.0)


def latencies(course, times):
    """(time of minimum in MIN_SEARCH, first time after it with power > 0)."""
    w = (times >= MIN_SEARCH[0]) & (times <= MIN_SEARCH[1])
    t_min = times[w][np.argmin(course[w])]
    after = (times > t_min) & (course > 0)
    return t_min, (times[after][0] if after.any() else np.nan)


def main():
    npz = np.load(OUT_DIR / "05_erd_ers_tfr_clean.npz")
    freqs, times = npz["freqs"], npz["times"]
    timing = pd.read_csv(OUT_DIR / "09_cue_timing_summary.csv")
    rows = []
    for ds in fig05b.DATASETS:
        for task in ("lower", "upper"):
            contra, _ = fig05b.site_maps(npz, ds, task)
            beta = fig05b.band_course(contra, freqs, fig05b.BETA_BAND)  # (n_sub, n_times)
            ga_min, ga_zero = latencies(beta.mean(axis=0), times)
            per = np.array([latencies(b, times) for b in beta])
            cue = timing.query("dataset == @ds and task == @task and measure == 'movement_interval'")
            rows.append(dict(
                dataset=ds, limb=LIMB[task], n_subjects=len(beta),
                cue_duration_median_s=round(float(cue.median_s.iloc[0]), 3),
                ga_beta_min_s=round(float(ga_min), 2), ga_return_to_baseline_s=round(float(ga_zero), 2),
                subj_beta_min_median_s=round(float(np.median(per[:, 0])), 2),
                subj_beta_min_q1_s=round(float(np.percentile(per[:, 0], 25)), 2),
                subj_beta_min_q3_s=round(float(np.percentile(per[:, 0], 75)), 2),
                subj_return_median_s=round(float(np.nanmedian(per[:, 1])), 2),
                subj_return_q1_s=round(float(np.nanpercentile(per[:, 1], 25)), 2),
                subj_return_q3_s=round(float(np.nanpercentile(per[:, 1], 75)), 2),
                subj_return_before_cue_end_pct=round(
                    100 * float(np.mean(per[:, 1] < cue.median_s.iloc[0])), 1),
                subj_no_return=int(np.isnan(per[:, 1]).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "05d_beta_rebound_latency.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
