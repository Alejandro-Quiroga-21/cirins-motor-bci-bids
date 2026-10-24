"""Technical Validation #5c - Subject-level ERD/ERS statistics for the paper.

Reads the per-subject summary of 05_erd_ers_check.py (the '_clean' run by
default, i.e. bad channels interpolated and ~18 Hz line notched out; --original
for the uncleaned one) and reduces it to one value per subject and limb, at the
physiologically expected site:
  - hand (task-upper): contralateral Laplacian (C3 for right-hand cues, C4 for
    left-hand cues), compared against the ipsilateral one;
  - foot (task-lower): Cz, compared against the mean of C3 and C4.
Left and right cues are averaged within subject (Dataset_A has both sides;
Dataset_B/C only the dominant side), so every subject contributes once.

Tests (two-sided Wilcoxon signed-rank, n = 30 per dataset x limb):
  - mu ERD (0.5-2.5 s) and beta ERD at the expected site vs 0;
  - post-movement beta rebound (3-5 s) at the expected site vs 0;
  - spatial specificity of the rebound: expected site vs control site(s);
  - (hand) mu ERD contralateral vs ipsilateral.

Usage:
    py -3.12 05c_erd_ers_stats.py [--original]

Writes:
    derivatives/validation/05c_erd_ers_limb_stats[_original].csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import OUT_DIR


def per_subject(s):
    rows = []
    for (ds, task, sub), d in s.groupby(["dataset", "task", "subject"]):
        if task == "upper":
            rows.append(dict(dataset=ds, limb="hand", subject=sub, site="contralateral C3/C4",
                             control="ipsilateral C3/C4",
                             mu=d["mu_erd_contra"].mean(), mu_ctrl=d["mu_erd_ipsi"].mean(),
                             beta=d["beta_erd_contra"].mean(),
                             rebound=d["beta_rebound_contra"].mean(),
                             rebound_ctrl=d["beta_rebound_ipsi"].mean()))
        else:
            rows.append(dict(dataset=ds, limb="foot", subject=sub, site="Cz", control="mean of C3, C4",
                             mu=d["mu_erd_Cz"].mean(), mu_ctrl=d[["mu_erd_C3", "mu_erd_C4"]].mean(1).mean(),
                             beta=d["beta_erd_Cz"].mean(),
                             rebound=d["beta_rebound_Cz"].mean(),
                             rebound_ctrl=d[["beta_rebound_C3", "beta_rebound_C4"]].mean(1).mean()))
    return pd.DataFrame(rows)


def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()
    src = "" if args.original else "_clean"
    s = pd.read_csv(OUT_DIR / f"05_erd_ers_subject_summary{src}.csv")
    p = per_subject(s)

    rows = []
    for (ds, limb), d in p.groupby(["dataset", "limb"], sort=True):
        rows.append(dict(
            dataset=ds, limb=limb, n_subjects=len(d), site=d["site"].iloc[0], control_site=d["control"].iloc[0],
            mu_erd_mean=d["mu"].mean(), mu_erd_sem=sem(d["mu"]),
            mu_erd_pct_subjects_negative=100 * (d["mu"] < 0).mean(), mu_erd_p=wilcoxon(d["mu"]).pvalue,
            beta_erd_mean=d["beta"].mean(), beta_erd_sem=sem(d["beta"]),
            beta_erd_pct_subjects_negative=100 * (d["beta"] < 0).mean(), beta_erd_p=wilcoxon(d["beta"]).pvalue,
            rebound_mean=d["rebound"].mean(), rebound_sem=sem(d["rebound"]),
            rebound_pct_subjects_positive=100 * (d["rebound"] > 0).mean(), rebound_p=wilcoxon(d["rebound"]).pvalue,
            rebound_control_mean=d["rebound_ctrl"].mean(),
            rebound_pct_subjects_site_gt_control=100 * (d["rebound"] > d["rebound_ctrl"]).mean(),
            rebound_site_vs_control_p=wilcoxon(d["rebound"], d["rebound_ctrl"]).pvalue,
            mu_control_mean=d["mu_ctrl"].mean(),
            mu_site_vs_control_p=wilcoxon(d["mu"], d["mu_ctrl"]).pvalue,
        ))
    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"05c_erd_ers_limb_stats{'_original' if args.original else ''}.csv"
    out.to_csv(out_path, index=False)

    fmt = out.copy()
    for c in fmt.columns:
        if c.endswith("_p"):
            fmt[c] = fmt[c].map(lambda v: f"{v:.1e}")
    print(fmt.round(1).drop(columns=["site", "control_site"]).to_string(index=False))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
