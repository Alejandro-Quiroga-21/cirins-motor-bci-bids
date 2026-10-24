"""Technical Validation #8 - Robustness of the neurophysiological checks.

Compares the VEP/AEP (04) and ERD/ERS (05) results computed on the data as
published against the same pipelines run with --clean (channels marked bad in
the BIDS _channels.tsv interpolated, ~18 Hz line notched out where detected;
see common.clean_raw). Reads only the CSV outputs of both runs - nothing is
re-processed.

Reports, per dataset:
  - VEP/AEP grand average: peak amplitude/latency before vs after, waveform
    correlation and maximum absolute difference;
  - ERD/ERS grand average: every summary metric before vs after;
  - the same comparisons restricted to the subjects with bad channels
    (sub-11, sub-12, sub-14), where any effect should concentrate.

Usage:
    py -3.12 08_robustness_compare.py

Writes:
    derivatives/validation/08_robustness_evoked.csv
    derivatives/validation/08_robustness_erd.csv
"""
import numpy as np
import pandas as pd

from common import OUT_DIR, recording_bads

PEAK_COLS = ["neg_peak_uv", "neg_peak_latency_ms", "pos_peak_uv", "pos_peak_latency_ms"]


def affected_subjects():
    return sorted({s for s in (f"sub-{i:02d}" for i in range(1, 31))
                   if recording_bads("Dataset_A", s, "lower", 1)})


def compare_evoked(affected):
    ga0 = pd.read_csv(OUT_DIR / "04_evoked_potential_grand_average.csv")
    ga1 = pd.read_csv(OUT_DIR / "04_evoked_potential_grand_average_clean.csv")
    s0 = pd.read_csv(OUT_DIR / "04_evoked_potential_subject_summary.csv")
    s1 = pd.read_csv(OUT_DIR / "04_evoked_potential_subject_summary_clean.csv")

    rows = []
    for ds in [c.replace("_grand_avg_uv", "") for c in ga0.columns if c.endswith("_grand_avg_uv")]:
        w0, w1 = ga0[f"{ds}_grand_avg_uv"].to_numpy(), ga1[f"{ds}_grand_avg_uv"].to_numpy()
        rows.append(dict(dataset=ds, scope="grand average (n=30)",
                         waveform_r=round(float(np.corrcoef(w0, w1)[0, 1]), 5),
                         max_abs_diff_uv=round(float(np.abs(w1 - w0).max()), 3)))
        for scope, sub in (("all subjects, mean of per-subject peaks", None),
                           ("affected subjects: " + ",".join(affected), affected)):
            a = s0[s0.dataset == ds]
            b = s1[s1.dataset == ds]
            if sub is not None:
                a, b = a[a.subject.isin(sub)], b[b.subject.isin(sub)]
            r = dict(dataset=ds, scope=scope)
            for c in PEAK_COLS:
                r[f"{c}_before"] = round(a[c].mean(), 2)
                r[f"{c}_after"] = round(b[c].mean(), 2)
            rows.append(r)
    return pd.DataFrame(rows)


def compare_erd(affected):
    g0 = pd.read_csv(OUT_DIR / "05_erd_ers_grand_average_summary.csv")
    g1 = pd.read_csv(OUT_DIR / "05_erd_ers_grand_average_summary_clean.csv")
    s0 = pd.read_csv(OUT_DIR / "05_erd_ers_subject_summary.csv")
    s1 = pd.read_csv(OUT_DIR / "05_erd_ers_subject_summary_clean.csv")
    keys = ["dataset", "task", "cue"]
    metrics = [c for c in g0.columns if c.startswith(("mu_erd_", "beta_erd_", "beta_rebound_"))
               and not c.endswith(("_contra", "_ipsi"))]
    m = g0.merge(g1, on=keys, suffixes=("_before", "_after"))
    out = m[keys].copy()
    out["scope"] = "grand average"
    for c in metrics:
        out[f"{c}_before"], out[f"{c}_after"] = m[f"{c}_before"], m[f"{c}_after"]
    diffs = np.abs(np.column_stack([m[f"{c}_after"] - m[f"{c}_before"] for c in metrics]))
    out["max_abs_change_pct_points"] = diffs.max(axis=1).round(2)

    sm = s0.merge(s1, on=keys + ["subject"], suffixes=("_before", "_after"))
    sa = sm[sm.subject.isin(affected)]
    sub_out = sa[keys + ["subject"]].copy()
    sub_out["scope"] = "affected subject"
    for c in metrics:
        sub_out[f"{c}_before"], sub_out[f"{c}_after"] = sa[f"{c}_before"], sa[f"{c}_after"]
    sd = np.abs(np.column_stack([sa[f"{c}_after"] - sa[f"{c}_before"] for c in metrics]))
    sub_out["max_abs_change_pct_points"] = sd.max(axis=1).round(2)

    others = sm[~sm.subject.isin(affected)]
    od = np.abs(np.column_stack([others[f"{c}_after"] - others[f"{c}_before"] for c in metrics]))
    return pd.concat([out, sub_out], ignore_index=True), metrics, od


def main():
    affected = affected_subjects()
    print(f"Subjects with channels marked bad: {affected}\n")

    ev = compare_evoked(affected)
    ev.to_csv(OUT_DIR / "08_robustness_evoked.csv", index=False)
    print("=== VEP/AEP ===")
    ga = ev[ev.scope.str.startswith("grand")][["dataset", "waveform_r", "max_abs_diff_uv"]]
    print(ga.to_string(index=False))
    cols = ["dataset", "scope"] + [f"{c}_{w}" for c in PEAK_COLS for w in ("before", "after")]
    print("\n" + ev[~ev.scope.str.startswith("grand")][cols].to_string(index=False))

    erd, metrics, others_diff = compare_erd(affected)
    erd.to_csv(OUT_DIR / "08_robustness_erd.csv", index=False)
    print("\n=== ERD/ERS: grand average, max |change| over all metrics (percentage points) ===")
    g = erd[erd.scope == "grand average"]
    print(g[["dataset", "task", "cue", "max_abs_change_pct_points"]].to_string(index=False))
    show = ["dataset", "task", "cue"] + [f"{c}_{w}" for c in ("mu_erd_C3", "mu_erd_Cz", "mu_erd_C4",
                                                            "beta_rebound_C3", "beta_rebound_Cz",
                                                            "beta_rebound_C4") for w in ("before", "after")]
    print("\n" + g[show].round(1).to_string(index=False))
    print("\n=== ERD/ERS: affected subjects, max |change| per condition (percentage points) ===")
    a = erd[erd.scope == "affected subject"]
    print(a[["subject", "dataset", "task", "cue", "max_abs_change_pct_points"]].to_string(index=False))
    print(f"\nUnaffected subjects (only the ~18 Hz notch applies): max |change| over all metrics "
          f"and conditions: median {np.median(others_diff.max(axis=1)):.2f}, "
          f"max {others_diff.max():.2f} percentage points")


if __name__ == "__main__":
    main()
