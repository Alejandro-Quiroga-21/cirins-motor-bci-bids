"""Technical Validation #2f - Subject-level statistics of the resting spectrum.

Reads 06_spectral_check.py outputs (06_spectral_psd.npz,
06_spectral_subject_summary.csv) - nothing is re-processed - and reports the
numbers quoted in the paper:
  - FOOOF parameters per dataset (alpha-peak prevalence, IAF, aperiodic
    exponent, goodness of fit);
  - alpha topography: relative alpha power (8-13 Hz / 2-40 Hz) at the
    posterior ROI vs frontal sites (Fp1, Fpz, Fp2, F3, Fz, F4), per dataset;
  - task modulation: posterior relative alpha in Dataset_B (no visual
    stimulation during rest) vs Dataset_A (static on-screen rest cue) and
    Dataset_C (mirror-animation feedback during part of the rest interval).
Paired two-sided Wilcoxon signed-rank tests across the 30 participants.

Usage:
    py -3.12 06c_spectral_stats.py

Writes:
    derivatives/validation/06c_spectral_stats.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import OUT_DIR

DATASETS = ["Dataset_A", "Dataset_B", "Dataset_C"]
FRONTAL = ["Fp1", "Fpz", "Fp2", "F3", "Fz", "F4"]
ALPHA_BAND = (8.0, 13.0)
TOTAL_BAND = (2.0, 40.0)


def main():
    npz = np.load(OUT_DIR / "06_spectral_psd.npz")
    df = pd.read_csv(OUT_DIR / "06_spectral_subject_summary.csv")
    f, ch = npz["freqs"], list(npz["channels"])
    a = (f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1])
    t = (f >= TOTAL_BAND[0]) & (f <= TOTAL_BAND[1])
    post = [ch.index(c) for c in npz["roi"]]
    front = [ch.index(c) for c in FRONTAL]

    rows, post_alpha = [], {}
    for ds in DATASETS:
        psd = npz[f"{ds}__psd"]
        assert list(npz[f"{ds}__subjects"]) == sorted(npz[f"{ds}__subjects"])
        rel = psd[..., a].sum(-1) / psd[..., t].sum(-1)  # (n_sub, n_ch)
        rp, rf = rel[:, post].mean(1), rel[:, front].mean(1)
        post_alpha[ds] = rp
        d = df[df.dataset == ds]
        rows.append(dict(
            dataset=ds, n_subjects=len(d),
            rest_seconds_median=d.rest_seconds.median(), rest_seconds_min=d.rest_seconds.min(),
            alpha_peak_n=int(d.alpha_peak_present.sum()),
            iaf_mean_hz=d.alpha_cf_hz.mean(), iaf_sd_hz=d.alpha_cf_hz.std(),
            iaf_min_hz=d.alpha_cf_hz.min(), iaf_max_hz=d.alpha_cf_hz.max(),
            aperiodic_exponent_mean=d.aperiodic_exponent.mean(),
            aperiodic_exponent_sd=d.aperiodic_exponent.std(),
            fooof_r2_median=d.fooof_r2.median(), fooof_r2_min=d.fooof_r2.min(),
            rel_alpha_posterior=rp.mean(), rel_alpha_frontal=rf.mean(),
            pct_subjects_posterior_gt_frontal=100 * (rp > rf).mean(),
            posterior_vs_frontal_p=wilcoxon(rp, rf).pvalue,
        ))
    out = pd.DataFrame(rows)
    for other in ("Dataset_A", "Dataset_C"):
        out.loc[out.dataset == other, "posterior_alpha_B_vs_this_p"] = \
            wilcoxon(post_alpha["Dataset_B"], post_alpha[other]).pvalue
        out.loc[out.dataset == other, "pct_subjects_B_gt_this"] = \
            100 * (post_alpha["Dataset_B"] > post_alpha[other]).mean()

    no_alpha = df[~df.alpha_peak_present].groupby("subject").size()
    out_path = OUT_DIR / "06c_spectral_stats.csv"
    out.to_csv(out_path, index=False)
    print(out.T.to_string(header=False))
    print(f"\nParticipants without a detected alpha peak in all three datasets: "
          f"{list(no_alpha[no_alpha == len(DATASETS)].index)}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
