"""Technical Validation #10b - Decoding benchmark figure for the paper.

Reads 10_classification_results.csv (10_classification_benchmark.py) - nothing
is re-processed. TS+LR pipeline throughout.
  a) within-dataset movement-vs-rest accuracy per participant (dots) and
     mean +/- SD, per dataset and limb; chance (0.5) and the median
     per-participant permutation 95% threshold;
  b) calibration transfer on the same test runs (2-3) of Dataset_B/C: trained
     within the dataset (leave-one-run-out), trained on Dataset_A without
     adaptation, and trained on Dataset_A with re-centring estimated from the
     unlabeled run 1 (mean +/- SEM);
  c) sensorimotor-channel control: accuracy with the 11 sensorimotor channels
     vs all 32 channels (within dataset), one marker per participant x limb;
     filled = still significant with the sensorimotor channels (permutation
     p < 0.05), open = not significant.

Usage:
    py -3.12 10b_classification_figure.py

Writes:
    derivatives/validation/Figure_Decoding.{png,pdf}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT_DIR

# Font scale: the figure is printed at ~55% of its drawn size, so every font is
# scaled up to stay legible (>= 7 pt) in the printed paper.
FS = 1.55
DATASETS = ["Dataset_A", "Dataset_B", "Dataset_C"]
LIMBS = ["foot", "hand"]
COLOR = {"Dataset_A": "#1f4e79", "Dataset_B": "#c0392b", "Dataset_C": "#2e7d32"}
SHORT = {"Dataset_A": "A", "Dataset_B": "B", "Dataset_C": "C"}


def sel(res, **kw):
    d = res
    for k, v in kw.items():
        d = d[d[k] == v]
    return d.set_index("subject").accuracy.sort_index()


def main():
    res = pd.read_csv(OUT_DIR / "10_classification_results.csv")
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9 * FS,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(11, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.6, wspace=0.45)

    # a) within dataset
    ax = fig.add_subplot(gs[0, :])
    labels, x = [], 0
    for ds in DATASETS:
        for limb in LIMBS:
            d = res[(res.analysis == "within") & (res.channels == "all 32") & (res.pipeline == "TS+LR")
                    & (res.test_runs == "all") & (res.test == ds) & (res.limb == limb)]
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(d)), d.accuracy, s=13, color=COLOR[ds],
                       alpha=0.6, lw=0)
            ax.errorbar(x + 0.3, d.accuracy.mean(), yerr=d.accuracy.std(), fmt="o", color="k",
                        ms=4, capsize=3, lw=1)
            ax.hlines(d.perm_threshold.median(), x - 0.25, x + 0.25, color="0.4", ls="--", lw=1)
            labels.append(f"{SHORT[ds]}, {limb}\n{100 * (d.p_perm < 0.05).mean():.0f}% sig.")
            x += 1
        x += 0.5
    ticks = [0, 1, 2.5, 3.5, 5, 6]
    ax.set_xticks(ticks, labels)
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Within dataset", fontsize=9 * FS, loc="left")

    # b) transfer, same test runs 2-3
    ax = fig.add_subplot(gs[1, :2])
    conds = [("within", dict(analysis="within", pipeline="TS+LR")),
             ("from A", dict(analysis="transfer", pipeline="TS+LR")),
             ("from A, re-centred", dict(analysis="transfer", pipeline="TS+LR recentred (run-1 estimate)"))]
    x0, ticks, tlabels = 0, [], []
    for ds in ("Dataset_B", "Dataset_C"):
        for limb in LIMBS:
            vals = [sel(res, channels="all 32", test=ds, limb=limb, test_runs="2-3", **c) for _, c in conds]
            xs = x0 + np.arange(len(conds)) * 0.6
            ax.errorbar(xs, [v.mean() for v in vals], yerr=[v.std() / np.sqrt(len(v)) for v in vals],
                        fmt="o-", color=COLOR[ds], ms=5, capsize=3, lw=1.6)
            ticks += list(xs)
            tlabels += [f"{n}" for n, _ in conds]
            ax.text(xs.mean(), 0.53, f"{SHORT[ds]}, {limb}", ha="center", fontsize=9 * FS, fontweight="bold")
            x0 += 2.4
    ax.set_xticks(ticks, tlabels, rotation=35, ha="right", fontsize=7.5 * FS)
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("Balanced accuracy (runs 2–3)")
    ax.set_title("Calibration transfer to Datasets B and C", fontsize=9 * FS, loc="left")

    # c) sensorimotor-channel control
    ax = fig.add_subplot(gs[1, 2])
    for ds in DATASETS:
        for limb, marker in zip(LIMBS, ("o", "^")):
            a = sel(res, analysis="within", channels="all 32", pipeline="TS+LR", test_runs="all",
                    test=ds, limb=limb)
            d = res[(res.analysis == "within") & (res.channels == "sensorimotor")
                    & (res.pipeline == "TS+LR") & (res.test_runs == "all") & (res.test == ds)
                    & (res.limb == limb)].set_index("subject").sort_index()
            sig = (d.p_perm < 0.05).to_numpy()
            s, a = d.accuracy, a.reindex(d.index)
            ax.scatter(a[sig], s[sig], s=14, color=COLOR[ds], marker=marker, alpha=0.7, lw=0,
                       label=f"{SHORT[ds]}, {limb} ({100 * sig.mean():.0f}% sig.)")
            ax.scatter(a[~sig], s[~sig], s=14, facecolors="none", edgecolors=COLOR[ds],
                       marker=marker, alpha=0.8, lw=0.8)
    ax.plot([0.3, 1], [0.3, 1], color="0.5", lw=0.8, ls="--")
    ax.set_xlim(0.3, 1), ax.set_ylim(0.3, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("All 32 channels")
    ax.set_ylabel("11 sensorimotor channels")
    ax.set_title("Sensorimotor channels only", fontsize=9 * FS, loc="left")
    # Legend outside the panel; filled markers = significant, open = not.
    ax.scatter([], [], s=14, facecolors="none", edgecolors="0.3", lw=0.8, label="open: not sig.")
    ax.legend(fontsize=7 * FS, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    # No figure title: in the paper it goes in the legend.
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"Figure_Decoding.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
