"""Technical Validation #5b - Composite ERD/ERS figure for the paper.

Reuses the per-subject ERD% maps saved by 05_erd_ers_check.py
(05_erd_ers_tfr.npz) - does not re-process any recording. One column per
dataset, four rows:
  a) hand movements: grand-average time-frequency ERD/ERS map at the
     contralateral Laplacian site (C3 for right-hand cues, C4 for left-hand cues);
  b) hand movements: mu (8-13 Hz) and beta (13-30 Hz) ERD/ERS time course,
     contralateral vs ipsilateral site (mean +/- SEM across subjects);
  c) foot movements: time-frequency map at Cz;
  d) foot movements: mu and beta time course at Cz.
Left and right cues are pooled per subject after mapping them to
contralateral/ipsilateral sites (Dataset_A has both sides; Dataset_B/C only
the dominant side).

By default it uses 05_erd_ers_tfr_clean.npz (05 run with --clean: bad channels
interpolated, ~18 Hz line notched out), the version reported in the paper;
--original builds the same figure from the uncleaned 05_erd_ers_tfr.npz.

Usage:
    py -3.12 05b_erd_ers_figure.py [--original]

Writes:
    derivatives/validation/Figure_ERD_ERS.{png,pdf}            (default, clean)
    derivatives/validation/Figure_ERD_ERS_original.{png,pdf}   (--original)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT_DIR

# Font scale: the figure is printed at ~55% of its drawn size, so every font is
# scaled up to stay legible (>= 7 pt) in the printed paper.
FS = 1.6
DATASETS = ["Dataset_A", "Dataset_B", "Dataset_C"]
COL_LABEL = {
    "Dataset_A": "Dataset A\ncalibration\n(visual cues)",
    "Dataset_B": "Dataset B\nopen loop\n(beep, no feedback)",
    "Dataset_C": "Dataset C\nsimulated closed loop\n(beep, sham feedback)",
}
MU_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
# The second dashed line marks the end of the movement cue: the median measured
# movement-cue duration of each dataset and limb (09_cue_timing_summary.csv;
# 2.78-2.79 s in Dataset A, 2.60 s in B, 2.64 s in C).
# Dataset C: the sham-feedback mirror animation starts at the end of the cue.
ANIMATION_S = {"upper": 3.0, "lower": 5.0}
VLIM = 60.0


def site_maps(npz, ds, task):
    """Per-subject (contra, ipsi) ERD% maps (n_freqs, n_times), pooling left/right
    cues. For the foot, 'contra' is Cz and 'ipsi' is the C3/C4 mean."""
    ch = list(npz["channels"])
    per_sub = {}
    for cue in ("left", "right"):
        key = f"{ds}__{task}__{cue}__erd"
        if key not in npz:
            continue
        erd = npz[key]
        subs = npz[f"{ds}__{task}__{cue}__subjects"]
        if task == "upper":
            c, i = ("C3", "C4") if cue == "right" else ("C4", "C3")
            contra, ipsi = erd[:, ch.index(c)], erd[:, ch.index(i)]
        else:
            contra = erd[:, ch.index("Cz")]
            ipsi = erd[:, [ch.index("C3"), ch.index("C4")]].mean(axis=1)
        for s, m_c, m_i in zip(subs, contra, ipsi):
            per_sub.setdefault(s, []).append((m_c, m_i))
    contra = np.stack([np.mean([p[0] for p in v], axis=0) for v in per_sub.values()])
    ipsi = np.stack([np.mean([p[1] for p in v], axis=0) for v in per_sub.values()])
    return contra, ipsi


def band_course(maps, freqs, band):
    fmask = (freqs >= band[0]) & (freqs <= band[1])
    return maps[:, fmask, :].mean(axis=1)  # (n_sub, n_times)


def plot_tfr(ax, grand, times, freqs, title, cue_end):
    im = ax.pcolormesh(times, freqs, grand, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, shading="auto")
    for x in (0, cue_end):
        ax.axvline(x, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=8.5 * FS)
    return im


def plot_course(ax, maps_c, maps_i, times, freqs, labels, cue_end, animation=None):
    ax.axhline(0, color="0.75", lw=0.8)
    if animation:  # Dataset C: mark the sham-feedback animation
        end = min(cue_end + animation, times[-1])  # the foot animation outlasts the epoch
        ax.axvspan(cue_end, end, color="0.85", alpha=0.6, lw=0, zorder=0)
        ax.text((cue_end + end) / 2, 0.97, "animation", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5 * FS, color="0.35")
    for x in (0, cue_end):
        ax.axvline(x, color="0.6", lw=0.8, ls="--")
    for band, color in ((MU_BAND, "#1f4e79"), (BETA_BAND, "#c0392b")):
        name = "mu" if band == MU_BAND else "beta"
        for maps, ls, lab in ((maps_c, "-", labels[0]), (maps_i, ":", labels[1])):
            bc = band_course(maps, freqs, band)
            m = bc.mean(axis=0)
            sem = bc.std(axis=0, ddof=1) / np.sqrt(bc.shape[0])
            ax.plot(times, m, color=color, ls=ls, lw=1.4, label=f"{name}, {lab}")
            if ls == "-":
                ax.fill_between(times, m - sem, m + sem, color=color, alpha=0.18, lw=0)
    ax.set_ylabel("ERD/ERS (%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()
    npz = np.load(OUT_DIR / ("05_erd_ers_tfr.npz" if args.original else "05_erd_ers_tfr_clean.npz"))
    out_suffix = "_original" if args.original else ""
    times, freqs = npz["times"], npz["freqs"]
    timing = pd.read_csv(OUT_DIR / "09_cue_timing_summary.csv").query("measure == 'movement_interval'")
    cue_end = {(r.dataset, r.task): r.median_s for r in timing.itertuples()}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9 * FS,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(4, 3, figsize=(11, 11.5), sharex=True,
                             gridspec_kw=dict(hspace=0.62, wspace=0.42))

    course_axes = {"upper": [], "lower": []}
    for col, ds in enumerate(DATASETS):
        for r0, task, site, labels in ((0, "upper", "contralateral", ("contra", "ipsi")),
                                       (2, "lower", "Cz", ("Cz", "C3/C4"))):
            contra, ipsi = site_maps(npz, ds, task)
            limb = "Hand" if task == "upper" else "Foot"
            im = plot_tfr(axes[r0, col], contra.mean(axis=0), times, freqs,
                          f"{limb} — {site} (n = {contra.shape[0]})", cue_end[(ds, task)])
            plot_course(axes[r0 + 1, col], contra, ipsi, times, freqs, labels, cue_end[(ds, task)],
                        animation=ANIMATION_S[task] if ds == "Dataset_C" else None)
            course_axes[task].append(axes[r0 + 1, col])
        axes[0, col].annotate(COL_LABEL[ds], (0.5, 1.30), xycoords="axes fraction",
                              ha="center", va="bottom", fontsize=9 * FS, fontweight="bold")
        axes[3, col].set_xlabel("Time from cue onset (s)")

    for ax in axes.flat:  # the animation span must not widen the time axis
        ax.set_xlim(times[0], times[-1])
        ax.set_xticks([-2, 0, 2, 4, 6])
    for task in course_axes:  # shared y-range per limb for comparability
        lo = min(a.get_ylim()[0] for a in course_axes[task])
        hi = max(a.get_ylim()[1] for a in course_axes[task])
        for a in course_axes[task]:
            a.set_ylim(lo, hi)
    # One-line legend above each row of time courses, outside the panels.
    for r in (1, 3):
        handles, labels = axes[r, 0].get_legend_handles_labels()
        axes[r, 1].legend(handles, labels, fontsize=7.5 * FS, loc="lower center",
                          bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, columnspacing=1.2, handlelength=1.8)

    # Colorbar in its own axis, right of the figure, spanning the TFR rows (a-c),
    # so it neither shrinks the TFR panels nor overlaps the legends.
    top = axes[0, 2].get_position()
    bottom = axes[2, 2].get_position()
    cax = fig.add_axes([top.x1 + 0.015, bottom.y0, 0.012, top.y1 - bottom.y0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("ERD/ERS (%), baseline −1.5 to −0.5 s")
    # No figure title: in the paper it goes in the legend.

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"Figure_ERD_ERS{out_suffix}.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
