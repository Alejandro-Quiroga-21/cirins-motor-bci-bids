"""Technical Validation #4c - Composite VEP/AEP figure for the paper.

Reuses the already-computed results of 04_evoked_potential_check.py (grand-
average ROI waveforms) and 04b_evoked_topomap.py (per-channel amplitude at each
dataset's own peak latency) to build one publication-ready, 3x3 figure:
one row per dataset (A: VEP, B/C: AEP), showing the grand-average waveform
(with the picked peak latencies marked) next to the scalp topography at the
negative and positive peak. Does not re-run any epoching/filtering.

By default it uses the '_clean' outputs (bad channels interpolated, ~18 Hz line
notched out: 04 and 04b run with --clean), which is the version reported in the
paper; --original builds the same figure from the uncleaned outputs, for
comparison (see 08_robustness_compare.py).

Usage:
    py -3.12 04c_paper_figure.py [--original]

Writes:
    derivatives/validation/Figure_VEP_AEP.{png,pdf}            (default, clean)
    derivatives/validation/Figure_VEP_AEP_original.{png,pdf}   (--original)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from common import OUT_DIR, EXPECTED_CHANNELS

mne.set_log_level("ERROR")

# Font scale: the figure is printed at ~55% of its drawn size, so every font is
# scaled up to stay legible (>= 7 pt) in the printed paper.
FS = 1.5
EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]
ROI = {"Dataset_A": ["O1", "Oz", "O2", "POz"], "Dataset_B": ["Fz", "Cz"], "Dataset_C": ["Fz", "Cz"]}
ROW_LABEL = {
    "Dataset_A": "Dataset A — all cue onsets (VEP)\nROI: O1, Oz, O2, POz · ref: native (CPz)",
    "Dataset_B": "Dataset B — auditory cues (AEP)\nROI: Fz, Cz · ref: linked mastoids",
    "Dataset_C": "Dataset C — auditory cues (AEP)\nROI: Fz, Cz · ref: linked mastoids",
}
# Component names of the picked (negative, positive) peaks: VEP P1/N1, AEP N1/P2.
PEAK_NAME = {"Dataset_A": ("N1", "P1"), "Dataset_B": ("N1", "P2"), "Dataset_C": ("N1", "P2")}


def build_info():
    info = mne.create_info(EEG_CHANNELS, sfreq=500.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()
    src = "" if args.original else "_clean"
    out_suffix = "_original" if args.original else ""
    wave_df = pd.read_csv(OUT_DIR / f"04_evoked_potential_grand_average{src}.csv")
    topo_df = pd.read_csv(OUT_DIR / f"04b_topomap_channel_data{src}.csv")
    info = build_info()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9 * FS,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    datasets = ["Dataset_A", "Dataset_B", "Dataset_C"]
    fig = plt.figure(figsize=(9.5, 9.6))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.5, 1, 1], wspace=0.45, hspace=0.95)

    for row, ds in enumerate(datasets):
        roi = ROI[ds]
        t = wave_df["time_s"].to_numpy() * 1000  # ms
        w = wave_df[f"{ds}_grand_avg_uv"].to_numpy()

        sub = topo_df[topo_df["dataset"] == ds]
        t_neg = sub["latency_neg_ms"].iloc[0]
        t_pos = sub["latency_pos_ms"].iloc[0]
        amp_neg = sub.set_index("channel").reindex(EEG_CHANNELS)["amp_at_neg_uv"].to_numpy()
        amp_pos = sub.set_index("channel").reindex(EEG_CHANNELS)["amp_at_pos_uv"].to_numpy()
        v_neg = float(w[np.argmin(np.abs(t - t_neg))])
        v_pos = float(w[np.argmin(np.abs(t - t_pos))])

        # --- waveform panel ---
        ax = fig.add_subplot(gs[row, 0])
        ax.axhline(0, color="0.75", lw=0.8, zorder=1)
        ax.axvline(0, color="0.75", lw=0.8, ls="--", zorder=1)
        ax.plot(t, w, color="#1f4e79", lw=1.6, zorder=3)
        ax.plot([t_neg, t_pos], [v_neg, v_pos], "o", color="#c0392b", ms=5, zorder=4)

        ymin_d, ymax_d = float(w.min()), float(w.max())
        yrange = ymax_d - ymin_d
        ax.set_ylim(ymin_d - 0.6 * yrange, ymax_d + 0.6 * yrange)

        # Labels go beyond each peak (above a positive one, below a negative one),
        # where the waveform cannot cross them; left-aligned at the peak so that the
        # early peaks do not run over the y axis.
        name_neg, name_pos = PEAK_NAME[ds]
        ax.annotate(f"{name_neg}  {v_neg:.1f} µV, {t_neg:.0f} ms".replace("-", "−"), (t_neg, v_neg),
                    textcoords="offset points", xytext=(-6, -9), fontsize=7.5 * FS, ha="left",
                    va="top", color="#c0392b")
        ax.annotate(f"{name_pos}  {v_pos:+.1f} µV, {t_pos:.0f} ms", (t_pos, v_pos),
                    textcoords="offset points", xytext=(-6, 9), fontsize=7.5 * FS, ha="left",
                    va="bottom", color="#c0392b")
        ax.set_xlim(-100, 500)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.yaxis.set_label_coords(-0.2, 0.5)  # same position in every row
        ax.set_title(ROW_LABEL[ds], fontsize=8.5 * FS, loc="left", pad=14)

        # --- topomap panels, in temporal order ---
        vmax = max(np.nanmax(np.abs(amp_neg)), np.nanmax(np.abs(amp_pos)))
        peaks = sorted([(amp_neg, t_neg, name_neg), (amp_pos, t_pos, name_pos)],
                       key=lambda p: p[1])
        for col, (amp, t_lat, label) in enumerate(peaks):
            ax_t = fig.add_subplot(gs[row, col + 1])
            im, _ = mne.viz.plot_topomap(
                amp, info, axes=ax_t, show=False, vlim=(-vmax, vmax), cmap="RdBu_r",
                mask=np.isin(EEG_CHANNELS, roi),
                mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor="k", markersize=6 * FS),
            )
            ax_t.set_title(f"{label}, {t_lat:.0f} ms", fontsize=8.5 * FS)
            if col == 1:
                # inset colour bar, so both topomaps of a row keep the same size
                cax = fig.colorbar(im, cax=ax_t.inset_axes([1.08, 0.15, 0.05, 0.7]))
                cax.set_label("µV", fontsize=8 * FS)
                cax.ax.tick_params(labelsize=7 * FS)

    # No figure title: in the paper it goes in the legend.
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"Figure_VEP_AEP{out_suffix}.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
