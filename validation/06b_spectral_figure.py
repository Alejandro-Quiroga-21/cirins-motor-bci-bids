"""Technical Validation #2c - Resting spectral profile figure for the paper.

Reuses 06_spectral_check.py outputs (06_spectral_psd.npz,
06_spectral_subject_summary.csv) - does not re-process any recording.
One column per dataset:
  a) posterior-ROI resting PSD (log-log): every subject (thin) and the grand
     average (thick), with the alpha band shaded;
  b) scalp topography of relative alpha power (8-13 Hz / 2-40 Hz), grand average;
and a bottom row with the per-subject FOOOF parameters across datasets:
  c) individual alpha frequency (subjects with a detected alpha peak);
  d) aperiodic (1/f) exponent.

Usage:
    py -3.12 06b_spectral_figure.py

Writes:
    derivatives/validation/Figure_Spectral.png
    derivatives/validation/Figure_Spectral.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from common import OUT_DIR

mne.set_log_level("ERROR")

DATASETS = ["Dataset_A", "Dataset_B", "Dataset_C"]
COL_LABEL = {
    "Dataset_A": "Dataset A — calibration",
    "Dataset_B": "Dataset B — open loop",
    "Dataset_C": "Dataset C — simulated closed loop",
}
COLOR = {"Dataset_A": "#1f4e79", "Dataset_B": "#c0392b", "Dataset_C": "#2e7d32"}
ALPHA_BAND = (8.0, 13.0)
TOTAL_BAND = (2.0, 40.0)
FMIN, FMAX = 1.0, 45.0


def build_info(ch_names):
    info = mne.create_info(list(ch_names), sfreq=500.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False, on_missing="ignore")
    return info


def strip(ax, df, col, ylabel):
    rng = np.random.default_rng(0)
    labels = []
    for i, ds in enumerate(DATASETS):
        v = df.loc[df["dataset"] == ds, col].dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=COLOR[ds], alpha=0.7, lw=0)
        ax.errorbar(i + 0.28, v.mean(), yerr=v.std(ddof=1), fmt="o", color="k", ms=4, capsize=3, lw=1)
        labels.append(f"{COL_LABEL[ds].split(' — ')[0]}\n(n = {len(v)})")
    ax.set_xticks(range(len(DATASETS)), labels)
    ax.set_xlim(-0.5, len(DATASETS) - 0.5)
    ax.set_ylabel(ylabel)


def main():
    npz = np.load(OUT_DIR / "06_spectral_psd.npz")
    df = pd.read_csv(OUT_DIR / "06_spectral_subject_summary.csv")
    freqs, ch = npz["freqs"], list(npz["channels"])
    roi = [ch.index(c) for c in npz["roi"]]
    info = build_info(ch)
    fmask = (freqs >= FMIN) & (freqs <= FMAX)
    amask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    tmask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 0.9, 1], hspace=0.55, wspace=0.9)

    rel_alpha = {}
    for ds in DATASETS:
        psd = npz[f"{ds}__psd"]  # (n_sub, n_ch, n_freqs)
        rel_alpha[ds] = (psd[..., amask].sum(-1) / psd[..., tmask].sum(-1)).mean(axis=0)
    vmin = min(v.min() for v in rel_alpha.values())
    vmax = max(v.max() for v in rel_alpha.values())

    for col, ds in enumerate(DATASETS):
        psd = npz[f"{ds}__psd"]
        roi_psd = psd[:, roi].mean(axis=1)  # (n_sub, n_freqs)
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2])
        ax.axvspan(*ALPHA_BAND, color="0.92", zorder=0)
        for s in roi_psd:
            ax.loglog(freqs[fmask], s[fmask], color=COLOR[ds], lw=0.5, alpha=0.3)
        ax.loglog(freqs[fmask], roi_psd.mean(axis=0)[fmask], color="k", lw=1.8)
        ax.set_xticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
        ax.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax.set_ylabel("PSD (µV²/Hz)")
        sub = df[df["dataset"] == ds]
        ax.set_title(f"{COL_LABEL[ds]}\nposterior ROI, n = {len(sub)}; "
                     f"alpha peak in {sub['alpha_peak_present'].sum()}/{len(sub)}", fontsize=8.5)

        ax_t = fig.add_subplot(gs[1, 2 * col:2 * col + 2])
        im, _ = mne.viz.plot_topomap(rel_alpha[ds], info, axes=ax_t, show=False,
                                     vlim=(vmin, vmax), cmap="viridis", contours=4)
        ax_t.set_title("relative alpha power (8–13 / 2–40 Hz)", fontsize=8)
    cax = fig.add_axes([0.92, 0.40, 0.012, 0.17])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)

    ax_c = fig.add_subplot(gs[2, 0:3])
    strip(ax_c, df, "alpha_cf_hz", "Individual alpha frequency (Hz)")
    ax_d = fig.add_subplot(gs[2, 3:6])
    strip(ax_d, df, "aperiodic_exponent", "Aperiodic (1/f) exponent")

    for ax, letter, x in ((fig.axes[0], "a", -0.18), (fig.axes[1], "b", -0.35),
                          (ax_c, "c", -0.18), (ax_d, "d", -0.18)):
        ax.text(x, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")

    fig.suptitle("Resting spectral profile: power spectra, alpha topography and FOOOF parameters (n = 30)",
                 fontsize=11, y=0.99)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"Figure_Spectral.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
