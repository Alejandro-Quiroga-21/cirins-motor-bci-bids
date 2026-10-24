"""Technical Validation #2 - Generic per-channel signal-quality metrics.

For every recording x EEG channel (32 scalp channels; TRIG/STATUS excluded),
computes:
  - variance / std (uV)
  - flat-channel flag (std below a near-dead threshold)
  - longest run of consecutive identical samples (amplifier saturation /
    disconnection proxy) and the % of samples inside such runs
  - % of samples with |amplitude| beyond a generous artifact threshold,
    computed on a 0.5 Hz high-pass filtered copy of the signal (see note)
  - std of that high-passed signal (uV), alongside the raw std
  - line-noise ratio: power in a narrow band around 50 Hz relative to the
    total 1-100 Hz power (Welch PSD)

Note on the amplitude threshold: the BIDS data are raw/unfiltered and keep the
electrode-skin DC offset (several mV), so on the raw signal nearly every sample
exceeds +/-150 uV (18553/19200 channel-recordings at ~100%) regardless of any
artifact. The threshold is therefore applied after removing the offset and slow
drifts (0.5 Hz high-pass), which is what the metric is meant to capture:
genuine large-amplitude artifacts (movement, electrode pops, blinks at
frontopolar sites).

Writes one row per (recording, channel) to
derivatives/validation/03_signal_quality_summary.csv, plus prints the worst
offenders per metric (this is a descriptive/QC report, not a pass/fail
gate - unlike the structural checks, signal quality is a spectrum).

Usage:
    py -3.12 03_signal_quality_check.py [--subjects sub-01 sub-02 ...]
"""
import argparse
import itertools

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS

mne.set_log_level("ERROR")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]

FLAT_STD_UV = 0.5          # near-dead channel threshold
SATURATION_RUN_SAMPLES = 25  # ~50 ms at 500 Hz of an unchanging value
AMPLITUDE_THRESHOLD_UV = 150.0
HIGHPASS_HZ = 0.5          # offset/drift removal before the amplitude threshold
LINE_FREQ = 50.0
LINE_HALF_BAND = 1.0       # +/- 1 Hz around 50 Hz


def longest_constant_run(x):
    """Length of the longest run of consecutive identical values in x."""
    if len(x) == 0:
        return 0
    changes = np.flatnonzero(np.diff(x) != 0)
    run_starts = np.concatenate(([0], changes + 1))
    run_ends = np.concatenate((changes, [len(x) - 1]))
    return int((run_ends - run_starts + 1).max())


def line_noise_ratio(x, sfreq):
    freqs, psd = welch(x, fs=sfreq, nperseg=min(2048, len(x)))
    total_mask = (freqs >= 1) & (freqs <= 100)
    line_mask = (freqs >= LINE_FREQ - LINE_HALF_BAND) & (freqs <= LINE_FREQ + LINE_HALF_BAND)
    total_power = psd[total_mask].sum()
    if total_power <= 0:
        return np.nan
    return float(psd[line_mask].sum() / total_power)


def check_recording(ds, sub_id, task, run):
    base = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}_eeg.vhdr"
    if not base.exists():
        return []

    raw = mne.io.read_raw_brainvision(base, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    sfreq = raw.info["sfreq"]
    data_uv = raw.get_data(units="uV")  # (n_channels, n_samples)
    hp_uv = raw.copy().filter(HIGHPASS_HZ, None, verbose=False).get_data(units="uV")

    rows = []
    for ch_name, x, x_hp in zip(raw.ch_names, data_uv, hp_uv):
        std = float(np.std(x))
        longest_run = longest_constant_run(x)
        pct_out_of_range = float(100 * np.mean(np.abs(x_hp) > AMPLITUDE_THRESHOLD_UV))
        ratio = line_noise_ratio(x, sfreq)
        rows.append(dict(
            dataset=ds, subject=sub_id, task=task, run=run, channel=ch_name,
            n_samples=len(x), std_uv=round(std, 3), std_uv_hp=round(float(np.std(x_hp)), 3),
            is_flat=std < FLAT_STD_UV,
            longest_constant_run=longest_run,
            has_saturation_run=longest_run >= SATURATION_RUN_SAMPLES,
            pct_samples_over_150uv_hp=round(pct_out_of_range, 3),
            line_noise_ratio_50hz=round(ratio, 4) if not np.isnan(ratio) else np.nan,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    n_recordings = 0
    for ds, n_runs in DATASETS.items():
        for sub_id in subjects:
            for task in TASKS:
                for run in range(1, n_runs + 1):
                    all_rows.extend(check_recording(ds, sub_id, task, run))
                    n_recordings += 1
                    if n_recordings % 50 == 0:
                        print(f"...{n_recordings} recordings processed")

    report = pd.DataFrame(all_rows)
    # A partial run (--subjects) must not overwrite the full-corpus summary.
    suffix = "_partial" if args.subjects else ""
    out_csv = OUT_DIR / f"03_signal_quality_summary{suffix}.csv"
    report.to_csv(out_csv, index=False)

    print(f"\nProcessed {n_recordings} recordings x {len(EEG_CHANNELS)} channels "
          f"= {len(report)} channel-recordings.")
    print(f"Written to {out_csv}\n")

    print("Flat channels (std < %.1f uV): %d / %d (%.3f%%)" % (
        FLAT_STD_UV, report["is_flat"].sum(), len(report),
        100 * report["is_flat"].mean()))
    print("Recordings with a saturation run (>= %d samples constant): %d" % (
        SATURATION_RUN_SAMPLES,
        report.loc[report["has_saturation_run"], ["dataset", "subject", "task", "run"]]
        .drop_duplicates().shape[0]))
    print("\nMedian std: raw %.2f uV, %.1f Hz high-passed %.2f uV" % (
        report["std_uv"].median(), HIGHPASS_HZ, report["std_uv_hp"].median()))
    pct = report["pct_samples_over_150uv_hp"]
    print("%% samples > %.0f uV (after %.1f Hz high-pass): median %.3f%%, 95th pct %.3f%%; "
          "channel-recordings > 1%%: %d / %d, > 5%%: %d" % (
              AMPLITUDE_THRESHOLD_UV, HIGHPASS_HZ, pct.median(), pct.quantile(0.95),
              (pct > 1).sum(), len(pct), (pct > 5).sum()))
    by_ch = pct.groupby(report["channel"]).agg(
        median="median", p95=lambda s: s.quantile(0.95), mean="mean")
    print("Channels with highest 95th percentile:")
    print(by_ch.round(3).sort_values("p95", ascending=False).head(8).to_string())
    print("\nTop 10 channel-recordings by %% samples > %.0f uV (high-passed):" % AMPLITUDE_THRESHOLD_UV)
    print(report.nlargest(10, "pct_samples_over_150uv_hp")
          [["dataset", "subject", "task", "run", "channel", "pct_samples_over_150uv_hp"]]
          .to_string(index=False))
    print("\nTop 10 channel-recordings by 50 Hz line-noise ratio:")
    print(report.nlargest(10, "line_noise_ratio_50hz")
          [["dataset", "subject", "task", "run", "channel", "line_noise_ratio_50hz"]]
          .to_string(index=False))
    print("\nMedian 50 Hz line-noise ratio across corpus: %.4f" %
          report["line_noise_ratio_50hz"].median())


if __name__ == "__main__":
    main()
