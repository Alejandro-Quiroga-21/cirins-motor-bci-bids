"""Technical Validation #2b - Resting spectral profile (1/f + alpha peak).

Complements the per-channel signal-quality metrics (03) with a physiological
check of the power spectrum: genuine scalp EEG shows an aperiodic 1/f-like
background plus an alpha (~8-12 Hz) peak that is largest over posterior
(parieto-occipital) sites. Neither survives a mislabelled montage, a wrong
amplitude scale in the spectrum's shape, or a non-EEG signal.

Method: for every recording, the signal (32 scalp channels) is re-referenced to
the common average. Resting segments are taken from the rest-cue intervals of
_events.tsv, starting REST_SKIP_S after each rest-cue onset (every rest cue
starts right after a movement cue ends, and the post-movement beta rebound
lasts ~2-2.5 s into the rest interval - see 05_erd_ers_check.py) and ending at
the rest cue's end. Segments whose 1-45 Hz peak-to-peak amplitude on any
channel exceeds REJECT_FACTOR x that channel's median across the recording's
segments are dropped. Welch PSDs (2 s windows, 50% overlap, linear detrend,
0.5 Hz resolution) are averaged over all retained windows per (dataset,
subject), pooling tasks and runs.

The posterior-ROI PSD of every subject is then parameterised with FOOOF
(Donoghue et al., 2020, Nat Neurosci; 2-40 Hz, 'fixed' aperiodic mode), giving
the aperiodic exponent (1/f slope), the goodness of fit, and the alpha peak
(strongest periodic peak with centre frequency in ALPHA_SEARCH): individual
alpha frequency (IAF) and peak power above the aperiodic fit.

Narrowband ~18 Hz interference: a spectral line far narrower than any EEG
rhythm (FWHM at the 0.1 Hz resolution limit), at a slightly drifting frequency
between ~17.5 and ~19.5 Hz, is present in most recordings of all three
datasets. It is continuous and stable within a recording, has no 2nd harmonic,
and appears on all channels at a similar sub-microvolt amplitude (largest at
mastoid/lateral sites), i.e. an environmental/instrumental interference, not
brain activity. It is quantified per subject here (LINE_SEARCH, whole-recording
high-resolution Welch on all channels: frequency, prominence over the local
background, and sinusoid-equivalent amplitude in uV). Because FOOOF would fit
it as a "beta peak", no beta-peak metric is reported.

Note: Dataset_C rest intervals include the ~4.9 s mirror-animation (sham
feedback) shown after each movement cue, i.e. the participant is watching a
moving visual stimulus for part of the segment; Dataset_A shows a static rest
cue on screen; Dataset_B has no visual stimulus. Posterior alpha is expected
to differ accordingly, so results are reported per dataset.

Usage:
    py -3.12 06_spectral_check.py [--subjects sub-01 sub-02 ...]

Writes:
    derivatives/validation/06_spectral_subject_summary.csv
    derivatives/validation/06_spectral_psd.npz   (per-subject, per-channel PSDs, for figures)
"""
import argparse
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # fooof 1.1 announces its rename to specparam
    from fooof import FOOOF

from common import cues_only
from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS

mne.set_log_level("ERROR")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]
POSTERIOR_ROI = ["P3", "Pz", "P4", "POz", "O1", "Oz", "O2"]

REST_SKIP_S = 2.5
MIN_SEGMENT_S = 2.0      # at least one full Welch window
LAST_REST_S = 5.0        # assumed length of a run's final rest (no end marker)
WELCH_NPERSEG_S = 2.0
REJECT_FACTOR = 5.0
FMAX_SAVE = 60.0

FOOOF_RANGE = (2.0, 40.0)
FOOOF_SETTINGS = dict(peak_width_limits=(1.0, 8.0), max_n_peaks=6,
                      min_peak_height=0.1, aperiodic_mode="fixed", verbose=False)
ALPHA_SEARCH = (7.0, 14.0)

LINE_SEARCH = (16.0, 21.0)   # narrowband interference (see docstring)
LINE_NPERSEG_S = 10.0        # 0.1 Hz resolution
LINE_PROMINENCE_MIN = 3.0    # peak / local-background median to count as present
LINE_DETECT_CHANNELS = ["CP1", "CP2", "Cz", "Pz"]  # see line_interference()


def recording_segments_psd(ds, sub_id, task, run):
    """Return (freqs, psd_sum (n_ch, n_freqs) in uV^2/Hz summed over windows,
    n_windows, rest_seconds, n_segments_total, n_segments_rejected, line) or
    None, where line = line_interference() of the whole recording."""
    base = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}"
    vhdr = base.with_name(base.name + "_eeg.vhdr")
    events_tsv = base.with_name(base.name + "_events.tsv")
    if not vhdr.exists() or not events_tsv.exists():
        return None

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    sfreq = raw.info["sfreq"]
    line = line_interference(raw.get_data(units="uV"), sfreq)  # native reference
    raw.set_eeg_reference("average", verbose=False)
    data = raw.get_data(units="uV")
    data_bp = raw.copy().filter(1.0, 45.0, verbose=False).get_data(units="uV")

    ev = cues_only(pd.read_csv(events_tsv, sep="\t"))
    rest = ev[ev["trial_type"] == "rest"]
    bounds = []
    for r in rest.itertuples():
        start = int(round((r.onset + REST_SKIP_S) * sfreq))
        # The last rest of a run has no following cue (duration n/a): cap it at the
        # programmed minimum rest length.
        dur = r.duration if pd.notna(r.duration) else LAST_REST_S
        stop = min(int(round((r.onset + dur) * sfreq)), data.shape[1])
        if stop - start >= MIN_SEGMENT_S * sfreq:
            bounds.append((start, stop))
    if not bounds:
        return None

    p2p = np.array([np.ptp(data_bp[:, a:b], axis=1) for a, b in bounds])  # (n_seg, n_ch)
    bad = (p2p > REJECT_FACTOR * np.median(p2p, axis=0)).any(axis=1)

    nperseg = int(WELCH_NPERSEG_S * sfreq)
    psd_sum, n_win, freqs, rest_s = None, 0, None, 0.0
    for (a, b), is_bad in zip(bounds, bad):
        if is_bad:
            continue
        seg = data[:, a:b]
        freqs, p = welch(seg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2,
                         detrend="linear", axis=-1)
        k = 1 + (seg.shape[1] - nperseg) // (nperseg // 2)  # windows in this segment
        psd_sum = p * k if psd_sum is None else psd_sum + p * k
        n_win += k
        rest_s += (b - a) / sfreq
    if psd_sum is None:
        return None
    keep = freqs <= FMAX_SAVE
    return freqs[keep], psd_sum[:, keep], n_win, rest_s, len(bounds), int(bad.sum()), line


def fooof_metrics(freqs, psd):
    fm = FOOOF(**FOOOF_SETTINGS)
    fm.fit(freqs, psd, list(FOOOF_RANGE))
    offset, exponent = fm.aperiodic_params_
    peaks = fm.peak_params_  # (n_peaks, 3): CF, PW, BW
    row = dict(aperiodic_offset=round(float(offset), 3),
               aperiodic_exponent=round(float(exponent), 3),
               fooof_r2=round(float(fm.r_squared_), 4),
               fooof_error=round(float(fm.error_), 4))
    lo, hi = ALPHA_SEARCH
    in_band = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)] if len(peaks) else peaks
    if len(in_band):
        cf, pw, bw = in_band[np.argmax(in_band[:, 1])]
        row.update(alpha_peak_present=True, alpha_cf_hz=round(float(cf), 2),
                   alpha_pw_log10=round(float(pw), 3), alpha_bw_hz=round(float(bw), 2))
    else:
        row.update(alpha_peak_present=False, alpha_cf_hz=np.nan,
                   alpha_pw_log10=np.nan, alpha_bw_hz=np.nan)
    return row


def line_interference(data_uv, sfreq):
    """Narrowband interference in LINE_SEARCH on a whole recording (native
    reference): (frequency Hz, prominence, median sinusoid amplitude uV).
    Frequency and prominence are taken from the mean PSD of LINE_DETECT_CHANNELS
    - the neighbours of the CPz reference, whose EEG background is smallest, so a
    sub-microvolt line common to all channels stands out most there (averaging
    prominence over all 32 channels dilutes it). Amplitude is the median over all
    channels of the sinusoid-equivalent amplitude of the excess power at the peak.
    """
    nperseg = int(LINE_NPERSEG_S * sfreq)
    f, p = welch(data_uv, fs=sfreq, nperseg=nperseg, detrend="linear", axis=-1)
    m = (f >= LINE_SEARCH[0]) & (f <= LINE_SEARCH[1])
    det = p[[EEG_CHANNELS.index(c) for c in LINE_DETECT_CHANNELS]].mean(axis=0)
    rel = det[m] / np.median(det[m])
    j = int(np.argmax(rel))
    f0 = f[m][j]
    pk = (f >= f0 - 0.15) & (f <= f0 + 0.15)
    bg = ((f >= f0 - 3) & (f <= f0 - 1)) | ((f >= f0 + 1) & (f <= f0 + 3))
    excess = (p[:, pk] - np.median(p[:, bg], axis=1, keepdims=True)).clip(0).sum(axis=1) * (f[1] - f[0])
    return float(f0), float(rel[j]), float(np.median(np.sqrt(2 * excess)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    suffix = "_partial" if args.subjects else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roi_idx = [EEG_CHANNELS.index(c) for c in POSTERIOR_ROI]
    rows, psds, line_rows = [], {}, []
    freqs = None
    n_recordings = seg_total = seg_rejected = 0

    for ds, n_runs in DATASETS.items():
        for sub_id in subjects:
            acc, n_win, rest_s = None, 0, 0.0
            for task in TASKS:
                for run in range(1, n_runs + 1):
                    res = recording_segments_psd(ds, sub_id, task, run)
                    n_recordings += 1
                    if n_recordings % 50 == 0:
                        print(f"...{n_recordings} recordings processed", flush=True)
                    if res is None:
                        continue
                    freqs, psum, k, secs, n_seg, n_bad, (f0, prom, amp) = res
                    line_rows.append(dict(dataset=ds, subject=sub_id, task=task, run=run,
                                          line_freq_hz=round(f0, 2), line_prominence=round(prom, 2),
                                          line_amplitude_uv=round(amp, 3),
                                          line_present=prom >= LINE_PROMINENCE_MIN))
                    rest_s += secs
                    acc = psum if acc is None else acc + psum
                    n_win += k
                    seg_total += n_seg
                    seg_rejected += n_bad
            if acc is None:
                continue
            psd = acc / n_win  # (n_ch, n_freqs), uV^2/Hz
            psds[(ds, sub_id)] = psd
            row = dict(dataset=ds, subject=sub_id, n_welch_windows=n_win,
                       rest_seconds=round(rest_s, 1))
            row.update(fooof_metrics(freqs, psd[roi_idx].mean(axis=0)))
            rows.append(row)

    print(f"\nRest segments: {seg_total} total, {seg_rejected} rejected "
          f"({100 * seg_rejected / max(seg_total, 1):.1f}%) at {REJECT_FACTOR}x median p-p")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"06_spectral_subject_summary{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Per-subject summary written to {out_csv}\n")

    npz = {"freqs": freqs, "channels": np.array(EEG_CHANNELS), "roi": np.array(POSTERIOR_ROI)}
    for ds in DATASETS:
        keys = sorted(k for k in psds if k[0] == ds)
        if keys:
            npz[f"{ds}__psd"] = np.stack([psds[k] for k in keys])
            npz[f"{ds}__subjects"] = np.array([k[1] for k in keys])
    np.savez_compressed(OUT_DIR / f"06_spectral_psd{suffix}.npz", **npz)

    summary = df.groupby("dataset").agg(
        n_subjects=("subject", "size"),
        pct_alpha_peak=("alpha_peak_present", lambda s: round(100 * s.mean(), 1)),
        iaf_mean=("alpha_cf_hz", "mean"), iaf_sd=("alpha_cf_hz", "std"),
        alpha_pw_mean=("alpha_pw_log10", "mean"),
        exponent_mean=("aperiodic_exponent", "mean"), exponent_sd=("aperiodic_exponent", "std"),
        r2_median=("fooof_r2", "median"), r2_min=("fooof_r2", "min"),
        rest_s_median=("rest_seconds", "median"),
    ).round(3)
    print(summary.to_string())
    print("\nSubjects without a detected alpha peak:")
    print(df.loc[~df["alpha_peak_present"], ["dataset", "subject", "aperiodic_exponent", "fooof_r2"]]
          .to_string(index=False) or "  none")

    lines = pd.DataFrame(line_rows)
    lines_out = OUT_DIR / f"06_line_interference_18hz{suffix}.csv"
    lines.to_csv(lines_out, index=False)
    present = lines[lines["line_present"]]
    print(f"\nNarrowband {LINE_SEARCH[0]:.0f}-{LINE_SEARCH[1]:.0f} Hz interference "
          f"(prominence >= {LINE_PROMINENCE_MIN}x): {len(present)} / {len(lines)} recordings "
          f"({100 * len(present) / max(len(lines), 1):.1f}%), "
          f"{present['subject'].nunique()} / {lines['subject'].nunique()} subjects")
    if len(present):
        print("  frequency: median %.2f Hz (range %.2f-%.2f); amplitude: median %.2f uV (max %.2f)" % (
            present["line_freq_hz"].median(), present["line_freq_hz"].min(), present["line_freq_hz"].max(),
            present["line_amplitude_uv"].median(), present["line_amplitude_uv"].max()))
        print(lines.groupby("dataset")["line_present"].mean().mul(100).round(1).to_string())
    print(f"Per-recording table written to {lines_out}")


if __name__ == "__main__":
    main()
