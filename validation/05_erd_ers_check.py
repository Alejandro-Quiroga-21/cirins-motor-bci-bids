"""Technical Validation #5 - Motor event-related (de)synchronisation (ERD/ERS).

Every movement cue (left/right) instructs the participant to execute a brief
movement (hand for task-upper, foot for task-lower). If the EEG, the event
timing and the channel labelling are correct, executed movements should produce
the canonical sensorimotor pattern (Pfurtscheller & Lopes da Silva, 1999):
  - mu (8-13 Hz) and beta (13-30 Hz) power decrease (ERD) during movement,
    over the sensorimotor cortex: lateralised to the contralateral hemisphere
    (C3 for right hand, C4 for left hand) for hand movements, and focal at the
    vertex (Cz) for foot movements;
  - a post-movement beta rebound (ERS) above baseline after the movement ends.
Only movement cues are analysed. Rest cues are not usable as a "no-ERD" control
in this paradigm: every rest cue starts exactly when the preceding movement cue
ends, so a rest-locked epoch's pre-cue baseline falls inside the movement and
its post-cue window contains the post-movement beta rebound (verified on a pilot:
+55-80% beta ERS at Cz after foot movements). The rebound is instead captured in
the movement-locked epochs (REBOUND_WINDOW below).

Method: for every recording, a small Hjorth Laplacian is derived at C3, Cz and C4
(each channel minus the mean of its four nearest neighbours; being a linear
combination with zero-sum weights it is independent of the native CPz
reference, which itself is not a recorded channel). The Laplacian signals are
band-pass filtered 1-45 Hz and epoched around every cue onset (using the
'sample' column of _events.tsv) from -3 s to +7 s. Artifact rejection is
relative, per recording: an epoch is dropped if its peak-to-peak amplitude on
any Laplacian channel exceeds REJECT_FACTOR times that channel's median
peak-to-peak across the recording's epochs. (A fixed microvolt threshold does
not transfer to Laplacian signals, whose typical epoch p-p is only ~15 uV: a
200 uV cut rejected 0.1% of epochs and let single transients of 10-35x the
median through, which dominated some subjects' average power by >1000%.)
Time-frequency power is computed with Morlet wavelets (4-40 Hz, n_cycles = f/2), averaged over epochs
per (dataset, subject, task, cue, channel), cropped to -2..+6 s to discard edge
effects, and expressed as ERD/ERS % relative to the pre-cue baseline
(-1.5..-0.5 s): negative = ERD, positive = ERS. Per-subject ERD/ERS maps are
averaged into grand averages per dataset/task/cue.

Summary metrics (per subject and grand average), with a priori windows:
  - ERD: mean mu / beta ERD% over MOVE_WINDOW (0.5-2.5 s, cue on screen/beep
    to end of the 2.5 s movement cue);
  - beta rebound: mean beta ERS% over REBOUND_WINDOW (3.0-5.0 s, after the
    movement cue ends and before the next cue).
For hand movements the contralateral/ipsilateral channel pair is reported
(C3/C4 depending on side), and a lateralisation check (contralateral ERD
stronger than ipsilateral) is summarised across subjects.

Usage:
    py -3.12 05_erd_ers_check.py [--subjects sub-01 sub-02 ...] [--clean]

--clean (robustness check): channels marked bad in the BIDS _channels.tsv are
interpolated and the ~18 Hz line is notched out before the Laplacian
(common.clean_raw); outputs get a '_clean' suffix. A partial run (--subjects)
writes '_partial' outputs and never overwrites the full-corpus files.

Writes:
    derivatives/validation/05_erd_ers_subject_summary.csv
    derivatives/validation/05_erd_ers_grand_average_summary.csv
    derivatives/validation/05_erd_ers_tfr.npz   (per-subject ERD% maps, for figures)
"""
import argparse

import mne
import numpy as np
import pandas as pd

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS, clean_raw

mne.set_log_level("ERROR")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]

# Hjorth small-Laplacian neighbours (CPz, the native reference, is not a channel).
LAPLACIAN = {
    "C3": ["FC5", "FC1", "CP5", "CP1"],
    "Cz": ["FC1", "FC2", "CP1", "CP2"],
    "C4": ["FC2", "FC6", "CP2", "CP6"],
}
LAP_CHANNELS = list(LAPLACIAN)
CUES = ["left", "right"]  # movement cues only (see docstring)

EPOCH_TMIN, EPOCH_TMAX = -3.0, 7.0  # padded for wavelet edge effects
CROP_TMIN, CROP_TMAX = -2.0, 6.0
BASELINE = (-1.5, -0.5)
REJECT_FACTOR = 5.0  # x per-channel median epoch peak-to-peak (see docstring)
FREQS = np.arange(4.0, 41.0, 1.0)
N_CYCLES = FREQS / 2.0
DECIM = 10  # 500 Hz -> 50 Hz time resolution of the TFR

MU_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
MOVE_WINDOW = (0.5, 2.5)
REBOUND_WINDOW = (3.0, 5.0)


def laplacian_raw(raw):
    """Return a new RawArray with the three Hjorth-Laplacian derivations."""
    data = raw.get_data()
    idx = {ch: i for i, ch in enumerate(raw.ch_names)}
    lap = np.vstack([
        data[idx[c]] - data[[idx[n] for n in nbrs]].mean(axis=0)
        for c, nbrs in LAPLACIAN.items()
    ])
    info = mne.create_info(LAP_CHANNELS, raw.info["sfreq"], ch_types="eeg")
    return mne.io.RawArray(lap, info, verbose=False)


def recording_power(ds, sub_id, task, run, clean=False):
    """Return {cue: (power_sum, n_epochs)} for one recording, where power_sum is
    the summed (not yet averaged) Morlet power (n_ch, n_freqs, n_times), plus
    the TFR times vector and (n_total, n_rejected) epoch counts. None if missing.
    """
    base = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}"
    vhdr = base.with_name(base.name + "_eeg.vhdr")
    events_tsv = base.with_name(base.name + "_events.tsv")
    if not vhdr.exists() or not events_tsv.exists():
        return None

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    if clean:
        clean_raw(raw, ds, sub_id, task, run)
    lap = laplacian_raw(raw)
    lap.filter(1.0, 45.0, verbose=False)

    ev_df = pd.read_csv(events_tsv, sep="\t")
    ev_df = ev_df[ev_df["trial_type"].isin(CUES)]
    code = {c: i + 1 for i, c in enumerate(CUES)}
    events = np.column_stack([
        ev_df["sample"].to_numpy(dtype=int),
        np.zeros(len(ev_df), dtype=int),
        ev_df["trial_type"].map(code).to_numpy(dtype=int),
    ])
    present = {c: code[c] for c in CUES if code[c] in events[:, 2]}
    epochs = mne.Epochs(
        lap, events, event_id=present, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False,
    )
    n_total = len(events)
    data = epochs.get_data()
    p2p = data.max(axis=-1) - data.min(axis=-1)  # (n_ep, n_ch)
    bad = (p2p > REJECT_FACTOR * np.median(p2p, axis=0)).any(axis=1)
    epochs.drop(np.flatnonzero(bad), reason="RELATIVE_P2P", verbose=False)
    n_kept_by_cue = {c: len(epochs[c]) if c in epochs.event_id else 0 for c in present}
    n_rejected = n_total - len(epochs)  # relative-p2p rejects (+ any epoch running past the recording)

    out = {}
    times = None
    for cue in present:
        if n_kept_by_cue[cue] == 0:
            continue
        data = epochs[cue].get_data()  # (n_ep, n_ch, n_times), V
        power = mne.time_frequency.tfr_array_morlet(
            data, sfreq=lap.info["sfreq"], freqs=FREQS, n_cycles=N_CYCLES,
            output="power", decim=DECIM, verbose=False,
        )  # (n_ep, n_ch, n_freqs, n_times)
        t = epochs.times[::DECIM]
        keep = (t >= CROP_TMIN - 1e-9) & (t <= CROP_TMAX + 1e-9)
        out[cue] = (power[..., keep].sum(axis=0), power.shape[0])
        times = t[keep]
    return out, times, n_total, n_rejected


def to_erd_percent(mean_power, times):
    """ERD/ERS % relative to the BASELINE window, per channel and frequency."""
    bmask = (times >= BASELINE[0]) & (times <= BASELINE[1])
    ref = mean_power[..., bmask].mean(axis=-1, keepdims=True)
    return 100.0 * (mean_power - ref) / ref


def band_window_mean(erd, times, band, window, ch):
    fmask = (FREQS >= band[0]) & (FREQS <= band[1])
    tmask = (times >= window[0]) & (times <= window[1])
    return float(erd[LAP_CHANNELS.index(ch)][fmask][:, tmask].mean())


def summary_metrics(erd, times, task, cue):
    """Mu/beta ERD in the movement window and beta ERS in the rebound window at
    C3, Cz, C4, plus contra/ipsi for hand movements."""
    row = {}
    for ch in LAP_CHANNELS:
        row[f"mu_erd_{ch}"] = band_window_mean(erd, times, MU_BAND, MOVE_WINDOW, ch)
        row[f"beta_erd_{ch}"] = band_window_mean(erd, times, BETA_BAND, MOVE_WINDOW, ch)
        row[f"beta_rebound_{ch}"] = band_window_mean(erd, times, BETA_BAND, REBOUND_WINDOW, ch)
    if task == "upper" and cue in ("left", "right"):
        contra, ipsi = ("C3", "C4") if cue == "right" else ("C4", "C3")
        for m in ("mu_erd", "beta_erd", "beta_rebound"):
            row[f"{m}_contra"] = row[f"{m}_{contra}"]
            row[f"{m}_ipsi"] = row[f"{m}_{ipsi}"]
    return {k: round(v, 2) for k, v in row.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--clean", action="store_true",
                        help="Robustness check: interpolate BIDS bad channels and notch out "
                             "the ~18 Hz line (common.clean_raw); outputs get '_clean'.")
    args = parser.parse_args()
    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    suffix = ("_clean" if args.clean else "") + ("_partial" if args.subjects else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_rows = []
    subject_erd = {}  # (ds, task, cue, sub_id) -> ERD% (n_ch, n_freqs, n_times)
    times = None
    n_recordings = 0
    epoch_total = epoch_rejected = 0

    for ds, n_runs in DATASETS.items():
        for sub_id in subjects:
            for task in TASKS:
                acc = {}  # cue -> [power_sum, n_epochs]
                for run in range(1, n_runs + 1):
                    res = recording_power(ds, sub_id, task, run, clean=args.clean)
                    n_recordings += 1
                    if n_recordings % 50 == 0:
                        print(f"...{n_recordings} recordings processed", flush=True)
                    if res is None:
                        continue
                    per_cue, t, n_tot, n_rej = res
                    epoch_total += n_tot
                    epoch_rejected += n_rej
                    if t is not None:
                        times = t
                    for cue, (psum, n) in per_cue.items():
                        if cue not in acc:
                            acc[cue] = [np.zeros_like(psum), 0]
                        acc[cue][0] += psum
                        acc[cue][1] += n

                for cue, (psum, n) in acc.items():
                    erd = to_erd_percent(psum / n, times)
                    subject_erd[(ds, task, cue, sub_id)] = erd
                    row = dict(dataset=ds, subject=sub_id, task=task, cue=cue, n_epochs=n)
                    row.update(summary_metrics(erd, times, task, cue))
                    subject_rows.append(row)

    print(f"\nEpochs: {epoch_total} total, {epoch_rejected} rejected "
          f"({100 * epoch_rejected / max(epoch_total, 1):.1f}%) at {REJECT_FACTOR}x median p-p")

    subj_df = pd.DataFrame(subject_rows)
    subj_out = OUT_DIR / f"05_erd_ers_subject_summary{suffix}.csv"
    subj_df.to_csv(subj_out, index=False)
    print(f"Per-subject summary written to {subj_out}")

    # Grand averages (mean of per-subject ERD% maps).
    grand_rows = []
    npz = {"times": times, "freqs": FREQS, "channels": np.array(LAP_CHANNELS)}
    for (ds, task, cue) in sorted({k[:3] for k in subject_erd}):
        keys = sorted(k for k in subject_erd if k[:3] == (ds, task, cue))
        stack = np.stack([subject_erd[k] for k in keys])  # (n_sub, n_ch, n_f, n_t)
        tag = f"{ds}__{task}__{cue}"
        npz[f"{tag}__erd"] = stack
        npz[f"{tag}__subjects"] = np.array([k[3] for k in keys])
        grand = stack.mean(axis=0)
        row = dict(dataset=ds, task=task, cue=cue, n_subjects=len(keys))
        row.update(summary_metrics(grand, times, task, cue))
        sub = subj_df[(subj_df.dataset == ds) & (subj_df.task == task) & (subj_df.cue == cue)]
        if "mu_erd_contra" in sub and sub["mu_erd_contra"].notna().any():
            row["pct_subjects_mu_contra_stronger"] = round(
                100 * (sub["mu_erd_contra"] < sub["mu_erd_ipsi"]).mean(), 1)
        # Expected site: Cz for foot, contralateral C3/C4 for hand.
        ch = "Cz" if task == "lower" else ("C3" if cue == "right" else "C4")
        row["expected_site"] = ch
        row["pct_subjects_mu_erd_negative"] = round(100 * (sub[f"mu_erd_{ch}"] < 0).mean(), 1)
        row["pct_subjects_beta_erd_negative"] = round(100 * (sub[f"beta_erd_{ch}"] < 0).mean(), 1)
        row["pct_subjects_beta_rebound_positive"] = round(
            100 * (sub[f"beta_rebound_{ch}"] > 0).mean(), 1)
        grand_rows.append(row)

    np.savez_compressed(OUT_DIR / f"05_erd_ers_tfr{suffix}.npz", **npz)
    ga_df = pd.DataFrame(grand_rows)
    ga_out = OUT_DIR / f"05_erd_ers_grand_average_summary{suffix}.csv"
    ga_df.to_csv(ga_out, index=False)
    print(f"Grand-average summary written to {ga_out}\n")
    cols = ["dataset", "task", "cue", "n_subjects",
            "mu_erd_C3", "mu_erd_Cz", "mu_erd_C4",
            "beta_erd_C3", "beta_erd_Cz", "beta_erd_C4",
            "beta_rebound_C3", "beta_rebound_Cz", "beta_rebound_C4"]
    print(ga_df[cols].to_string(index=False))
    extra = [c for c in ga_df.columns if c.startswith("pct_")]
    print("\n" + ga_df[["dataset", "task", "cue"] + extra].to_string(index=False))


if __name__ == "__main__":
    main()
