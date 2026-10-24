"""Technical Validation #4b - Scalp topography of the VEP / AEP.

Companion to 04_evoked_potential_check.py. The ROI-averaged waveforms in that
script already show the expected P1-N1 (VEP) and N1-P2 (AEP) complexes at 2-4
picked channels; this script computes the full 32-channel grand-average evoked
response per dataset (same per-dataset reference choice, same epoching) and
plots scalp topomaps at each dataset's own peak negative/positive latency, to
confirm the expected spatial topography directly - occipital maximum for the
VEP (Dataset_A), fronto-central maximum for the AEP (Dataset_B/C) - rather than
relying on the picked channels alone.

The pipeline reproduces 04_evoked_potential_check.py exactly - epochs rejected
on the ROI channels only (peak-to-peak > REJECT_UV), and all retained epochs of
a subject pooled before averaging - so the ROI-mean grand average here is the
same waveform as in 04, and the peak latencies at which the topographies are
drawn are the latencies reported by 04.

Usage:
    py -3.12 04b_evoked_topomap.py [--clean]

--clean: interpolate BIDS bad channels and notch out the ~18 Hz line first
(common.clean_raw), as in 04_evoked_potential_check.py --clean; outputs get a
'_clean' suffix. The paper figure (04c) uses the --clean outputs.

Writes:
    derivatives/validation/04b_evoked_topomap_<dataset>[_clean].png
    derivatives/validation/04b_topomap_channel_data[_clean].csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS, clean_raw

mne.set_log_level("ERROR")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]
MODALITY = {"Dataset_A": "visual", "Dataset_B": "auditory", "Dataset_C": "auditory"}
REFERENCE_BY_DATASET = {"Dataset_A": "native", "Dataset_B": "linked_mastoids",
                         "Dataset_C": "linked_mastoids"}
ROI = {"visual": ["O1", "Oz", "O2", "POz"], "auditory": ["Fz", "Cz"]}

# Same stimulus-carrying cues as 04_evoked_potential_check.py.
STIM_TRIAL_TYPES = {"Dataset_A": ("rest", "left", "right"),
                    "Dataset_B": ("left", "right"),
                    "Dataset_C": ("left", "right")}

TMIN, TMAX = -0.2, 0.6
BASELINE = (None, 0)
REJECT_UV = 150.0
NEG_WINDOW = (0.06, 0.25)
POS_WINDOW = (0.05, 0.30)


def build_info():
    """Channel layout (order + 10-20 positions) shared by every recording."""
    info = mne.create_info(EEG_CHANNELS, sfreq=500.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def load_all_channel_sum(ds, sub_id, task, run, reference, roi, clean=False):
    """Return (sum over retained epochs (32, n_times) in uV, n_epochs, times),
    or (None, None) if the recording is missing or no epoch survives."""
    base = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}"
    vhdr = base.with_name(base.name + "_eeg.vhdr")
    events_tsv = base.with_name(base.name + "_events.tsv")
    if not vhdr.exists() or not events_tsv.exists():
        return None, None

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    raw.reorder_channels(EEG_CHANNELS)
    if clean:
        clean_raw(raw, ds, sub_id, task, run)
    if reference == "linked_mastoids":
        raw.set_eeg_reference(["M1", "M2"], verbose=False)
    # "native" (Dataset_A): keep the dataset's own CPz reference, no change.
    raw.filter(0.5, 30.0, verbose=False)

    ev_df = pd.read_csv(events_tsv, sep="\t")
    ev_df = ev_df[ev_df["trial_type"].isin(STIM_TRIAL_TYPES[ds])]
    events = np.column_stack([
        ev_df["sample"].to_numpy(dtype=int),
        np.zeros(len(ev_df), dtype=int),
        np.ones(len(ev_df), dtype=int),
    ])
    epochs = mne.Epochs(
        raw, events, event_id={"cue": 1}, tmin=TMIN, tmax=TMAX, baseline=BASELINE,
        preload=True, verbose=False,
    )
    data_uv = epochs.get_data(picks=EEG_CHANNELS, units="uV")  # (n_epochs, 32, n_times)
    # Reject exactly as 04_evoked_potential_check.py does: peak-to-peak on the
    # dataset's ROI channels only (so both scripts keep the same epochs).
    roi_idx = [EEG_CHANNELS.index(ch) for ch in roi]
    roi_p2p = np.ptp(data_uv[:, roi_idx], axis=-1)  # (n_epochs, n_roi)
    keep = (roi_p2p <= REJECT_UV).all(axis=1)
    if not keep.any():
        return None, None
    # Sum + count (not a per-recording mean), so that the subject average pools
    # epochs across recordings with equal weight, as 04 does.
    return data_uv[keep].sum(axis=0), int(keep.sum()), epochs.times


def pick_peak_time(waveform_uv, times, window, mode):
    mask = (times >= window[0]) & (times <= window[1])
    idx = np.argmin(waveform_uv[mask]) if mode == "neg" else np.argmax(waveform_uv[mask])
    return float(times[mask][idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    suffix = "_clean" if args.clean else ""
    subjects = [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    info = build_info()
    shared_times = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    channel_rows = []
    n_recordings = 0

    for ds, n_runs in DATASETS.items():
        reference = REFERENCE_BY_DATASET[ds]
        roi = ROI[MODALITY[ds]]
        subject_means = []
        for sub_id in subjects:
            total, n_total = None, 0
            for task in TASKS:
                for run in range(1, n_runs + 1):
                    res = load_all_channel_sum(ds, sub_id, task, run, reference, roi,
                                               clean=args.clean)
                    n_recordings += 1
                    if n_recordings % 50 == 0:
                        print(f"...{n_recordings} recordings processed")
                    if res[0] is None:
                        continue
                    s, n, times = res
                    if shared_times is None:
                        shared_times = times
                    total = s if total is None else total + s
                    n_total += n
            if n_total:
                subject_means.append(total / n_total)  # (32, n_times)

        grand = np.mean(subject_means, axis=0)  # (32, n_times)

        # Use the same ROI-mean grand-average waveform as 04_evoked_potential_check.py
        # to pick the peak latencies, for consistency with the reported numbers there.
        roi_idx = [EEG_CHANNELS.index(ch) for ch in roi]
        roi_wave = grand[roi_idx].mean(axis=0)
        t_neg = pick_peak_time(roi_wave, shared_times, NEG_WINDOW, "neg")
        t_pos = pick_peak_time(roi_wave, shared_times, POS_WINDOW, "pos")

        i_neg = np.argmin(np.abs(shared_times - t_neg))
        i_pos = np.argmin(np.abs(shared_times - t_pos))
        topo_neg = grand[:, i_neg]
        topo_pos = grand[:, i_pos]

        for ch, vneg, vpos in zip(EEG_CHANNELS, topo_neg, topo_pos):
            channel_rows.append(dict(dataset=ds, modality=MODALITY[ds], reference=reference,
                                      channel=ch, latency_neg_ms=round(t_neg * 1000, 1),
                                      amp_at_neg_uv=round(float(vneg), 3),
                                      latency_pos_ms=round(t_pos * 1000, 1),
                                      amp_at_pos_uv=round(float(vpos), 3)))

        vmax = max(np.abs(topo_neg).max(), np.abs(topo_pos).max())
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
        for ax, topo, t, label in [
            (axes[0], topo_neg, t_neg, "negative peak"),
            (axes[1], topo_pos, t_pos, "positive peak"),
        ]:
            im, _ = mne.viz.plot_topomap(topo, info, axes=ax, show=False,
                                          vlim=(-vmax, vmax), cmap="RdBu_r",
                                          names=roi, mask=np.isin(EEG_CHANNELS, roi),
                                          mask_params=dict(marker="o", markerfacecolor="none",
                                                            markeredgecolor="k", markersize=6))
            ax.set_title(f"{label}\n{t*1000:.0f} ms", fontsize=10)
        fig.suptitle(f"{ds} ({MODALITY[ds]}, ref={reference}) grand-average topography, n={len(subject_means)} subjects",
                     fontsize=10)
        fig.colorbar(im, ax=axes, shrink=0.8, label="uV")
        out_png = OUT_DIR / f"04b_evoked_topomap_{ds}{suffix}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{ds}: topomap saved to {out_png} (peaks at {t_neg*1000:.0f} ms / {t_pos*1000:.0f} ms)")

    ch_df = pd.DataFrame(channel_rows)
    ch_out = OUT_DIR / f"04b_topomap_channel_data{suffix}.csv"
    ch_df.to_csv(ch_out, index=False)
    print(f"\nPer-channel topography data written to {ch_out}")


if __name__ == "__main__":
    main()
