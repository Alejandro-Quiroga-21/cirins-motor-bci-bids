"""Technical Validation #4 - Sensory evoked-potential check (VEP / AEP).

In Dataset_A (calibration) every recorded event (rest/left/right cue) is the
onset of an image on the screen, i.e. a visual stimulus; the movement cues
(arrows) were also accompanied by the 1,000 Hz beep (see 04d_dataset_a_audiovisual.py). In Dataset_B and
Dataset_C (open loop / sham feedback) the beep is delivered only at the onset
of the movement cues; the rest rows carry no sound. If the EEG signal, its
event timing and its channel labelling are all correct, epoching around the
onsets that carry a stimulus (STIM_TRIAL_TYPES) and averaging should reveal
the corresponding canonical sensory evoked potential:
  - Dataset_A (visual): a visual evoked potential (VEP) over occipital sites
    (O1, Oz, O2, POz), expected within roughly 60-250 ms post-onset.
  - Dataset_B/Dataset_C (auditory beep): an auditory N1-P2 complex, largest at
    fronto-central sites (Fz, Cz), expected within roughly 60-300 ms post-onset.

This is a descriptive/QC check (a real evoked potential should be visible in the
grand average and in most individual subjects), not a strict pass/fail gate -
unlike a structural check, "signal looks physiologically real" is a spectrum.

Method: for every recording, the raw signal (all 32 scalp channels) is
re-referenced to the common average (the dataset's native reference, CPz, sits
right next to the auditory ROI - Fz/Cz - and distorts/cancels evoked activity
there through simple subtraction; average reference is standard practice for
scalp ERP topography and does not privilege any single site), band-pass
filtered (0.5-30 Hz, typical for early sensory ERPs), then restricted to the
region-of-interest (ROI) channels for its dataset. Epochs are built around every
stimulus onset (all cues in Dataset_A, movement cues only in Dataset_B/C; using the 'sample' column already present in _events.tsv - no
re-detection needed) from -0.2 s to +0.6 s, baseline-corrected on the
pre-stimulus window, and epochs with a peak-to-peak amplitude beyond REJECT_UV
on the ROI channels are dropped. All
epochs for a given subject (pooled across that dataset's runs/tasks) are
averaged into a per-subject ROI waveform; per-subject waveforms are then
averaged into one grand-average waveform per dataset. Peak latency/amplitude of
the most prominent negative and positive deflection within canonical search
windows are reported per subject and for the grand average.

Usage:
    py -3.12 04_evoked_potential_check.py [--subjects sub-01 sub-02 ...] [--clean]

--clean (robustness check): channels marked bad in the BIDS _channels.tsv are
interpolated and the ~18 Hz line is notched out before any other step
(common.clean_raw); outputs get a '_clean' suffix.

Writes:
    derivatives/validation/04_evoked_potential_subject_summary.csv
    derivatives/validation/04_evoked_potential_grand_average.csv
"""
import argparse

import mne
import numpy as np
import pandas as pd

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS, clean_raw

mne.set_log_level("ERROR")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]

MODALITY = {"Dataset_A": "visual", "Dataset_B": "auditory", "Dataset_C": "auditory"}
ROI = {"visual": ["O1", "Oz", "O2", "POz"], "auditory": ["Fz", "Cz"]}

# "final" reference choice per dataset, picked by comparing native (CPz)/average/
# linked-mastoids on the full 30-subject corpus: CPz (native) gives a clean, textbook
# P1-N1 VEP over the occipital ROI and is left alone; linked mastoids gives the
# largest, cleanest auditory N1-P2 complex over the fronto-central ROI (CPz sits right
# next to Fz/Cz and distorts/cancels the AEP there; average reference is a middle
# ground that is worse than mastoids for the AEP and worse than native for the VEP).
REFERENCE_BY_DATASET = {"Dataset_A": "native", "Dataset_B": "linked_mastoids",
                         "Dataset_C": "linked_mastoids"}

# Cues whose onset carries the sensory stimulus of the dataset. Dataset_A: every
# cue is an image onset. Dataset_B/C: the beep sounds at movement cues only.
STIM_TRIAL_TYPES = {"Dataset_A": ("rest", "left", "right"),
                    "Dataset_B": ("left", "right"),
                    "Dataset_C": ("left", "right")}

TMIN, TMAX = -0.2, 0.6
BASELINE = (None, 0)
REJECT_UV = 150.0  # peak-to-peak epoch rejection threshold on ROI channels

# Search windows for peak picking (seconds post-stimulus).
NEG_WINDOW = (0.06, 0.25)
POS_WINDOW = (0.05, 0.30)


def load_epoch_waveforms(ds, sub_id, task, run, roi, reference, clean=False):
    """Return (n_epochs, n_times) array of ROI-averaged, baseline-corrected
    epoch waveforms in uV, the shared `times` vector and the number of
    stimulus onsets before rejection. (None, None, 0) if the recording/events
    are missing; (None, None, n) if no epoch survives rejection.
    """
    base = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}"
    vhdr = base.with_name(base.name + "_eeg.vhdr")
    events_tsv = base.with_name(base.name + "_events.tsv")
    if not vhdr.exists() or not events_tsv.exists():
        return None, None, 0

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    if clean:
        clean_raw(raw, ds, sub_id, task, run)
    if reference == "average":
        raw.set_eeg_reference("average", verbose=False)
    elif reference == "linked_mastoids":
        raw.set_eeg_reference(["M1", "M2"], verbose=False)
    elif reference == "native":
        pass  # keep the dataset's native reference (CPz)
    else:
        raise ValueError(f"Unknown reference '{reference}'")
    raw.filter(0.5, 30.0, verbose=False)
    roi = [ch for ch in roi if ch in raw.ch_names]
    raw.pick(roi)

    ev_df = pd.read_csv(events_tsv, sep="\t")
    ev_df = ev_df[ev_df["trial_type"].isin(STIM_TRIAL_TYPES[ds])]
    events = np.column_stack([
        ev_df["sample"].to_numpy(dtype=int),
        np.zeros(len(ev_df), dtype=int),
        np.ones(len(ev_df), dtype=int),
    ])
    epochs = mne.Epochs(
        raw, events, event_id={"cue": 1}, tmin=TMIN, tmax=TMAX, baseline=BASELINE,
        preload=True, verbose=False, reject=dict(eeg=REJECT_UV * 1e-6),
    )
    if len(epochs) == 0:
        return None, None, len(ev_df)

    data_uv = epochs.get_data(picks=roi, units="uV")  # (n_epochs, n_roi_ch, n_times)
    roi_mean = data_uv.mean(axis=1)  # (n_epochs, n_times)
    return roi_mean, epochs.times, len(ev_df)


def pick_peaks(waveform_uv, times):
    neg_mask = (times >= NEG_WINDOW[0]) & (times <= NEG_WINDOW[1])
    pos_mask = (times >= POS_WINDOW[0]) & (times <= POS_WINDOW[1])
    neg_i = np.argmin(waveform_uv[neg_mask])
    pos_i = np.argmax(waveform_uv[pos_mask])
    return dict(
        neg_peak_latency_ms=round(float(times[neg_mask][neg_i] * 1000), 1),
        neg_peak_uv=round(float(waveform_uv[neg_mask][neg_i]), 3),
        pos_peak_latency_ms=round(float(times[pos_mask][pos_i] * 1000), 1),
        pos_peak_uv=round(float(waveform_uv[pos_mask][pos_i]), 3),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--reference", choices=["average", "linked_mastoids", "native", "final"],
                         default="final",
                         help="'final' picks the best reference per dataset (see "
                              "REFERENCE_BY_DATASET): native for the VEP, linked "
                              "mastoids for the AEP.")
    parser.add_argument("--clean", action="store_true",
                        help="Robustness check: interpolate channels marked bad in the BIDS "
                             "_channels.tsv and notch out the ~18 Hz line where detected "
                             "(common.clean_raw). Outputs get a '_clean' suffix.")
    args = parser.parse_args()
    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    suffix = "" if args.reference == "final" else f"_{args.reference}"
    if args.clean:
        suffix += "_clean"
    if args.subjects:  # a partial run never overwrites the full-corpus outputs
        suffix += "_partial"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_rows = []
    subject_waveforms = {}  # (ds, sub_id) -> mean waveform (uV)
    shared_times = None
    n_recordings = 0

    for ds, n_runs in DATASETS.items():
        roi = ROI[MODALITY[ds]]
        reference = REFERENCE_BY_DATASET[ds] if args.reference == "final" else args.reference
        for sub_id in subjects:
            epoch_list = []
            n_events = 0
            for task in TASKS:
                for run in range(1, n_runs + 1):
                    w, times, n_ev = load_epoch_waveforms(ds, sub_id, task, run, roi, reference,
                                                          clean=args.clean)
                    n_recordings += 1
                    n_events += n_ev
                    if n_recordings % 50 == 0:
                        print(f"...{n_recordings} recordings processed")
                    if w is None:
                        continue
                    if shared_times is None:
                        shared_times = times
                    epoch_list.append(w)

            if not epoch_list:
                continue
            stacked = np.vstack(epoch_list)  # (n_epochs_total, n_times)
            subj_mean = stacked.mean(axis=0)
            subject_waveforms[(ds, sub_id)] = subj_mean

            row = dict(dataset=ds, subject=sub_id, modality=MODALITY[ds],
                       reference=reference, roi=",".join(roi), n_events=n_events,
                       n_epochs=stacked.shape[0])
            row.update(pick_peaks(subj_mean, shared_times))
            subject_rows.append(row)

    subj_df = pd.DataFrame(subject_rows)
    subj_out = OUT_DIR / f"04_evoked_potential_subject_summary{suffix}.csv"
    subj_df.to_csv(subj_out, index=False)
    print(f"\nPer-subject summary written to {subj_out}")

    # Grand average per dataset.
    grand_rows = []
    ga_wide = {"time_s": shared_times}
    for ds in DATASETS:
        waves = [v for (d, s), v in subject_waveforms.items() if d == ds]
        if not waves:
            continue
        reference = REFERENCE_BY_DATASET[ds] if args.reference == "final" else args.reference
        grand = np.mean(waves, axis=0)
        ga_wide[f"{ds}_grand_avg_uv"] = grand
        peaks = pick_peaks(grand, shared_times)
        grand_rows.append(dict(dataset=ds, modality=MODALITY[ds], reference=reference,
                                roi=",".join(ROI[MODALITY[ds]]), n_subjects=len(waves), **peaks))
        print(f"\n{ds} ({MODALITY[ds]}, reference={reference}, ROI={ROI[MODALITY[ds]]}, n={len(waves)} subjects):")
        print(f"  negative peak: {peaks['neg_peak_uv']} uV @ {peaks['neg_peak_latency_ms']} ms")
        print(f"  positive peak: {peaks['pos_peak_uv']} uV @ {peaks['pos_peak_latency_ms']} ms")

    ga_df = pd.DataFrame(ga_wide)
    ga_out = OUT_DIR / f"04_evoked_potential_grand_average{suffix}.csv"
    ga_df.to_csv(ga_out, index=False)
    print(f"\nGrand-average waveforms written to {ga_out}")

    print("\nGrand-average peak summary:")
    print(pd.DataFrame(grand_rows).to_string(index=False))


if __name__ == "__main__":
    main()
