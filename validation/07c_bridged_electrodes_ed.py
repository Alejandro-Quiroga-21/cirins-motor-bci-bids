"""Technical Validation #7c - Gel bridges by electrical distance (cross-check of 07).

07_bad_channels.py flags gel bridging from 30-45 Hz correlation plus amplitude. This
script applies the standard electrical-distance method (Tenke & Kayser, 2001;
Alschuler et al., 2014) as implemented in mne.preprocessing.compute_bridged_electrodes
(defaults: 0.5-30 Hz, 2 s epochs, local-minimum cutoff 16 uV^2, a pair is bridged
when it is below the cutoff in at least 50% of the epochs) to every recording.

As in 07, a pair is reported per participant when it is bridged in at least 75% of
that participant's recordings (electrodes stay in place for the whole session), and
the channels involved are compared with the nine bad channels of 07.

Usage:
    py -3.12 07c_bridged_electrodes_ed.py [--subjects sub-01 ...]

Writes:
    derivatives/validation/07c_bridged_pairs_per_recording.csv
    derivatives/validation/07c_bridged_pairs_per_participant.csv
    derivatives/validation/07c_bridged_cache/<sub>.csv   (resumable)
"""
import argparse
import warnings

import mne
import pandas as pd

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS, recording_bads

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]
PERSIST = 0.75
CACHE = OUT_DIR / "07c_bridged_cache"


def subject_rows(sub_id):
    path = CACHE / f"{sub_id}.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for ds, n_runs in DATASETS.items():
        for task in TASKS:
            for run in range(1, n_runs + 1):
                vhdr = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}_eeg.vhdr"
                raw = mne.io.read_raw_brainvision(vhdr, preload=True)
                raw.pick(EEG_CHANNELS)
                bridged, _ = mne.preprocessing.compute_bridged_electrodes(raw, verbose=False)
                names = sorted("-".join(sorted((raw.ch_names[a], raw.ch_names[b]))) for a, b in bridged)
                rows.append(dict(subject=sub_id, dataset=ds, task=task, run=run,
                                 n_pairs=len(names), pairs=";".join(names)))
                del raw
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"{sub_id} done", flush=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    CACHE.mkdir(parents=True, exist_ok=True)
    per_rec = pd.concat([subject_rows(s) for s in subjects], ignore_index=True)
    if args.subjects:
        return
    per_rec.to_csv(OUT_DIR / "07c_bridged_pairs_per_recording.csv", index=False)

    out = []
    for sub_id, d in per_rec.groupby("subject"):
        counts = {}
        for p in d.pairs.fillna(""):
            for pair in filter(None, p.split(";")):
                counts[pair] = counts.get(pair, 0) + 1
        persistent = sorted(p for p, c in counts.items() if c >= PERSIST * len(d))
        chans = sorted({c for p in persistent for c in p.split("-")})
        bads_07 = sorted(recording_bads("Dataset_A", sub_id, "lower", 1))
        out.append(dict(subject=sub_id, n_recordings=len(d),
                        recordings_with_any_bridge_pct=round(100 * (d.n_pairs > 0).mean(), 1),
                        persistent_pairs=";".join(persistent), persistent_channels=";".join(chans),
                        bad_channels_07=";".join(bads_07)))
    out = pd.DataFrame(out)
    out.to_csv(OUT_DIR / "07c_bridged_pairs_per_participant.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[(out.persistent_pairs != "") | (out.bad_channels_07 != "")].to_string(index=False))


if __name__ == "__main__":
    main()
