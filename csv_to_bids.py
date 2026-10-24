"""Convert the raw per-run CSV exports in sourcedata/ into three sibling BIDS-EEG
datasets (BrainVision format), one per task (Dataset_A/B/C), under
Dataset_X/sub-<id>/eeg/. Within each dataset, the upper/lower limb condition is the
BIDS task entity (task-lower / task-upper).

Usage:
    python csv_to_bids.py [--subjects S01 S02 ...] [--out-root PATH] [--dry-run]

A full rebuild of the released datasets is:
    python csv_to_bids.py                                  # recordings + root files
    python validation/07_bad_channels.py                   # bad-channel proposal
    python validation/07b_write_bad_channels.py            # recording-level _channels.tsv
The dataset-level ("root") files of each dataset - participants.tsv/.json,
channels.tsv, events.json, dataset_description.json, CITATION.cff,
task-*_eeg.json, README, CHANGES - are curated by hand and kept under
code/bids_root_files/<Dataset>/; this script copies them into each output
dataset, so a rebuild never loses them (as happened on 2026-09-17, when only the
per-recording files were regenerated). Edit them there, not in the datasets.
Each recording also gets an _eeg.json sidecar holding only its RecordingDuration;
everything else is inherited from the dataset-level task-*_eeg.json.
--out-root writes a complete copy elsewhere (e.g. to verify that the released
datasets are reproduced exactly) without touching the released ones.

Assumptions (confirmed with the data owner):
  - Sampling rate is read from the "Time:<rate>Hz" first-column header of each file.
  - The 32 scalp-electrode columns are already in volts (V) and are handed off to
    pybv as-is (no rescaling), which itself converts volts -> the target unit ("uV")
    for storage.
    CORRECTED 2026-09-23: two earlier versions of this script (and every BIDS
    recording produced by them) assumed the raw CSV columns were in millivolts, then
    microvolts, scaling every EEG sample down by an erroneous extra factor of 1000
    (mV assumption) and then a further erroneous 1e6 factor (uV assumption). Both were
    caught by technical-validation script 03_signal_quality_check.py: the mV
    assumption gave a corpus-wide median per-channel std of ~0.05 uV, and the uV
    assumption gave ~0.00005 uV - both physiologically implausible (real scalp EEG is
    ~10-50 uV). Reading the raw sourcedata CSV values directly as volts (no scaling)
    gives per-channel std of 19-45 uV for central/motor channels (C3/Cz/C4) and
    77-264 uV for frontal channels (Fp1/Fp2/F3, elevated by eye-movement/blink
    artifact, as expected) - both the magnitude and the frontal > central > occipital
    topography are physiologically correct, checked in two independent recordings.
    See Dataset_A/CHANGES for the resulting dataset-wide correction.
  - "Channel 33" is the TTL trigger/event code channel and "Channel 34" is the
    amplifier's running sample counter (+1 per sample, continuous across runs until
    the amplifier is restarted); neither is a voltage signal, so both are written
    unscaled ("n/a" unit) with a resolution of 1. (Earlier versions used pybv's
    default 0.1 resolution for every channel: with float32 storage, counter values
    above ~1.7e6 then exceed the exact-integer range once divided by 0.1 and were
    stored within +/-0.2 of the true integer. EEG channels keep 0.1 uV.)
  - The trailing run index in the source filename (..._<n>.csv) maps directly to
    run-01, run-02, ... It follows the source-file numbering, which is not always
    the acquisition order: in sub-06, sub-24 and sub-26 one run 3 was acquired later
    (see the amplifier counter, validation/09_timing_and_session.py).
  - T1 -> Dataset_A (calibration, 4 runs), T2 -> Dataset_B (open-loop, 3 runs),
    T3 -> Dataset_C (closed-loop, 3 runs). Lower/Upper -> task-lower/task-upper.
  - sourcedata/S03/ holds the recording originally exported as S31 (files renamed
    S31_* -> S03_*): participant S03 was recorded twice, and the first recording had
    its open-/closed-loop movement cues mislabelled to the non-dominant side (see
    Dataset_A/CHANGES, 1.0.0). It is converted as sub-03.
"""
import argparse
import json
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pybv

# Expected/intentional: TRIG and STATUS are non-voltage channels written with unit
# "n/a" so pybv does not rescale them; this is outside the strict BrainVision spec
# (which only names uV) but is read back correctly by MNE/EEGLAB via the Resolution
# field, as verified against the source CSVs.
warnings.filterwarnings("ignore", message="Encountered small Greek letter mu")
warnings.filterwarnings("ignore", message="Encountered unsupported non-voltage units")

# The deposit root is the parent of code/, wherever the deposit was unpacked.
BIDS_ROOT = Path(__file__).resolve().parents[1]  # default output root (--out-root overrides)
SOURCE_ROOT = BIDS_ROOT / "sourcedata"
ROOT_FILES_DIR = Path(__file__).resolve().parent / "bids_root_files"
ROOT_FILES = ["participants.tsv", "participants.json", "channels.tsv", "events.json",
              "dataset_description.json", "CITATION.cff", "task-lower_eeg.json",
              "task-upper_eeg.json", "README", "CHANGES"]

EEG_CHANNELS_RAW = [
    "FP1", "FPZ", "FP2", "F7", "F3", "FZ", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "M1", "T7", "C3", "CZ", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "PZ", "P4", "P8", "POZ", "O1", "OZ", "O2",
]
EEG_CHANNELS_BIDS = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "M1", "T7", "C3", "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2",
]
AUX_RAW = ["Channel 33", "Channel 34"]
AUX_BIDS = ["TRIG", "STATUS"]
AUX_TYPES = ["TRIG", "MISC"]

DATASET_MAP = {
    "1": ("Dataset_A", 4),
    "2": ("Dataset_B", 3),
    "3": ("Dataset_C", 3),
}
LIMB_TASK_MAP = {"Lower": "lower", "Upper": "upper"}

TRIAL_TYPE_MAP = {"Rest": "rest", "Left": "left", "Right": "right"}

# Dataset_C (simulated closed loop): the sham-feedback mirror animation starts at the
# onset of the rest cue that follows each movement cue and lasts a programmed time.
# It has no marker of its own in the acquisition exports, so an explicit "feedback"
# row (onset = that rest onset, duration = programmed animation length, value = n/a)
# is added after each such rest row of _events.tsv. The native .vmrk markers are
# left untouched.
FEEDBACK_DATASET = "Dataset_C"
FEEDBACK_S = {"lower": 5.0, "upper": 3.0}


def add_feedback_rows(rows, task):
    """rows: list of (onset_s, duration_str, trial_type, value_str, sample) in order.
    Returns the list with a "feedback" row after every rest row that follows a
    movement cue."""
    out = []
    for i, row in enumerate(rows):
        out.append(row)
        onset, _, ttype, _, sample = row
        if ttype == "rest" and i > 0 and rows[i - 1][2] in ("left", "right"):
            out.append((onset, f"{FEEDBACK_S[task]:.3f}", "feedback", "n/a", sample))
    return out

FNAME_RE = re.compile(r"^S(\d+)_(Lower|Upper)_T([123])_(\d+)\.csv$")


def parse_sfreq(first_col_name):
    m = re.search(r"([\d.]+)\s*Hz", first_col_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse sampling rate from column '{first_col_name}'")
    return float(m.group(1))


def write_root_files(out_root, dry_run=False):
    """Copy the curated dataset-level files into each output dataset."""
    for dataset_dir, _ in DATASET_MAP.values():
        src, dst = ROOT_FILES_DIR / dataset_dir, out_root / dataset_dir
        missing = [f for f in ROOT_FILES if not (src / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{src} is missing {missing}")
        if dry_run:
            print(f"[dry-run] {len(ROOT_FILES)} root files -> {dst}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in ROOT_FILES:
            shutil.copy2(src / f, dst / f)
        print(f"OK  {len(ROOT_FILES)} root files -> {dst}")


def write_eeg_sidecar(out_dir, fname_base, n_samples, sfreq):
    """Recording-level _eeg.json: only the fields that differ between recordings."""
    sidecar = {"RecordingDuration": round(n_samples / sfreq, 3)}
    (out_dir / f"{fname_base}.json").write_text(json.dumps(sidecar, indent=4) + "\n",
                                                encoding="utf-8")


def convert_file(csv_path, bids_num, dataset_dir, task, run_idx, out_root=BIDS_ROOT, dry_run=False):
    df = pd.read_csv(csv_path, dtype={"Event Id": str}, low_memory=False)
    sfreq = parse_sfreq(df.columns[0])

    eeg_V = df[EEG_CHANNELS_RAW].to_numpy(dtype=np.float64).T  # (32, n), already volts
    aux = df[AUX_RAW].to_numpy(dtype=np.float64).T  # (2, n)
    data = np.vstack([eeg_V, aux])
    ch_names = EEG_CHANNELS_BIDS + AUX_BIDS
    units = ["uV"] * len(EEG_CHANNELS_BIDS) + ["n/a"] * len(AUX_BIDS)
    # 0.1 uV for EEG; integer resolution for the trigger code and sample counter,
    # which must stay exact integers (see module docstring).
    resolution = np.array([0.1] * len(EEG_CHANNELS_BIDS) + [1.0] * len(AUX_BIDS))

    ev_mask = df["Event Id"].notna() & (df["Event Id"].astype(str).str.strip() != "")
    ev_idx = np.where(ev_mask.to_numpy())[0]
    ev_labels_raw = df.loc[ev_idx, "Event Id"].astype(str).str.strip().tolist()
    ev_value = df.loc[ev_idx, "Channel 33"].to_numpy(dtype=np.float64)
    onset_sec = ev_idx / sfreq
    trial_types = [TRIAL_TYPE_MAP.get(lbl, lbl.lower()) for lbl in ev_labels_raw]

    durations = np.full(len(onset_sec), np.nan)
    if len(onset_sec) > 1:
        durations[:-1] = onset_sec[1:] - onset_sec[:-1]

    sub_label = f"sub-{bids_num:02d}"
    fname_base = f"{sub_label}_task-{task}_run-{run_idx:02d}_eeg"
    out_dir = out_root / dataset_dir / sub_label / "eeg"

    if dry_run:
        print(f"[dry-run] {csv_path.name} -> {dataset_dir}/{out_dir.relative_to(out_root / dataset_dir)}/{fname_base} "
              f"(n={data.shape[1]} samples, sfreq={sfreq}, n_events={len(ev_idx)})")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    vmrk_events = [
        {"onset": int(idx), "duration": 1, "description": lbl, "type": "Comment"}
        for idx, lbl in zip(ev_idx, ev_labels_raw)
    ]

    pybv.write_brainvision(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        fname_base=fname_base,
        folder_out=out_dir,
        events=vmrk_events if vmrk_events else None,
        unit=units,
        resolution=resolution,
        overwrite=True,
    )

    write_eeg_sidecar(out_dir, fname_base, data.shape[1], sfreq)

    # events.tsv
    ev_rows = []
    for onset, dur, ttype, val, idx in zip(onset_sec, durations, trial_types, ev_value, ev_idx):
        dur_str = "n/a" if np.isnan(dur) else f"{dur:.3f}"
        val_str = "n/a" if np.isnan(val) else str(int(val))
        ev_rows.append((onset, dur_str, ttype, val_str, int(idx)))
    if dataset_dir == FEEDBACK_DATASET:
        ev_rows = add_feedback_rows(ev_rows, task)
    ev_lines = ["onset\tduration\ttrial_type\tvalue\tsample"]
    ev_lines += [f"{o:.3f}\t{d}\t{t}\t{v}\t{s}" for o, d, t, v, s in ev_rows]
    (out_dir / f"{sub_label}_task-{task}_run-{run_idx:02d}_events.tsv").write_text(
        "\n".join(ev_lines) + "\n", encoding="utf-8"
    )

    print(f"OK  {csv_path.name} -> {dataset_dir}/{fname_base} "
          f"(n={data.shape[1]} samples, sfreq={sfreq}, n_events={len(ev_idx)})")


def iter_source_files(subjects_filter=None):
    for sub_dir in sorted(SOURCE_ROOT.glob("S*")):
        if not sub_dir.is_dir():
            continue
        if subjects_filter and sub_dir.name not in subjects_filter:
            continue
        for csv_path in sorted(sub_dir.glob(f"{sub_dir.name}_*.csv")):
            m = FNAME_RE.match(csv_path.name)
            if not m:
                print(f"WARN: skipping unrecognized filename {csv_path.name}", file=sys.stderr)
                continue
            num_str, limb, tnum, run_str = m.groups()
            bids_num = int(num_str)
            dataset_dir, expected_runs = DATASET_MAP[tnum]
            task = LIMB_TASK_MAP[limb]
            run_idx = int(run_str)
            if run_idx > expected_runs:
                print(f"WARN: {csv_path.name} run index {run_idx} exceeds expected "
                      f"{expected_runs} for {dataset_dir}/task-{task}", file=sys.stderr)
            yield csv_path, bids_num, dataset_dir, task, run_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None,
                         help="Limit to these source subject folder names, e.g. S01 S02")
    parser.add_argument("--out-root", type=Path, default=BIDS_ROOT,
                        help="Where Dataset_A/B/C are written (default: the released location)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    write_root_files(args.out_root, dry_run=args.dry_run)
    n = 0
    for csv_path, bids_num, dataset_dir, task, run_idx in iter_source_files(args.subjects):
        convert_file(csv_path, bids_num, dataset_dir, task, run_idx, out_root=args.out_root,
                     dry_run=args.dry_run)
        n += 1
    print(f"Processed {n} files.")


if __name__ == "__main__":
    main()
