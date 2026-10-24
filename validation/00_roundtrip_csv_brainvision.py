"""Technical Validation #0 - Round-trip check: original CSV exports vs released BrainVision.

csv_to_bids.py writes the EEG with pybv as IEEE float32 samples with a resolution of
0.1 uV per unit (stored value = voltage in uV / 0.1), and TRIG and STATUS with a
resolution of 1. This script reads every original CSV export in sourcedata/ and the
corresponding released BrainVision file (with MNE), and reports for each recording:
  - number of samples in both files;
  - maximum absolute difference of the 32 EEG channels, in uV, and the same relative
    to the channel's standard deviation (float32 storage gives ~1e-7 relative error);
  - whether TRIG and STATUS are identical;
  - whether the event samples in _events.tsv match the CSV rows with an "Event Id".

Usage:
    py -3.12 00_roundtrip_csv_brainvision.py [--subjects S01 S02 ...]

Writes:
    derivatives/validation/00_roundtrip_csv_brainvision.csv
    derivatives/validation/00_roundtrip_cache/<S..>.csv  (resumable)
"""
import argparse
import importlib.util
import warnings

import mne
import numpy as np
import pandas as pd

from common import BIDS_ROOT, OUT_DIR

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

spec = importlib.util.spec_from_file_location("c2b", BIDS_ROOT / "code" / "csv_to_bids.py")
c2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2b)
CACHE = OUT_DIR / "00_roundtrip_cache"


def check(csv_path, bids_num, ds, task, run):
    df = pd.read_csv(csv_path, dtype={"Event Id": str}, low_memory=False)
    eeg_csv_uV = df[c2b.EEG_CHANNELS_RAW].to_numpy(dtype=np.float64).T * 1e6
    aux_csv = df[c2b.AUX_RAW].to_numpy(dtype=np.float64).T
    ev_csv = np.where((df["Event Id"].notna() & (df["Event Id"].astype(str).str.strip() != "")).to_numpy())[0]
    del df
    stem = f"sub-{bids_num:02d}_task-{task}_run-{run:02d}"
    eeg_dir = BIDS_ROOT / ds / f"sub-{bids_num:02d}" / "eeg"
    raw = mne.io.read_raw_brainvision(eeg_dir / f"{stem}_eeg.vhdr", preload=True)
    eeg_bv_uV = raw.get_data(picks=c2b.EEG_CHANNELS_BIDS) * 1e6
    aux_bv = raw.get_data(picks=c2b.AUX_BIDS)
    ev = pd.read_csv(eeg_dir / f"{stem}_events.tsv", sep="\t")
    ev_bids = np.sort(ev.loc[ev.trial_type != "feedback", "sample"].to_numpy())
    same_n = eeg_csv_uV.shape[1] == eeg_bv_uV.shape[1]
    diff = np.abs(eeg_csv_uV - eeg_bv_uV) if same_n else np.full(1, np.nan)
    return dict(source=csv_path.name, dataset=ds, recording=stem,
                n_samples_csv=eeg_csv_uV.shape[1], n_samples_bv=eeg_bv_uV.shape[1],
                eeg_max_abs_diff_uV=float(diff.max()),
                eeg_max_rel_diff=float((diff.max(axis=1) / eeg_csv_uV.std(axis=1)).max()) if same_n else np.nan,
                trig_status_identical=bool(same_n and np.array_equal(aux_csv, aux_bv)),
                events_identical=bool(np.array_equal(ev_csv, ev_bids)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None, help="source folders, e.g. S01")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    by_sub = {}
    for item in c2b.iter_source_files(args.subjects):
        by_sub.setdefault(item[0].parent.name, []).append(item)
    frames = []
    for sub, items in by_sub.items():
        path = CACHE / f"{sub}.csv"
        if not path.exists():
            pd.DataFrame([check(*it) for it in items]).to_csv(path, index=False)
            print(f"{sub} done", flush=True)
        frames.append(pd.read_csv(path))
    res = pd.concat(frames, ignore_index=True)
    if args.subjects:
        print(res.to_string(index=False))
        return
    res.to_csv(OUT_DIR / "00_roundtrip_csv_brainvision.csv", index=False)
    print(f"{len(res)} recordings; samples equal in {int((res.n_samples_csv == res.n_samples_bv).sum())}; "
          f"max EEG difference {res.eeg_max_abs_diff_uV.max():.2e} uV "
          f"(max relative to channel SD {res.eeg_max_rel_diff.max():.1e}); "
          f"TRIG/STATUS identical in {int(res.trig_status_identical.sum())}; "
          f"event samples identical in {int(res.events_identical.sum())}")


if __name__ == "__main__":
    main()
