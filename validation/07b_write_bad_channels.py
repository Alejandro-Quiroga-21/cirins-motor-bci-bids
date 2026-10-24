"""Technical Validation #2e - Write approved bad channels into the BIDS datasets.

Reads the tier-"bad" rows of derivatives/validation/07_bad_channels_proposal.csv
(produced by 07_bad_channels.py) and, for every recording of each affected
subject in Dataset_A/B/C, writes a recording-level
sub-XX_task-<task>_run-<run>_channels.tsv: an exact copy of the dataset-wide
channels.tsv with status=bad and a status_description for the flagged
channels. By the BIDS inheritance principle this file takes precedence over the
dataset-wide channels.tsv for that recording; all other subjects keep using the
dataset-wide file. EEG samples are not modified.

The channel list was reviewed and approved by the dataset authors on
2026-09-30; APPROVED below pins it, and the script refuses to write if the
proposal file no longer matches it (e.g. after a threshold change in 07).

Usage:
    py -3.12 07b_write_bad_channels.py [--bids-root PATH]

--bids-root: folder holding Dataset_A/B/C (default: the released location), e.g.
the --out-root of a csv_to_bids.py rebuild.
"""
import argparse
from pathlib import Path

import pandas as pd

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS

APPROVED = {
    "sub-11": {"FC1", "FC2", "C3"},
    "sub-12": {"F3", "Fz", "F4", "FC1", "FC2"},
    "sub-14": {"Fz"},
}


def describe(row):
    if row.criterion == "bridged":
        return (f"gel bridge within {row.group} (30-45 Hz correlation >= 0.985 in "
                f"{row.fraction_of_recordings:.0%} of the session's recordings) with unstable "
                f"contact (0.5 Hz high-passed std {row.amplitude_vs_corpus:.1f}x corpus median "
                f"for this channel); persistent over the whole session")
    return (f"high noise / poor contact ({row.detail}; in "
            f"{row.fraction_of_recordings:.0%} of the session's recordings); "
            f"persistent over the whole session")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, default=BIDS_ROOT)
    root = parser.parse_args().bids_root
    prop = pd.read_csv(OUT_DIR / "07_bad_channels_proposal.csv")
    bad = prop[prop["tier"] == "bad"]
    found = {s: set(g["channel"]) for s, g in bad.groupby("subject")}
    if found != APPROVED:
        raise SystemExit(f"Proposal does not match the approved list.\n  proposal: {found}\n"
                         f"  approved: {APPROVED}\nNothing written.")

    n_written = 0
    for ds, n_runs in DATASETS.items():
        base = pd.read_csv(root / ds / "channels.tsv", sep="\t", dtype=str,
                           keep_default_na=False)
        for sub_id, g in bad.groupby("subject"):
            table = base.copy()
            for row in g.itertuples():
                idx = table.index[table["name"] == row.channel]
                assert len(idx) == 1, (ds, sub_id, row.channel)
                table.loc[idx, "status"] = "bad"
                table.loc[idx, "status_description"] = describe(row)
            for task in TASKS:
                for run in range(1, n_runs + 1):
                    eeg_dir = root / ds / sub_id / "eeg"
                    stem = f"{sub_id}_task-{task}_run-{run:02d}"
                    assert (eeg_dir / f"{stem}_eeg.vhdr").exists(), (ds, stem)
                    out = eeg_dir / f"{stem}_channels.tsv"
                    table.to_csv(out, sep="\t", index=False, lineterminator="\n")
                    n_written += 1
        print(f"{ds}: recording-level channels.tsv written for {sorted(found)}")
    print(f"\n{n_written} files written.")


if __name__ == "__main__":
    main()
