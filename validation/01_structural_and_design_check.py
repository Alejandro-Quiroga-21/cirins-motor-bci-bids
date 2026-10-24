"""Technical Validation #1 - Structural integrity + experimental-design checks.

Walks every expected recording across the three sibling BIDS datasets
(Dataset_A/B/C) and checks, per sub-XX x task-{lower,upper} x run-N:

  Structural:
    - .vhdr/.eeg/.vmrk + _events.tsv all present
    - sampling rate == 500 Hz
    - channel names/order/count match the dataset-wide channels.tsv (34 ch)

  Experimental design:
    - event count matches the documented protocol (41/run in Dataset_A,
      21/run in Dataset_B/C)
    - rest count == movement count + 1 (protocol starts and separates every
      movement with a rest cue)
    - trial_type -> TTL value mapping is internally consistent: the raw TRIG
      code is not a fixed 3-way map (Dataset_A gives left/right their own
      codes, 32/128, while Dataset_B/C - dominant-limb only - reuse a single
      generic movement code, 32, regardless of which side that is; a "rest"
      event's code depends on what preceded it, and the very first rest of
      every run is a distinct fixed-duration baseline cue). The expected
      value per (dataset, trial_type, preceding trial_type) is therefore
      learned empirically from the whole corpus (majority vote) and any row
      deviating from its own group's majority is flagged as a candidate
      trigger anomaly. See 01_ttl_value_encoding_summary.csv for the learned
      table.
    - first event of every run is "rest"
    - Dataset_B/C (dominant-limb-only tasks): movement trial_type is
      restricted to the participant's dominant side for that limb
      (footedness for task-lower, handedness for task-upper) - this is a
      regression test for the S03/S31 mislabelling bug documented in
      Dataset_A/CHANGES.

  Cross-dataset metadata:
    - participants.tsv is row-identical across Dataset_A/B/C
    - channels.tsv is identical across Dataset_A/B/C

Usage:
    py -3.12 01_structural_and_design_check.py [--subjects sub-01 sub-02 ...]

Writes a per-issue CSV and prints a summary to stdout.
"""
import argparse

import mne
import pandas as pd

from common import cues_only
from common import (
    BIDS_ROOT, OUT_DIR, DATASETS, DOMINANT_ONLY, TASKS, N_SUBJECTS,
    EXPECTED_SFREQ, EXPECTED_CHANNELS, EXPECTED_N_EVENTS,
    learn_ttl_value_expectations,
)

mne.set_log_level("ERROR")


def issue(rows, **kwargs):
    rows.append(kwargs)


def check_participants_and_channels(rows):
    """Cross-dataset metadata identity (participants.tsv, channels.tsv)."""
    ref_participants = None
    ref_channels = None
    for ds in DATASETS:
        p_path = BIDS_ROOT / ds / "participants.tsv"
        c_path = BIDS_ROOT / ds / "channels.tsv"
        if not p_path.exists():
            issue(rows, dataset=ds, subject="", task="", run="", category="metadata",
                  detail="participants.tsv missing")
            continue
        if not c_path.exists():
            issue(rows, dataset=ds, subject="", task="", run="", category="metadata",
                  detail="channels.tsv missing")
            continue

        participants = pd.read_csv(p_path, sep="\t")
        if len(participants) != N_SUBJECTS:
            issue(rows, dataset=ds, subject="", task="", run="", category="metadata",
                  detail=f"participants.tsv has {len(participants)} rows, expected {N_SUBJECTS}")
        if participants.isna().any().any():
            issue(rows, dataset=ds, subject="", task="", run="", category="metadata",
                  detail="participants.tsv contains missing values")

        channels = pd.read_csv(c_path, sep="\t")
        ch_names = channels["name"].tolist()
        if ch_names != EXPECTED_CHANNELS:
            issue(rows, dataset=ds, subject="", task="", run="", category="metadata",
                  detail=f"channels.tsv order/content mismatch: {ch_names}")

        if ref_participants is None:
            ref_participants = participants
            ref_channels = channels
        else:
            if not participants.equals(ref_participants):
                issue(rows, dataset=ds, subject="", task="", run="", category="metadata",
                      detail="participants.tsv differs from Dataset_A's")
            if not channels.equals(ref_channels):
                issue(rows, dataset=ds, subject="", task="", run="", category="metadata",
                      detail="channels.tsv differs from Dataset_A's")

    return ref_participants


def dominant_side(participants, sub_id, task):
    row = participants.loc[participants["participant_id"] == sub_id]
    if row.empty:
        return None
    col = "footedness" if task == "lower" else "handedness"
    return row.iloc[0][col]


def check_recording(rows, ds, sub_id, task, run, participants, ttl_lookup):
    base = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}"
    vhdr = base.with_name(base.name + "_eeg.vhdr")
    eeg = base.with_name(base.name + "_eeg.eeg")
    vmrk = base.with_name(base.name + "_eeg.vmrk")
    events_tsv = base.with_name(base.name + "_events.tsv")

    ctx = dict(dataset=ds, subject=sub_id, task=task, run=run)

    missing = [f.name for f in (vhdr, eeg, vmrk, events_tsv) if not f.exists()]
    if missing:
        issue(rows, **ctx, category="structural", detail=f"missing files: {missing}")
        return  # can't check further without the files

    # --- header checks ---
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
    if abs(raw.info["sfreq"] - EXPECTED_SFREQ) > 1e-6:
        issue(rows, **ctx, category="structural",
              detail=f"sfreq={raw.info['sfreq']}, expected {EXPECTED_SFREQ}")
    if raw.ch_names != EXPECTED_CHANNELS:
        issue(rows, **ctx, category="structural",
              detail=f"channel names/order mismatch: {raw.ch_names}")

    # --- event-design checks ---
    events = cues_only(pd.read_csv(events_tsv, sep="\t", dtype={"value": str}))
    n_events = len(events)
    expected_n = EXPECTED_N_EVENTS[ds]
    if n_events != expected_n:
        issue(rows, **ctx, category="design",
              detail=f"n_events={n_events}, expected {expected_n}")

    n_rest = (events["trial_type"] == "rest").sum()
    n_move = events["trial_type"].isin(["left", "right"]).sum()
    if n_rest != n_move + 1:
        issue(rows, **ctx, category="design",
              detail=f"n_rest={n_rest}, n_move={n_move} (expected n_rest == n_move + 1)")

    if events.iloc[0]["trial_type"] != "rest":
        issue(rows, **ctx, category="design",
              detail=f"first event is '{events.iloc[0]['trial_type']}', expected 'rest'")

    prev_type = events["trial_type"].shift(1).fillna("__FIRST__")
    expected = [
        ttl_lookup.get((ds, tt, pt))
        for tt, pt in zip(events["trial_type"], prev_type)
    ]
    mismatches = [
        (i, tt, pt, val, exp)
        for i, (tt, pt, val, exp) in enumerate(
            zip(events["trial_type"], prev_type, events["value"], expected)
        )
        if val != exp
    ]
    if mismatches:
        detail = "; ".join(
            f"row {i}: {tt} (prev={pt}) value={val}, expected={exp}"
            for i, tt, pt, val, exp in mismatches
        )
        issue(rows, **ctx, category="design",
              detail=f"{len(mismatches)} TTL value anomalies vs. learned per-condition majority: {detail}")

    if ds in DOMINANT_ONLY and participants is not None:
        dom = dominant_side(participants, sub_id, task)
        moved_sides = set(events.loc[events["trial_type"].isin(["left", "right"]), "trial_type"])
        if dom is not None and moved_sides - {dom}:
            issue(rows, **ctx, category="design",
                  detail=f"non-dominant-side movement cues found: {moved_sides}, "
                         f"dominant side is '{dom}' ({'footedness' if task == 'lower' else 'handedness'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None,
                         help="e.g. sub-01 sub-02 (default: all 30)")
    args = parser.parse_args()

    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    participants = check_participants_and_channels(rows)

    print("Learning empirical TTL value expectations from the full corpus...")
    ttl_lookup, ttl_summary = learn_ttl_value_expectations()
    ttl_summary_path = OUT_DIR / "01_ttl_value_encoding_summary.csv"
    ttl_summary.to_csv(ttl_summary_path, index=False)
    print(f"TTL value encoding summary written to {ttl_summary_path}")

    n_checked = 0
    for ds, n_runs in DATASETS.items():
        for sub_id in subjects:
            for task in TASKS:
                for run in range(1, n_runs + 1):
                    check_recording(rows, ds, sub_id, task, run, participants, ttl_lookup)
                    n_checked += 1

    report = pd.DataFrame(rows)
    out_csv = OUT_DIR / "01_structural_and_design_check_issues.csv"
    report.to_csv(out_csv, index=False)

    print(f"Checked {n_checked} recordings across {len(subjects)} subjects.")
    print(f"Total issues found: {len(report)}")
    if len(report):
        print(report["category"].value_counts().to_string())
        print(f"\nFull report written to {out_csv}")
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
