"""Technical Validation #2 - QC table of known TTL-trigger event anomalies.

This does NOT modify any raw data. Dataset_A/B/C's `_events.tsv` files ship
exactly as recorded: `trial_type` comes from the acquisition software's own
"Event Id" label and `value` is a direct sample of the raw hardware TRIG
line (see code/csv_to_bids.py, convert_file()) - two independent sources.
Editing either to "fix" an anomaly would mean overwriting a genuine raw
measurement with a guess.

Instead, this script identifies every event whose raw TTL value deviates
from the empirically-learned per-condition majority (see
01_structural_and_design_check.py's docstring and
01_ttl_value_encoding_summary.csv for why a fixed rest/left/right map is
the wrong model) and writes them to a single derivatives-level QC table,
`flagged_events.tsv`, enriched with each event's duration next to the
typical (median) duration for its condition - the two anomaly mechanisms
identified by manual inspection (a spurious short-duration re-trigger
immediately after a genuine cue in Dataset_A; a missed first-transition of
the run in Dataset_B/C) both show up as a duration far from that median,
which is why standard duration-based epoch rejection in any downstream
analysis will naturally exclude these trials without special-casing them.

Usage:
    py -3.12 02_flag_event_anomalies.py

Writes derivatives/validation/flagged_events.tsv.
"""
import pandas as pd

from common import OUT_DIR, learn_ttl_value_expectations, load_all_events


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all events and learning per-condition TTL value majority...")
    corpus = load_all_events()
    ttl_lookup, ttl_summary = learn_ttl_value_expectations(corpus)

    corpus["expected_value"] = [
        ttl_lookup.get((ds, tt, pt))
        for ds, tt, pt in zip(corpus["dataset"], corpus["trial_type"], corpus["prev_type"])
    ]

    duration_median = (
        corpus.groupby(["dataset", "task", "trial_type", "prev_type"])["duration"]
        .median()
        .rename("duration_group_median")
        .reset_index()
    )
    corpus = corpus.merge(duration_median, on=["dataset", "task", "trial_type", "prev_type"], how="left")

    flagged = corpus[corpus["value"] != corpus["expected_value"]].copy()
    flagged["reason"] = "ttl_value_anomaly"
    flagged["duration_ratio_vs_typical"] = (
        flagged["duration"] / flagged["duration_group_median"]
    ).round(2)

    out_cols = [
        "dataset", "subject", "task", "run", "event_index", "onset", "duration",
        "duration_group_median", "duration_ratio_vs_typical",
        "trial_type", "prev_type", "value", "expected_value", "reason",
    ]
    flagged = flagged[out_cols].sort_values(["dataset", "subject", "task", "run", "event_index"])

    out_path = OUT_DIR / "flagged_events.tsv"
    flagged.to_csv(out_path, sep="\t", index=False)

    n_total = len(corpus)
    n_files = corpus.groupby(["dataset", "subject", "task", "run"]).ngroups
    n_flagged_files = flagged.groupby(["dataset", "subject", "task", "run"]).ngroups

    print(f"Total events in corpus: {n_total} (across {n_files} recordings)")
    print(f"Flagged events: {len(flagged)} ({100 * len(flagged) / n_total:.3f}%), "
          f"in {n_flagged_files} recordings")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
