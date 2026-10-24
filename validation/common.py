"""Shared constants and helpers for the technical-validation scripts."""
from pathlib import Path

import pandas as pd

BIDS_ROOT = Path(__file__).resolve().parents[2]  # .../Cirins_BIDS (this file is in code/validation)
OUT_DIR = BIDS_ROOT / "derivatives" / "validation"

DATASETS = {"Dataset_A": 4, "Dataset_B": 3, "Dataset_C": 3}
DOMINANT_ONLY = {"Dataset_B", "Dataset_C"}  # tasks performed with dominant limb only
TASKS = ["lower", "upper"]
N_SUBJECTS = 30

EXPECTED_SFREQ = 500.0
EXPECTED_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "M1", "T7", "C3", "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2", "TRIG", "STATUS",
]
EXPECTED_N_EVENTS = {"Dataset_A": 41, "Dataset_B": 21, "Dataset_C": 21}


_LINE_TABLE = None
LINE_NOTCH_WIDTH_HZ = 1.0


def recording_bads(ds, sub_id, task, run):
    """Channels marked status=bad for this recording (recording-level
    _channels.tsv, falling back to the dataset-wide channels.tsv)."""
    stem = f"{sub_id}_task-{task}_run-{run:02d}"
    path = BIDS_ROOT / ds / sub_id / "eeg" / f"{stem}_channels.tsv"
    if not path.exists():
        path = BIDS_ROOT / ds / "channels.tsv"
    ch = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return ch.loc[ch["status"] == "bad", "name"].tolist()


def clean_raw(raw, ds, sub_id, task, run):
    """Robustness-check cleaning, applied in place to a Raw holding only the 32
    scalp channels (native reference) and returned:
      1. channels marked bad in the BIDS _channels.tsv are interpolated
         (spherical spline, standard_1020 positions);
      2. the narrowband ~18 Hz interference is removed where it was detected
         (06_spectral_check.py: line_present) with a 1 Hz-wide FIR notch centred
         on that recording's measured frequency. (Sinusoid fitting - MNE's
         'spectrum_fit' - removed it fully when the frequency was stable but left
         a ~4x residual peak when it drifted by a few tenths of a Hz within the
         recording; the 1 Hz notch tolerates that drift and leaves power outside
         the notch unchanged.)
    The published data are never modified; this is only used by the analysis
    scripts' --clean mode.
    """
    global _LINE_TABLE
    import mne
    bads = recording_bads(ds, sub_id, task, run)
    if bads:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                        match_case=False, on_missing="raise")
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True, verbose=False)

    if _LINE_TABLE is None:
        _LINE_TABLE = pd.read_csv(OUT_DIR / "06_line_interference_18hz.csv").set_index(
            ["dataset", "subject", "task", "run"])
    row = _LINE_TABLE.loc[(ds, sub_id, task, run)]
    if bool(row["line_present"]):
        raw.notch_filter(freqs=[float(row["line_freq_hz"])], notch_widths=LINE_NOTCH_WIDTH_HZ,
                         method="fir", verbose=False)
    return raw


# Cue rows of _events.tsv. Dataset_C also has "feedback" rows (sham-feedback
# animation, programmed duration) that are not cues; the analyses use cues only.
CUE_TRIAL_TYPES = ("rest", "left", "right")


def cues_only(ev):
    """Drop the non-cue rows (Dataset_C "feedback") of an _events.tsv DataFrame."""
    return ev[ev["trial_type"].isin(CUE_TRIAL_TYPES)].reset_index(drop=True)


def load_all_events():
    """Read every _events.tsv in the three datasets into one long DataFrame,
    annotated with dataset/task/subject/run/event_index/prev_type.
    """
    rows = []
    for ds in DATASETS:
        for f in (BIDS_ROOT / ds).glob("sub-*/eeg/*_events.tsv"):
            df = cues_only(pd.read_csv(f, sep="\t", dtype={"value": str}))
            df["event_index"] = range(len(df))
            df["prev_type"] = df["trial_type"].shift(1).fillna("__FIRST__")
            df["dataset"] = ds
            df["subject"] = f.parts[-3]
            df["task"] = "lower" if "_task-lower_" in f.name else "upper"
            df["run"] = int(f.name.split("_run-")[1][:2])
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def learn_ttl_value_expectations(corpus=None):
    """Empirically learn the expected raw TTL 'value' for each
    (dataset, trial_type, preceding trial_type) combination, as the
    majority value observed across the whole corpus (see the docstring of
    01_structural_and_design_check.py for why this - not a fixed 3-way map -
    is the correct model of the TRIG channel). Returns the lookup dict plus
    a tidy summary DataFrame (dominant value's share and N per group).
    """
    if corpus is None:
        corpus = load_all_events()

    grouped = corpus.groupby(["dataset", "trial_type", "prev_type"])["value"]
    summary = grouped.agg(
        expected_value=lambda s: s.mode().iloc[0],
        n_total="size",
        n_matching_mode=lambda s: (s == s.mode().iloc[0]).sum(),
    ).reset_index()
    summary["pct_matching"] = (100 * summary["n_matching_mode"] / summary["n_total"]).round(2)

    lookup = {
        (r.dataset, r.trial_type, r.prev_type): r.expected_value
        for r in summary.itertuples()
    }
    return lookup, summary
