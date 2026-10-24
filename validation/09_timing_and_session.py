"""Technical Validation #9 - Cue timing and session timeline.

1. Cue timing, from every _events.tsv (onsets only, so independent of the
   'duration' column):
   - movement-cue interval: movement-cue onset to the next event's onset
     (programmed: 2.5 s in all datasets);
   - steady-state rest interval: rest-cue onset to the next movement-cue onset,
     for every rest cue except the first of each run (the first, baseline rest
     has its own fixed length and is reported separately). Programmed: random
     5-6 s; in Dataset_C the mirror-animation feedback adds a fixed latency
     after each movement cue, so its rest intervals are longer.
   This reproduces the values quoted in Dataset_C's task-*_eeg.json.

2. Session timeline. The STATUS channel holds the amplifier's running sample
   counter (it increments by exactly 1 per sample, continues across runs and is
   identical to the 'Channel 34' column of the source CSV exports). Values are
   rounded to integers first: in a few files (e.g. sub-09 Dataset_B/C upper)
   the channel was written with a 0.1 resolution, so counters above ~1.7e6
   exceed float32 integer precision and are stored within +/-0.2 of the true
   integer; rounding recovers the exact counter.
   While the amplifier keeps running, the counter places consecutive runs on a
   common time axis, including the pauses between them. It restarts when the
   amplifier is restarted, which shows up as a run whose counter range overlaps
   an already-recorded one. Runs are therefore ordered by counter within
   "segments" (maximal sets of runs whose counter ranges do not overlap), and
   the session span (first sample of the first run to last sample of the last
   run) is reported only for participants whose whole session is a single
   segment; for the others only the recorded EEG time is reliable.

3. Drift of the ~18 Hz interference (if 06_spectral_check.py has been run):
   the per-recording line frequency (06_line_interference_18hz.csv) is
   compared between the lower- and upper-limb blocks of each participant (the
   upper-limb block is recorded after the lower-limb one) and correlated with
   the run's start time within counter-continuous sessions.

Usage:
    py -3.12 09_timing_and_session.py

Writes:
    derivatives/validation/09_cue_timing_summary.csv
    derivatives/validation/09_session_runs.csv       (one row per recording)
    derivatives/validation/09_session_summary.csv    (one row per participant)
    derivatives/validation/09_line_drift.csv         (one row per participant)
"""
import mne
import numpy as np
import pandas as pd

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_SFREQ, load_all_events

mne.set_log_level("ERROR")

DOCUMENTED = {  # Dataset_C task-*_eeg.json
    ("Dataset_C", "lower", "rest_interval"): "10.45 +/- 0.30 s (range 9.89-10.98, n=810)",
    ("Dataset_C", "upper", "rest_interval"): "8.42 +/- 0.38 s (range approx. 8.0-9.29, n=810)",
    ("Dataset_C", "lower", "movement_interval"): "2.64 +/- 0.10 s (n=900)",
    ("Dataset_C", "upper", "movement_interval"): "2.65 +/- 0.25 s (n=900)",
}


def cue_timing():
    ev = load_all_events()
    ev = ev.sort_values(["dataset", "subject", "task", "run", "onset"])
    g = ev.groupby(["dataset", "subject", "task", "run"])
    ev["next_onset"] = g["onset"].shift(-1)
    ev["next_type"] = g["trial_type"].shift(-1)
    ev["interval"] = ev["next_onset"] - ev["onset"]

    move = ev[ev.trial_type.isin(["left", "right"]) & ev.interval.notna()]
    rest_ss = ev[(ev.trial_type == "rest") & (ev.event_index > 0)
                 & ev.next_type.isin(["left", "right"])]
    first_rest = ev[(ev.trial_type == "rest") & (ev.event_index == 0) & ev.interval.notna()]

    rows = []
    for name, d in (("movement_interval", move), ("rest_interval", rest_ss),
                    ("first_rest_interval", first_rest)):
        for (ds, task), x in d.groupby(["dataset", "task"]):
            v = x["interval"]
            rows.append(dict(dataset=ds, task=task, measure=name, n=len(v),
                             mean_s=v.mean(), sd_s=v.std(), median_s=v.median(),
                             p1_s=v.quantile(0.01), p99_s=v.quantile(0.99),
                             min_s=v.min(), max_s=v.max(),
                             documented=DOCUMENTED.get((ds, task, name), "")))
    return pd.DataFrame(rows)


def session_timeline():
    rows = []
    for sub_i in range(1, N_SUBJECTS + 1):
        sub_id = f"sub-{sub_i:02d}"
        for ds, n_runs in DATASETS.items():
            for task in TASKS:
                for run in range(1, n_runs + 1):
                    vhdr = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}_eeg.vhdr"
                    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
                    c = np.round(raw.get_data(picks=["STATUS"])[0])
                    steps = np.diff(c)
                    rows.append(dict(subject=sub_id, dataset=ds, task=task, run=run,
                                     counter_start=int(c[0]), counter_end=int(c[-1]),
                                     n_samples=len(c), duration_s=len(c) / EXPECTED_SFREQ,
                                     n_bad_increments=int((steps != 1).sum())))
    runs = pd.DataFrame(rows).sort_values(["subject", "counter_start"]).reset_index(drop=True)
    runs["order"] = runs.groupby("subject").cumcount() + 1
    first = runs.groupby("subject")["counter_start"].transform("min")
    runs["start_min"] = (runs.counter_start - first) / EXPECTED_SFREQ / 60
    prev_end = runs.groupby("subject")["counter_end"].shift(1)
    runs["pause_before_s"] = (runs.counter_start - prev_end) / EXPECTED_SFREQ

    summ = []
    for sub_id, r in runs.groupby("subject"):
        n_overlap = int((r.pause_before_s < 0).sum())
        continuous = n_overlap == 0
        blocks = []  # consecutive runs of the same dataset+task, in counter order
        for key in zip(r.task, r.dataset):
            if not blocks or blocks[-1] != key:
                blocks.append(key)
        summ.append(dict(
            subject=sub_id,
            single_counter_segment=continuous,
            span_min=(r.counter_end.max() - r.counter_start.min()) / EXPECTED_SFREQ / 60
            if continuous else np.nan,
            recorded_min=r.duration_s.sum() / 60,
            recorded_A_min=r.loc[r.dataset == "Dataset_A", "duration_s"].sum() / 60,
            recorded_B_min=r.loc[r.dataset == "Dataset_B", "duration_s"].sum() / 60,
            recorded_C_min=r.loc[r.dataset == "Dataset_C", "duration_s"].sum() / 60,
            pause_median_s=r.pause_before_s.median() if continuous else np.nan,
            pause_max_min=r.pause_before_s.max() / 60 if continuous else np.nan,
            overlapping_runs=n_overlap,
            bad_counter_increments=int(r.n_bad_increments.sum()),
            block_order=" > ".join(f"{t}-{d[-1]}" for t, d in blocks) if continuous
            else "n/a (counter restarted)",
        ))
    return runs, pd.DataFrame(summ)


def line_drift(runs, summ):
    path = OUT_DIR / "06_line_interference_18hz.csv"
    if not path.exists():
        return None
    lines = pd.read_csv(path)
    m = lines.merge(runs[["subject", "dataset", "task", "run", "start_min"]],
                    on=["subject", "dataset", "task", "run"])
    m = m[m.line_present]
    continuous = set(summ.loc[summ.single_counter_segment, "subject"])
    rows = []
    for sub_id, d in m.groupby("subject"):
        by_task = d.groupby("task")["line_freq_hz"].median()
        r_time = np.nan
        if sub_id in continuous and len(d) > 4 and d.line_freq_hz.std() > 0:
            r_time = float(np.corrcoef(d.start_min, d.line_freq_hz)[0, 1])
        rows.append(dict(subject=sub_id, n_recordings_with_line=len(d),
                         freq_median_hz=d.line_freq_hz.median(),
                         freq_lower_hz=by_task.get("lower", np.nan),
                         freq_upper_hz=by_task.get("upper", np.nan),
                         upper_minus_lower_hz=by_task.get("upper", np.nan) - by_task.get("lower", np.nan),
                         r_freq_vs_time=r_time))
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timing = cue_timing()
    timing.to_csv(OUT_DIR / "09_cue_timing_summary.csv", index=False)
    print("=== Cue timing (s) ===")
    print(timing.drop(columns=["documented"]).round(2).to_string(index=False))
    doc = timing[timing.documented != ""]
    print("\nDocumented (Dataset_C task-*_eeg.json) vs measured:")
    for r in doc.itertuples():
        print(f"  {r.dataset} {r.task} {r.measure}: documented {r.documented} | measured "
              f"{r.mean_s:.2f} +/- {r.sd_s:.2f} s (range {r.min_s:.2f}-{r.max_s:.2f}, n={r.n})")

    runs, summ = session_timeline()
    runs.to_csv(OUT_DIR / "09_session_runs.csv", index=False)
    summ.to_csv(OUT_DIR / "09_session_summary.csv", index=False)
    print("\n=== Session timeline ===")
    print(summ.round(1).to_string(index=False))
    cont = summ[summ.single_counter_segment]
    print(f"\nRecorded EEG per participant: median {summ.recorded_min.median():.1f} min "
          f"(range {summ.recorded_min.min():.1f}-{summ.recorded_min.max():.1f}); per dataset medians "
          f"A {summ.recorded_A_min.median():.1f}, B {summ.recorded_B_min.median():.1f}, "
          f"C {summ.recorded_C_min.median():.1f} min")
    print(f"Participants with a continuous counter over the whole session: {len(cont)}/{len(summ)}; "
          f"first-to-last-run span: median {cont.span_min.median():.1f} min "
          f"(range {cont.span_min.min():.1f}-{cont.span_min.max():.1f}); "
          f"median pause between runs {cont.pause_median_s.median():.1f} s")
    print(f"Counter integrity after rounding: {int(summ.bad_counter_increments.sum())} non-unit "
          f"increments within runs")
    print("Block orders (continuous-counter participants):")
    print(cont.block_order.value_counts().to_string())

    drift = line_drift(runs, summ)
    if drift is not None:
        drift.to_csv(OUT_DIR / "09_line_drift.csv", index=False)
        both = drift.dropna(subset=["upper_minus_lower_hz"])
        r = drift.r_freq_vs_time.dropna()
        print("\n=== ~18 Hz line drift ===")
        print(f"Participants with the line: {len(drift)}; participant medians "
              f"{drift.freq_median_hz.min():.2f}-{drift.freq_median_hz.max():.2f} Hz")
        print(f"Upper minus lower block ({len(both)} participants with the line in both): "
              f"mean {both.upper_minus_lower_hz.mean():.2f} Hz, min {both.upper_minus_lower_hz.min():.2f}, "
              f"max {both.upper_minus_lower_hz.max():.2f}; lower in {(both.upper_minus_lower_hz < 0).sum()}")
        print(f"Correlation with time into session (continuous-counter participants, n={len(r)}): "
              f"median r {r.median():.2f}; r > 0.5 in {(r > 0.5).sum()}, r < -0.5 in {(r < -0.5).sum()}")


if __name__ == "__main__":
    main()
