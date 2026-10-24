"""Technical Validation #2d - Objective bad-channel detection (proposal only).

Detects scalp channels that are unusable because of a recording problem (not a
transient artifact), using two reproducible criteria evaluated on every
recording and then aggregated per subject. Electrodes are placed once per
subject for the whole ~2 h session, so a genuine electrode problem shows up in
most of that subject's 20 recordings; a channel is proposed as bad for the
subject only when the criterion holds in at least MIN_FRACTION of them.

1. Gel bridge: two electrodes short-circuited by electrolyte record the same
   signal, including their high-frequency noise. The Pearson correlation is
   computed on the 30-45 Hz band-passed signal (native reference), where
   genuinely neighbouring channels are only moderately correlated (typically
   < 0.96, even for O1-Oz with strong alpha or Fp1-Fpz with blinks), unlike the
   broadband signal (dominated by alpha/blinks, r > 0.98 for many healthy
   neighbours). A pair is bridged in a recording if r >= BRIDGE_R.
2. Noisy channel: std of the 1-45 Hz signal divided by the median std of its
   N_NEIGHBOURS nearest neighbours (standard_1020 positions) is >= NOISE_RATIO.
   Comparing to neighbours (not to the corpus-wide channel median) avoids false
   positives at channels next to the CPz reference, whose amplitude is
   naturally small.

Tiers. High 30-45 Hz correlation alone is not specific: subjects with strong,
spatially broad muscle (EMG) activity (one component explaining 75-90% of their
30-45 Hz variance, vs ~40% corpus median) make healthy neighbours correlate
above BRIDGE_R, and neither correlation nor the normalised electrical distance
var(a-b)/mean(var a, var b) separates such pairs cleanly from genuine bridges
(overlap around 0.01). What does separate the unambiguous case (a gel smear
over several electrodes with poor contact) is that the bridged electrodes
also carry an abnormal amplitude. Hence:
  - tier "bad": noisy (criterion 2), or member of a bridged group (connected
    component of bridged pairs) whose median amplitude is >= AMP_RATIO x the
    corpus-wide median of those channels. Amplitude = std of the 0.5 Hz
    high-passed signal (as std_uv_hp in 03_signal_quality_check.py): the excess
    of a smeared, poorly contacting electrode group is mostly slow (0.5-1 Hz)
    drift, which a 1 Hz cut would hide. The decision is per group so that a
    bridged group is never split between tiers;
  - tier "possible": bridged group with normal amplitude - documented, not
    marked bad.

Nothing in the BIDS datasets is modified: this script only writes a proposal
to derivatives/. Writing the approved channels as status=bad in _channels.tsv is
a separate step.

Usage:
    py -3.12 07_bad_channels.py [--subjects sub-01 sub-02 ...]

Writes:
    derivatives/validation/07_bad_channels_per_recording.csv
    derivatives/validation/07_bad_channels_proposal.csv
"""
import argparse
import itertools

import mne
import numpy as np
import pandas as pd

from common import BIDS_ROOT, OUT_DIR, DATASETS, TASKS, N_SUBJECTS, EXPECTED_CHANNELS

mne.set_log_level("ERROR")

EEG_CHANNELS = [ch for ch in EXPECTED_CHANNELS if ch not in ("TRIG", "STATUS")]

BRIDGE_BAND = (30.0, 45.0)
BRIDGE_R = 0.985
NOISE_BAND = (1.0, 45.0)
NOISE_RATIO = 3.0
N_NEIGHBOURS = 4
MIN_FRACTION = 0.75  # of the subject's recordings
AMP_RATIO = 4.0      # bridged group + this x corpus channel median std -> tier "bad"
AMP_HIGHPASS = 0.5   # Hz, amplitude measure (matches std_uv_hp of script 03)


def nearest_neighbours():
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    lookup = {k.lower(): v for k, v in pos.items()}
    xyz = np.array([lookup[c.lower()] for c in EEG_CHANNELS])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return {i: list(np.argsort(d[i])[1:N_NEIGHBOURS + 1]) for i in range(len(EEG_CHANNELS))}


NEIGHBOURS = nearest_neighbours()
PAIRS = list(itertools.combinations(range(len(EEG_CHANNELS)), 2))


def recording_metrics(ds, sub_id, task, run):
    """Return (noise_ratio per channel (n_ch,), high-band r per pair (n_pairs,),
    0.5 Hz high-passed std per channel (n_ch,), uV) or None."""
    vhdr = BIDS_ROOT / ds / sub_id / "eeg" / f"{sub_id}_task-{task}_run-{run:02d}_eeg.vhdr"
    if not vhdr.exists():
        return None
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    amp_std = raw.copy().filter(AMP_HIGHPASS, None, verbose=False).get_data(units="uV").std(axis=1)
    broad = raw.copy().filter(*NOISE_BAND, verbose=False).get_data()
    high = raw.filter(*BRIDGE_BAND, verbose=False).get_data()

    std = broad.std(axis=1)
    ratio = np.array([std[i] / np.median(std[NEIGHBOURS[i]]) for i in range(len(std))])
    r = np.corrcoef(high)
    return ratio, np.array([r[a, b] for a, b in PAIRS]), amp_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
    suffix = "_partial" if args.subjects else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rec_rows, proposal_rows = [], []
    per_subject = {}  # sub_id -> (ratios (n_rec, n_ch), rs (n_rec, n_pairs), stds (n_rec, n_ch))
    n_recordings = 0
    for sub_id in subjects:
        ratios, rs, stds = [], [], []
        for ds, n_runs in DATASETS.items():
            for task in TASKS:
                for run in range(1, n_runs + 1):
                    res = recording_metrics(ds, sub_id, task, run)
                    n_recordings += 1
                    if n_recordings % 50 == 0:
                        print(f"...{n_recordings} recordings processed", flush=True)
                    if res is None:
                        continue
                    ratio, r, std = res
                    ratios.append(ratio)
                    rs.append(r)
                    stds.append(std)
                    for i in np.flatnonzero(ratio >= NOISE_RATIO):
                        rec_rows.append(dict(dataset=ds, subject=sub_id, task=task, run=run,
                                             channel=EEG_CHANNELS[i], criterion="noisy",
                                             value=round(float(ratio[i]), 2)))
                    for k in np.flatnonzero(r >= BRIDGE_R):
                        a, b = PAIRS[k]
                        rec_rows.append(dict(dataset=ds, subject=sub_id, task=task, run=run,
                                             channel=f"{EEG_CHANNELS[a]}-{EEG_CHANNELS[b]}",
                                             criterion="bridged", value=round(float(r[k]), 4)))
        if ratios:
            per_subject[sub_id] = (np.array(ratios), np.array(rs), np.array(stds))

    # Corpus-wide median std per channel (reference for the bridged-amplitude test).
    corpus_std = np.median(np.vstack([v[2] for v in per_subject.values()]), axis=0)

    for sub_id, (ratios, rs, stds) in per_subject.items():
        amp = np.median(stds, axis=0) / corpus_std  # (n_ch,)

        noisy_frac = (ratios >= NOISE_RATIO).mean(axis=0)
        for i in np.flatnonzero(noisy_frac >= MIN_FRACTION):
            proposal_rows.append(dict(
                subject=sub_id, channel=EEG_CHANNELS[i], tier="bad", criterion="noisy",
                fraction_of_recordings=round(float(noisy_frac[i]), 2),
                amplitude_vs_corpus=round(float(amp[i]), 1),
                detail=f"std {np.median(ratios[:, i]):.1f}x its {N_NEIGHBOURS} nearest neighbours, "
                       f"{amp[i]:.1f}x corpus median"))

        bridge_frac = (rs >= BRIDGE_R).mean(axis=0)
        partners = {}
        for k in np.flatnonzero(bridge_frac >= MIN_FRACTION):
            a, b = PAIRS[k]
            partners.setdefault(a, []).append((b, k))
            partners.setdefault(b, []).append((a, k))
        # Bridged groups = connected components of the bridged-pair graph.
        group_of, groups = {}, []
        for start in sorted(partners):
            if start in group_of:
                continue
            stack, members = [start], []
            group_of[start] = len(groups)
            while stack:
                n = stack.pop()
                members.append(n)
                for m, _ in partners[n]:
                    if m not in group_of:
                        group_of[m] = len(groups)
                        stack.append(m)
            groups.append(sorted(members))
        group_amp = [float(np.median(amp[g])) for g in groups]
        for i, lst in sorted(partners.items()):
            ks = [k for _, k in lst]
            g = group_of[i]
            proposal_rows.append(dict(
                subject=sub_id, channel=EEG_CHANNELS[i],
                tier="bad" if group_amp[g] >= AMP_RATIO else "possible", criterion="bridged",
                group="/".join(EEG_CHANNELS[j] for j in groups[g]),
                group_amplitude_vs_corpus=round(group_amp[g], 1),
                fraction_of_recordings=round(float(bridge_frac[ks].min()), 2),
                amplitude_vs_corpus=round(float(amp[i]), 1),
                detail=f"30-45 Hz r {np.median(rs[:, ks]):.3f} with "
                       + "/".join(EEG_CHANNELS[j] for j, _ in lst)
                       + f"; std {amp[i]:.1f}x corpus median"))

    rec_df = pd.DataFrame(rec_rows)
    rec_out = OUT_DIR / f"07_bad_channels_per_recording{suffix}.csv"
    rec_df.to_csv(rec_out, index=False)
    prop = pd.DataFrame(proposal_rows)
    prop_out = OUT_DIR / f"07_bad_channels_proposal{suffix}.csv"
    prop.to_csv(prop_out, index=False)

    print(f"\nPer-recording flags written to {rec_out} ({len(rec_df)} rows)")
    print(f"Proposal written to {prop_out}\n")
    if prop.empty:
        print("No channel meets the persistent bad-channel criteria.")
    else:
        for tier in ("bad", "possible"):
            t = prop[prop["tier"] == tier]
            print(f"--- tier '{tier}': {len(t)} channel(s) in {t['subject'].nunique()} subject(s)")
            if len(t):
                print(t.drop(columns="tier").to_string(index=False))
            print()

    # Near-misses, to judge how sharp the thresholds are.
    if not rec_df.empty:
        n_rec_per_sub = 4 * 2 + 3 * 2 + 3 * 2
        freq = (rec_df.groupby(["subject", "channel", "criterion"]).size() / n_rec_per_sub).rename("fraction")
        near = freq[(freq >= 0.25) & (freq < MIN_FRACTION)].reset_index()
        print(f"\nNear-misses (flagged in 25-75% of the subject's recordings):")
        print(near.round(2).to_string(index=False) if len(near) else "  none")


if __name__ == "__main__":
    main()
