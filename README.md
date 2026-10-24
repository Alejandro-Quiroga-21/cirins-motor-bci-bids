# cirins-motor-bci-bids

Conversion, technical-validation and stimulus-presentation code of the **CIRINS Motor
BCI EEG dataset**: 32-channel EEG of 30 healthy volunteers performing executed hand
and ankle movements in three BCI-like tasks (calibration, open loop and simulated
closed loop), released as three BIDS datasets.

- **Data:** Zenodo, https://doi.org/10.5281/zenodo.17980607 (this DOI always points to the
  latest version of the deposit). The deposit also contains a copy of this code in
  `code.zip`.
- **Paper:** Data Descriptor submitted to *Scientific Data* (citation to be added
  on publication).

## Using this code with the data

The scripts locate the data relative to their own position. Download and extract
the Zenodo deposit, then put this repository in place of its `code/` folder:

    <deposit>/
    ├── Dataset_A/  Dataset_B/  Dataset_C/
    ├── sourcedata/
    ├── derivatives/
    └── code/        <- this repository

for example with `git clone https://github.com/<user>/cirins-motor-bci-bids code`
inside the extracted deposit.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0 (see the deposit).

---

# Code of the deposit

Python 3.12; package versions are pinned in `requirements.txt`
(`python -m pip install -r requirements.txt`).

The scripts locate the data relative to their own position, so the deposit must
keep its layout: `Dataset_A/`, `Dataset_B/`, `Dataset_C/`, `sourcedata/`,
`code/` and `derivatives/` side by side. No script modifies the released EEG
samples. Every script prints its usage with `--help` or in its docstring.

## 1. Rebuilding the BIDS datasets from the original exports

    python csv_to_bids.py --out-root <folder>            # recordings + dataset-level files
    python validation/07_bad_channels.py                 # bad-channel proposal
    python validation/07b_write_bad_channels.py --bids-root <folder>

`bids_root_files/` holds the hand-curated dataset-level files (participants,
channels, events.json, dataset_description.json, README, CHANGES, CITATION.cff)
that `csv_to_bids.py` copies into each dataset. The `stimuli/` folders of
Dataset_A (cue images) and Dataset_C (animation frames) are copied by hand from
the acquisition computer.

`stimulus_presentation/` holds the pygame scripts that presented the cues
(see its README).

## 2. Technical validation

Run from `code/validation/`, in this order. Outputs go to
`derivatives/validation/`; the files distributed there are the reference
outputs against which a re-run can be compared.

| Step | Script | Main outputs |
|---|---|---|
| Conversion check | `00_roundtrip_csv_brainvision.py` (needs sourcedata/) | `00_roundtrip_csv_brainvision.csv` |
| Integrity | `01_structural_and_design_check.py` | `01_structural_and_design_check_issues.csv`, `01_ttl_value_encoding_summary.csv` |
| Integrity | `02_flag_event_anomalies.py` | `flagged_events.tsv` |
| Signal quality | `03_signal_quality_check.py` | `03_signal_quality_summary.csv` |
| Spectrum, ~18 Hz line | `06_spectral_check.py`, `06c_spectral_stats.py`, `06b_spectral_figure.py` | `06_spectral_subject_summary.csv`, `06_line_interference_18hz.csv`, `06c_spectral_stats.csv`, `Figure_Spectral.*` |
| Bad channels | `07_bad_channels.py`, `07b_write_bad_channels.py` | `07_bad_channels_per_recording.csv`, `07_bad_channels_proposal.csv` |
| Gel bridges (electrical distance) | `07c_bridged_electrodes_ed.py` | `07c_bridged_pairs_per_recording.csv`, `07c_bridged_pairs_per_participant.csv` |
| Timing, session | `09_timing_and_session.py` | `09_cue_timing_summary.csv`, `09_session_runs.csv`, `09_session_summary.csv`, `09_line_drift.csv` |
| Evoked potentials | `04_evoked_potential_check.py` and `04b_evoked_topomap.py`, each once without and once with `--clean`; then `04c_paper_figure.py`; `04d_dataset_a_audiovisual.py` (after 06) | `04_evoked_potential_*`, `04b_topomap_channel_data*`, `Figure_VEP_AEP*`, `04d_dataset_a_audiovisual_*` |
| ERD/ERS | `05_erd_ers_check.py` without and with `--clean`; `05c_erd_ers_stats.py`; `05b_erd_ers_figure.py`; `05d_beta_rebound_latency.py` | `05_erd_ers_*`, `05c_erd_ers_limb_stats.csv`, `05d_beta_rebound_latency.csv`, `Figure_ERD_ERS*` |
| Robustness | `08_robustness_compare.py` | `08_robustness_evoked.csv`, `08_robustness_erd.csv` |
| Decoding benchmark | `10_classification_benchmark.py`, `10b_classification_figure.py` | `10_classification_*.csv`, `Figure_Decoding.*` |
| Benchmark controls | `11_classification_controls.py` | `11_classification_controls_*.csv` |
| Horizontal eye movements (Dataset A) | `12_heog_left_right.py` | `12_heog_left_right_subjects.csv`, `12_heog_left_right_summary.csv` |

`--clean` interpolates the channels marked as bad and removes the ~18 Hz line
(`common.clean_raw`); the values reported in the Data Descriptor are the
`_clean` ones, and `08_robustness_compare.py` compares them with the results on
the data exactly as released. `06_spectral_check.py` must run before any
`--clean` step, which reads `06_line_interference_18hz.csv`.

The evoked potentials are time-locked to the cues that carry a stimulus: every
cue in Dataset_A (image onsets) and the movement cues only in Dataset_B and
Dataset_C (the beep sounds at movement cues; rest onsets carry no sound). In
Dataset_A the movement cues (arrows) were also accompanied by the beep;
`04d_dataset_a_audiovisual.py` compares the fronto-central response to movement
and rest cues of Dataset_A.
