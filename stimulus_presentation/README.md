# Stimulus-presentation scripts

The pygame scripts used during acquisition. The only change is that the image
folders, which pointed to the acquisition computer, now point to the deposited
stimuli (`Dataset_A/stimuli/`, `Dataset_C/stimuli/`), and `Task_1.py` reads only
the `.svg` files there, in alphabetical order. The stimuli were shown on a second
monitor; a 70 x 80 px square in the bottom-left corner, covered by the photoresistor of the event-
synchronisation system, changed colour at every cue.

| Script | Task | Dataset | Cues |
|---|---|---|---|
| `Task_1.py` | calibration | Dataset_A | arrow images + 1,000 Hz, 100 ms beep for movement; rest image |
| `Task_2.py` | open loop | Dataset_B | beep only, blank screen |
| `Task_3.py` | simulated closed loop | Dataset_C | beep, then the mirror animation (24 frames/s) |

Each run: 10 movement cues per side (Task 1) or 10 movement cues (Tasks 2 and 3),
2.5 s movement-cue period, 5-6 s random rest (0.1 s steps), 3 s first and last rest.
`winsound.Beep` blocks for the duration of the tone, so the measured movement-cue
period is about 0.1 s longer than programmed (Task 1 also includes rendering the
image). In `Task_3.py`, `miembro` (Inferior = foot, Superior = hand) and
`lateralidad` (left/right, the participant's dominant side) were set before each
recording. The stimulus files are in `Dataset_A/stimuli/` and `Dataset_C/stimuli/`.
