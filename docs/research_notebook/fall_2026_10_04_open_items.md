# sun 4 oct 2026: open items after parts 1 to 4

## 1. SUMO driver types calibrated to NGSIM US-101 (`scripts/planted_tau_sweep.py`, `data/planted_tau_sweep.csv`)

a common multiplier on IDM tau (ratios between the types kept), two demands, two seeds, 600 s; KS distance against US-101 for the rows before the lane drop:

| tau factor | demand (veh/h) | median speed (km/h) | KS speed | median headway (s) | KS headway |
|---|---|---|---|---|---|
| 0.70 | 3600 | 34.9 | 0.41 | 1.64 | 0.11 |
| 0.70 | 4500 | 24.1 | 0.43 | 1.55 | 0.09 |
| 0.85 | 4500 | 17.6 | 0.46 | 1.74 | 0.18 |
| 1.00 (plan table) | 3600 | 25.6 | 0.42 | 2.15 | 0.31 |
| 1.00 (plan table) | 4500 | 21.8 | 0.44 | 1.94 | 0.28 |

US-101: median speed 48.2 km/h, median headway 1.49 s. adopted: tau x 0.7, i.e. conservative 1.26 s, normal 0.84 s, aggressive 0.42 s (`scripts/planted_drivers.py`). all 70 runs repeated. in the jam the headway distribution now matches US-101 (KS 0.087, median 1.52 s against 1.49 s; before 0.309 and 2.06 s). the speed distribution does not (KS 0.46): the lane drop network produces stop and go or free flow, never the synchronized 40 to 60 km/h flow of US-101; that needs a different network (a long weaving section), not a parameter.

a first sweep was wrong: the worker processes kept the module state, so each job multiplied the tau already scaled by the previous job. caught because different factors gave identical rows; fixed with one process per job.

calibrated planted drivers, AUC aggressive vs normal (`data/planted_summary.csv`; uncalibrated in `data/planted_summary_tau1.csv`):

| scenario | reference (metres) | headway variant | UAH agent |
|---|---|---|---|
| highway low | 0.884 | 0.843 | 0.883 |
| highway medium | 0.866 | 0.831 | 0.814 |
| jam | 0.746 | 0.762 | 0.638 |
| merge | 0.789 | 0.844 | 0.705 |
| roundabout | 0.924 | 0.975 | 0.884 |
| urban | 0.868 | 0.904 | 0.857 |
| weather | 0.807 | 0.824 | 0.834 |

with the types closer together in absolute headway, the advantage headway had by construction shrank: headway is now ahead in 5 of 7 settings (behind on the free and medium highway).

## 2. label cut offs fitted (`scripts/fit_cutoffs.py`, `data/cutoffs.csv`, `data/cutoffs_fitted.json`)

Youden J on labelled data; held out (UAH leave one driver out, SUMO split by seed):

| data | proximity | cut offs | aggressive recall | normal labelled aggressive | balanced accuracy |
|---|---|---|---|---|---|
| UAH | metres | 35 / 70 | 22.6% | 0.7% | 0.609 |
| UAH | metres | fitted (41) | 74.1% | 23.1% | 0.755 |
| UAH | headway | fitted (41) | 77.7% | 33.8% | 0.720 |
| SUMO | metres | 35 / 70 | 4.7% | 0% | 0.447 |
| SUMO | metres | fitted (29 / 43) | 84.5% | 24.7% | 0.655 |
| SUMO | headway | fitted (28 / 43) | 82.5% | 15.8% | 0.723 |

fitted on all data: UAH aggressive 41.2 (metres), 40.3 (headway); SUMO conservative 29.9 / 28.0, aggressive 42.9 / 43.1. adopted in the reference (decision 4 Oct): **conservative < 29, aggressive >= 42** (`model/aggressiveness_model.py` THRESHOLDS). every label in the repo follows (ellipse colours, context fallback, SUMO scripts, tests).

effect:
- UAH leave one driver out, hand-set index: window accuracy 0.768 (0.70 with 70), trip accuracy 0.967 (0.60 with 70; the trained agent: 0.90).
- planted SUMO, vehicle level (all 70 runs): aggressive recall 87.1% (5612 of 6441), normal labelled aggressive 27.5% (5388 of 19592), conservative recall 66.0%.
- GT vs agent agreement (`scripts/sumo_runner.py`, 5 seeds; part 1 numbers were with 35 / 70): highway 84.5% (sd 0.3; always conservative would give 61.6%), urban 98.7% (66.4%), weather 90.7% (88.9%). the GT now labels 21.5% (highway), 30.6% (urban) and 4.9% (weather) of the samples aggressive, against 1.2%, 5.8% and 0.4% before: the planted aggressive vehicles of the validation scenarios are recognised.
- warning: with the gap in metres, dense traffic is now labelled mostly aggressive: 60 to 79% of NGSIM windows (US-101 61.8%, I-80 79.1%, Lankershim 60.1%, Peachtree 41.0%, pNEUMA 43.6%), and 74% of the planted normal vehicles in the jam (4214 of 5713; 59% of their windows, against 10% on the free highway). with headway the NGSIM shares are 6.9 to 37.9% and the planted normal windows in the jam 33%. the fitted cut offs make the density problem sharper, not smaller: shares of aggressive drivers in dense traffic should not be reported in metres. the metres vs headway decision is the next one.

## 3. labelled dense traffic data: search

no open dataset combines human aggressiveness labels, gap to the vehicle ahead and dense traffic. checked:

| dataset | labels | signals | dense | access |
|---|---|---|---|---|
| 100-DrivingStyle (Zhang, Wang, Chen, Xi, arXiv 2406.07894, 2024) | human: per driver aggressiveness, 5 point scale, self and in-car expert, 100 drivers | speed, accel, steering, pedals (100 Hz); no gap, no lane | urban + highway, not stated | repository in the paper returns 404; email to the authors drafted (`../drafts/email_100_drivingstyle.md`) |
| DriveDNA (Hugging Face HenryYHW/DriveDNA, 2026) | rule generated primitives only (close following, hard braking, ...) | speed, accel, lane offset, lead distance, headway, TTC (10 Hz), 465 drivers | yes, stop and go context | gated research licence (Hugging Face account), about 329 GB in total, signal tables only needed |
| POLIDriving (Quito, 2024) | accident risk level, 1,980 expert verified rows | OBD speed, accel, GPS; no gap | yes, heavy urban traffic | public (GitHub laboratorioAI/polidriving) |
| CAN-DSAD (IEEE DataPort) | aggressive braking / lane change / acceleration events, 16 drivers | CAN | unclear | DataPort account |
| HDD (Honda) | manoeuvres and causes (congestion), not style | CAN, video, LiDAR | yes | request |
| SHRP2, 100-Car | crash and near crash events | full | yes | formal application |

decision (4 Oct): combine 100-DrivingStyle (human labels, speed and accel terms) with DriveDNA (all terms in dense traffic, driver level consistency). both wait on Hadi: sending the email, accepting the DriveDNA licence and logging in with `huggingface-cli login`.

## 4. hand checks: one arterial row and one pNEUMA second (`docs/hand_checks/hand_check_arterial_pneuma.md`)

- NGSIM Lankershim, vehicle 100065 (period 1, id 65), t = 29 s: speed 13.35 km/h from the 2-D path, accel -1.202 m/s2, gap 4.61 m, wave 0.224 m (lane centre per 10 m of road): score 77.12, equal to the pipeline.
- pNEUMA, car 19, second 37: speed 13.67 km/h, accel -0.913 m/s2, leader car 46 stopped 13.76 m ahead in the cone, gap 9.26 m: score 57.19, equal to the pipeline (difference 6e-12). the first hand version skipped stopped vehicles as leaders and disagreed; the loader keeps a stopped car's last heading on purpose (queues), and the hand check now does the same.
- the arterial check found a real problem first: Lankershim and Peachtree curve in the fixed Local_X frame (within lane spread of Local_X about 2.1 m), so one lane centre per lane turned curvature into wave. fixed with a centre per 10 m of road (offset above 1.5 m: 24.3% to 1.9% of rows on Lankershim); part 3 numbers updated.
- tests: `tests/test_hand_checks.py` (arterial and pNEUMA rows hard coded).

## 5. RL ego: right of way and lane change safety

| | collisions (200 epochs) | mean reward, last 100 epochs | epochs with reward < 150 |
|---|---|---|---|
| urban, speed mode 6 | 14 | | |
| urban, speed mode 30 (+ right of way, + red light) | 13 to 14 | 195.9 | 2 |
| urban, speed mode 31 (all safety) | 0 | 145.0 | 43 |
| highway, lane change mode 0 | 176 | | |
| highway, lane change mode 512 | 1 | | |

- highway: `setLaneChangeMode(EGO_ID, 512)` (lane changes the agent asks for respect the safe gaps of others) removes the side collisions: 176 to 1; reward of the last epoch 276.8 to 285.0.
- urban: speed mode 30 keeps the junction collisions rare (13 in 200 episodes, all in the first 27 s when the ego crosses the junction); speed mode 31 removes them but the ego stalls in 43 of the last 100 episodes (the phantom stop the spring work removed). kept: 30. removing the last collisions needs a change in the RL task (for example a reward for yielding), not a speed mode.

## status

- [x] tune SUMO tau to NGSIM (headway matches, speed does not)
- [x] fit the cut offs and adopt them (29 / 42)
- [x] search for labelled dense data; route chosen
- [ ] 100-DrivingStyle: email the authors (Hadi)
- [ ] DriveDNA: accept the licence, `huggingface-cli login` (Hadi), then download the signal tables
- [x] hand check one arterial row and one pNEUMA second
- [x] ego right of way (highway fixed; urban rare collisions left, needs an RL change)
- [ ] metres vs headway (Dr. Daher): every result now argues for headway in dense traffic, UAH free flow is the only exception (0.03 AUC)
