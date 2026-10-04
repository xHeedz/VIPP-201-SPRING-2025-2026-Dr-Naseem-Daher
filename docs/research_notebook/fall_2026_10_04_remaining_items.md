# sun 4 oct 2026: remaining technical items, DriveDNA sample, October deck

## UAH lane estimator state (finding 10)

`datasets/uah.py` `load_trip(trip, lane_state=2)`: only lane rows whose estimator state is 2 (detected) are used now (column 4 of PROC_LANE_DETECTION). leave one driver out (`data/uah_lane_state.csv`):

| lane rows | hand-set index AUC | agent AUC | window accuracy (hand-set) |
|---|---|---|---|
| every state | 0.829 | 0.761 | 0.768 |
| detected only | 0.831 | 0.762 | 0.769 |

lane coverage 0.996 to 0.975 (a detected row is almost always within the 1 s matching tolerance). one second of the hand checked aggressive window changes (lane offset 0.134 to 0.127 m), score 86.89 to 86.87; tests and pages updated. all UAH outputs regenerated: hand-set AUC 0.831, trip accuracy with the cut off 42: 0.933 (0.967 before this change), fitted UAH cut off 41.1.

## pNEUMA, all ten drones of one slot

24 Oct 2018, 08:30 to 09:00, drones 1 to 10 (about 3 GB, `../28:9:2026/pneuma/`): 15,306 scored vehicles (cars, taxis, buses, trucks), 149,917 windows. each drone processed on its own (local frame, leader search), cached as `*.per_second.csv.gz`.

| | metres | headway |
|---|---|---|
| median score | 48.4 | 15.5 |
| windows labelled aggressive (>= 42) | 58.9% | 8.4% |
| median score by quartile of vehicles within 50 m | 35.1 / 47.8 / 52.7 / 56.6 | 10.2 / 15.7 / 17.8 / 18.6 |

same pattern as NGSIM: in metres the score climbs with crowding, driven by proximity (41 points on average).

## US-101 like layout (item 5): attempted, not solved

`planted_drivers.py` scenario `us101`: 5 lanes at 65 mph, on ramp onto an auxiliary lane, off ramp 400 m later; `scripts/us101_sweep.py`, `data/us101_sweep.csv`.

- first sweep (6,000 to 9,000 veh/h): gridlock at every demand. cause: cars bound for the off ramp started in any lane, stopped at the end of the auxiliary lane waiting to cut in, and blocked it (64 to 88% of rows stopped in auxiliary lanes 1 to 3, even at 3,000 veh/h). fixed: exit bound cars start on the right (`departLane="0"`).
- after the fix: best 7,000 + 1,200 veh/h, median speed 38 km/h (US-101 48), headway KS 0.14, speed KS 0.43. the speed distribution stays split between stopped (25th percentile 0 km/h) and free (75th percentile 83 km/h), no better than the lane drop network (0.43 to 0.46).
- reading: with IDM the traffic breaks down into stop and go instead of holding synchronized flow. reproducing US-101 speeds likely needs a different car following model, not a different layout. left open.

## RL ego, urban junction (item 6)

`scripts/sumo_urban_agent.py`:
- bug: collisions only warn (the ego keeps driving), and the crash flag was read on policy steps only (every third simulation step), so two of three collisions never reached the reward. now any collision since the last policy step is penalised.
- crossing traffic counted as a threat only within 8 to 15 m and +-10 m along the ego's path; at 14 m/s the ego needs about 20 m to stop. now the threat distance is at least the stopping distance (decel 5 m/s2) plus 5 m, window +-25 m.
- result (200 episodes): episodes with a junction collision 3 before, 4 after (episodes 1, 39, 85, 134; 39 and 85 identical in both runs, a crossing NPC drives into the ego); episodes with reward below 150 in the last 100: 2 before, 8 after. not solved; the collisions are rare, early (exploration) and partly caused by NPCs. the crash flag fix is kept (it was a bug).

## DriveDNA

full release (HenryYHW/DriveDNA, 465 drivers): gated, request sent, waiting for the authors. DriveDNA-Sample (auto approved): 63 drives, 5 drivers, 14 cars; only the signal CSVs downloaded (0.55 GB, `../28:9:2026/drivedna_sample/`). loader `datasets/drivedna.py`: radar gap (`leadOne_dRel`), lane offset from `laneLeft_y` / `laneRight_y`, human driving only. 8 of 14 car models log no lane positions, so the comparisons use the three term score (no lane term for any car). `scripts/score_drivedna.py`, `data/drivedna_summary.csv`:

| | metres | headway |
|---|---|---|
| slow car following (< 30 km/h, radar leader), 2,163 windows: median score | 51.3 | 7.2 |
| same: labelled aggressive | 77.3% | 0.3% |
| free driving (>= 60 km/h, no leader), 3,106 windows: median score | 18.3 | 18.3 |
| same: labelled aggressive | 1.2% | 1.3% |
| driver_078, spread of the per car median across 13 cars (sd) | 7.2 | 7.1 |
| shared Honda Civic, range of the per driver median (4 drivers) | 13.3 | 14.6 |

real radar car following confirms the density effect without simulation or video tracking: in a queue the metres index calls three windows in four aggressive, headway almost none; in free driving the two agree. one driver's score still moves by about 7 points from car to car (vehicle and route effects, the confound DriveDNA is built to measure); different drivers in the same car differ by about 13 points.

## October deck

`scripts/build_october_deck.py` appends 14 slides to the September deck (18 in total, same fonts, colours and layout), written to `../28:9:2026/Presentation/VIPP 301A Session Results October 2026.pptx` (the September file is not changed). native charts from `data/`, the planted jam animation (`scripts/planted_animation.py`, `results/figures/planted_jam_animation.gif`: true type, metres label and headway label of the same cars), and the US-101 replay video. checked by rendering every slide through Keynote; the skill validator needs Python 3.10 and was not run.

## status

- [x] UAH lane estimator state
- [x] pNEUMA all ten drones of one slot
- [ ] US-101 synchronized flow (IDM breaks down instead; needs another car following model)
- [~] RL ego junction collisions (crash flag bug fixed, collisions still in 4 of 200 episodes)
- [x] DriveDNA sample scored; full release waiting for approval
- [ ] 100-DrivingStyle: email to the authors (Hadi)
- [x] October deck
