# sun 4 oct 2026: part 2, planted drivers in SUMO

goal: every simulated driver has a true label, so the index is scored against ground truth instead of against a second copy of itself (the GT vs agent agreement of part 1).

## setup (`scripts/planted_drivers.py`)

three vTypes, IDM car following, SL2015 sublane lane changing (`--lateral-resolution 0.8`), mix 20 / 60 / 20, label in the vehicle id:

| type | tau (s) | accel (m/s2) | decel (m/s2) | speedFactor | lcAssertive | sigma |
|---|---|---|---|---|---|---|
| conservative | 1.8 | 1.5 | 3.0 | 0.9 | 0.5 | 0.2 |
| normal | 1.2 | 2.6 | 4.5 | 1.0 | 1.0 | 0.5 |
| aggressive | 0.6 | 3.5 | 6.0 | 1.2 | 3.0 | 0.7 |

lateral imprecision `lcSigma` 0.2 for every type (same value on purpose): wave can only separate types through lane change behaviour, not through a planted lateral parameter.

scenarios, 10 seeds each (70 runs, 900 s, first 120 s discarded, 0.1 s steps):

| scenario | network | demand | state (seed 0) |
|---|---|---|---|
| highway low | 3 lanes, 3 km, 120 km/h, lane drop 3 to 2 at 2.2 km | 1200 veh/h | free flow, median 112 km/h |
| highway medium | same | 2400 veh/h | near capacity, 5% of rows before the drop below 30 km/h |
| jam | same | 4500 veh/h | queue at the drop, 58% below 30 km/h |
| merge | 2 lane urban motorway, on ramp, 300 m acceleration lane | 3000 veh/h (20% from the ramp) | 14% of rows stopped |
| urban | 4 arm traffic light intersection, 2 lanes, 50 km/h | 1200 veh/h | 20% stopped (red phases) |
| roundabout | single lane, 4 arms, 50 km/h | 1000 veh/h | 11% stopped |
| weather | highway medium, every type speedFactor x 0.8, decel x 0.7 | 2400 veh/h | |

the density levels were measured, not guessed: a first version with a 1 lane work zone jammed at every demand (cars stall at the end of the closing lane, 12% of rows below 30 km/h even at 600 veh/h); with `departLane="best"` every car was inserted in the lanes that continue, which capped insertion and never formed a queue. both fixed before the 70 runs used here.

one code path: every vehicle logged at 10 Hz (speed, accel, gap to the leader + minGap, offset from the lane centre, lane, edge), reduced to one row per second the UAH way (`datasets/sumo_log.py`: speed once per second, 3 s mean, derivative as accel), then `datasets/uah.py` `windows` (10 s, step 5 s) and `index_features`, exactly like UAH. raw logs: `data/sumo_planted/` (not in git, about 270 MB); summaries `data/planted_*.csv`.

## results (`scripts/planted_eval.py`, mean and 95% CI over 10 seeds)

AUC aggressive vs normal, window level (`results/figures/planted_auc.png`):

| scenario | reference (gap in m) | time headway variant | UAH trained agent | conservative vs normal (reference) |
|---|---|---|---|---|
| highway low | 0.897 +- 0.009 | 0.891 +- 0.007 | 0.876 +- 0.011 | 0.707 |
| highway medium | 0.841 +- 0.031 | 0.878 +- 0.016 | 0.747 +- 0.017 | 0.721 |
| jam | 0.717 +- 0.013 | 0.790 +- 0.024 | 0.659 +- 0.007 | 0.673 |
| merge | 0.754 +- 0.012 | 0.863 +- 0.013 | 0.680 +- 0.008 | 0.673 |
| urban | 0.876 +- 0.007 | 0.931 +- 0.006 | 0.863 +- 0.006 | 0.707 |
| roundabout | 0.916 +- 0.007 | 0.981 +- 0.006 | 0.886 +- 0.006 | 0.642 |
| weather | 0.824 +- 0.014 | 0.869 +- 0.008 | 0.838 +- 0.014 | 0.758 |

- the index ranks planted aggressive drivers above normal ones in every scenario (AUC 0.72 to 0.92), worst in the jam and the merge.
- the UAH trained agent is below the hand-set index everywhere except weather (UAH has no weather trips; its weather weights are untrained).
- single terms: proximity is the best single term in 5 of 7 scenarios (roundabout 0.909, urban 0.850); speed only on the free highway (0.805); wave 0.50 to 0.59, as expected with equal lcSigma.
- sensor noise (`model/noise.py` gaussian): highway medium at 1x 0.842 (0.841 clean), weather at 2x 0.821 (0.824 clean). the window means average the noise out.

vehicle level labels with the cut offs 35 / 70 (all 70 runs):

| truth / label | conservative | normal | aggressive |
|---|---|---|---|
| conservative | 5375 | 1138 | 0 |
| normal | 12124 | 7426 | 3 |
| aggressive | 344 | 6034 | 56 |

the ranking works, the cut offs do not: 56 of 6434 planted aggressive vehicles (0.9%) reach 70, all in the jam; most normal drivers fall below 35. the cut offs 35 / 70 were never fitted to anything. the agent's learned threshold, or a percentile threshold (part 4), is the fix; the hand-set cut offs should not be used to report shares of aggressive drivers.

## density sweep (same highway, `results/figures/planted_density_sweep.png`, `data/planted_density_headway.csv`)

median window score per true type, and mean points from the proximity term:

| proximity | type | low | medium | jam | prox points low / medium / jam |
|---|---|---|---|---|---|
| metres | conservative | 27.0 | 24.0 | 27.8 | 0.0 / 5.0 / 22.7 |
| metres | normal | 31.8 | 29.5 | 45.7 | 0.2 / 8.7 / 33.0 |
| metres | aggressive | 45.3 | 47.9 | 62.4 | 3.1 / 17.9 / 44.9 |
| headway | conservative | 27.3 | 22.7 | 13.5 | 0.4 / 1.1 / 2.3 |
| headway | normal | 34.1 | 32.8 | 23.0 | 2.5 / 8.1 / 10.6 |
| headway | aggressive | 50.8 | 58.7 | 44.5 | 10.2 / 25.0 / 28.0 |

- with the gap in metres the index climbs with density: the same normal drivers gain 14 points from low to jam, almost all from proximity (0.2 to 33.0 points), and the share of normal windows at 70 or more goes 0% / 2.1% / 5.9%. this is the NGSIM US-101 effect, now shown with drivers whose type is known.
- with time headway the proximity points of normal drivers stay at 2.5 to 10.6 and the scores do not climb (they drop in the jam because the speed term falls).

## time headway vs metres on UAH (`scripts/headway_vs_metres.py`, `data/headway_vs_metres.csv`)

leave one driver out, AUC normal vs aggressive:

| proximity | hand-set index | trained agent |
|---|---|---|
| metres | 0.829 (sd 0.114) | 0.761 (sd 0.098) |
| headway (thw_max 3 s) | 0.795 (sd 0.145) | 0.744 (sd 0.114) |

headway is slightly worse on UAH, for every held out driver (largest drop D1, 0.606 to 0.505). the metres numbers reproduce `data/uah_lodo_results.csv` exactly.

reading the two together:
- SUMO favours headway partly by construction: the planted types differ mainly in IDM tau, which is a desired time headway.
- UAH is mostly free flow on motorways and secondary roads with no jams, so it cannot show the density effect that headway removes; and UAH gaps come from a phone camera that sees a car ahead only 47% of the time.
- decision for Dr. Daher: headway removes the density climb (SUMO) at a cost of 0.03 AUC on free flow real drivers (UAH). a labelled set with dense traffic (part 3) is needed to settle it. until then the reference stays in metres and every result reports both.

## calibration against NGSIM US-101 (`data/planted_calibration.csv`, `results/figures/planted_calibration.png`)

| density | sim speed median (km/h) | NGSIM | KS D | sim headway median (s) | NGSIM | KS D |
|---|---|---|---|---|---|---|
| low | 110.8 | 48.2 | 0.99 | 5.03 | 1.49 | 0.60 |
| medium | 93.9 | 48.2 | 0.71 | 2.39 | 1.49 | 0.33 |
| jam | 13.6 | 48.2 | 0.50 | 2.06 | 1.49 | 0.31 |

all p < 0.001: the simulation is not calibrated to US-101. US-101 in the extract is dense synchronized flow at about 48 km/h; the simulation is either free (90 to 120 km/h) or a stop and go queue, never in between. simulated headways start at about 0.8 s and peak at 1.4 s, NGSIM peaks at about 1.0 s: the tau values of the plan table are too large for US-101. next step: lower tau for all three types (for example x 0.7) and add a demand level between medium and jam, then repeat the KS test.

## RL ego braking (finding 12 of part 1)

`setSpeedMode(EGO_ID, 0)` changed to 6 in `sumo_urban_agent.py`, `sumo_highway_agent.py` and `collect_data.py`: the safe speed check stays off (no phantom stops), the vType limits accel 3.0 and decel 5.0 m/s2 apply. one 200 epoch run each:

| | ego emergency braking warnings | collisions (all involve the ego) | reward, last epoch |
|---|---|---|---|
| urban, mode 0 | 5328 | 7 | 201.3 |
| urban, mode 6 | 0 | 14 | 217.8 |
| highway, mode 0 | 4781 | 149 | 274.0 |
| highway, mode 6 | 0 | 176 | 276.8 |

the 30 m/s2 braking is gone. collisions rise a little because the ego can no longer stop in one step; they exist in both modes because the ego ignores right of way and lane change safety (`setLaneChangeMode(EGO_ID, 0)`). making the ego respect right of way changes the RL task and is left open.

## NGSIM ellipse replay

`scripts/ngsim_replay_sumo.py` scores with `index_features` and `accel_1hz` (line 97). the part 1 fixes were in the SUMO collectors, not in the NGSIM path, so the replay already uses the verified features (hand check, car 983). its colours default to the context score (`--color-by context`, the index relative to the cars within 100 m, line 99); `--color-by absolute` shows the plain index, where dense traffic looks aggressive because of the proximity term. for Dr. Daher both views are worth showing side by side, with the reason (density climb above).

## status

- [x] three vTypes, mix 20 / 60 / 20, label in the vehicle id, AUC and confusion matrices
- [x] raw traci at 10 Hz, features offline through the UAH path
- [x] 10 seeds per scenario, mean and 95% CI
- [x] density sweep (index climbs with density in metres, not with headway)
- [x] time headway vs metres on UAH LODO
- [x] new scenarios: on ramp merge, stop and go jam, roundabout
- [x] weather: lower speedFactor and decel, sensor noise 2x
- [x] calibration check against NGSIM (fails, direction known)
- [x] NGSIM ellipse replay on the verified features
- [ ] tune tau to US-101 and repeat the KS test
- [ ] fit the cut offs (or drop them for percentiles in part 4)
- [ ] ego right of way
