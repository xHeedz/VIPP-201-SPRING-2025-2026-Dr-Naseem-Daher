# sat 3 oct 2026: part 1, verification of the index

goal from the october feedback (Dr. Daher): hand check every number, explain the high scores, before anything new is presented.

## the reference index (one definition, `model/aggressiveness_model.py`)

AI = min(100 (0.5 n_s^2 + 0.2 n_a + 0.8 n_p^2 + 0.4 n_w), 100)

- n_s = min(v / 150 km/h, 1)
- n_a = min(|a| / 5 m/s2, 1)
- n_p = 1 - gap / 50 m if 0 < gap <= 50 m, else 0 (gap = bumper to bumper gap to the car ahead in the same lane, 0 = no car ahead)
- n_w = min(|offset from the lane centre| / 1.5 m, 1)
- labels: < 29 conservative, < 42 normal, else aggressive (fitted on UAH and calibrated SUMO, `fall_2026_10_04_open_items.md`; the numbers on this page were computed with the earlier cut offs 35 / 70)

constants `SPEED_MAX_KMH`, `ACCEL_MAX_MS2`, `PROX_MAX_M`, `WAVE_MAX_M`, `WEIGHTS`, `THRESHOLDS`; functions `index_features`, `original_score`, `label`, `breakdown`. every script imports these. SUMO inputs come from one function, `model/sumo_features.py` `npc_features`.

- [x] four definitions found, not three:
  - `aggressiveness_model.get_ai_score`: kept as the reference
  - `aggressiveness_core._formula` (linear prox, wave / 2, weight 0.3, / 1.8, cut offs 30/65): now a wrapper of the reference
  - `sumo_runner._agg_formula` (linear v / speed_ref, |a| / 4, gap / gap_ref, wave / 2, weights 0.8/0.05/0.6/0.05, per scenario refs and thresholds, the May calibration behind the 91.0 / 81.6 / 88.3% agreement): removed, both assessors use the reference
  - the UAH trained agent (learned weights and threshold): unchanged, built on `index_features`
- [x] printed traces (`sumo_urban_agent`, `sumo_highway_agent`, `collect_data`, `build_validation_ppt`) and the 30/65 label in `env/scenarios/urban_scenario.py` now use `breakdown` / `label`
- legacy/ keeps its own copy (not used by any current script)

## fixes 1 to 3 (SUMO NPC collectors)

same three bugs in three files: `sumo_urban_agent.py`, `sumo_highway_agent.py` and `collect_data.py` (the third copy was found while removing duplicate formulas).

| | before | after |
|---|---|---|
| wave | std of global y over 30 samples (travel distance) | abs(getLateralLanePosition) |
| accel | dv / 0.1 s, real dt 0.3 s (urban), 0.4 s (highway) | getAcceleration |
| prox | straight line distance to the ego car | getLeader distance + minGap, no leader = 0 |

| | urban before | urban after | highway before | highway after |
|---|---|---|---|---|
| median score | 46.7 | 6.5 | 44.4 | 25.9 |
| wave > 1.5 m | 99.4% | 0.0% | 22.0% | 0.0% |
| abs(accel) >= 5 | 7.9% | 0.1% | 15.2% | 0.2% |
| score = 100 | 1.34% | 0.00% | 2.34% | 0.03% |
| score >= 70 | 5.2% | 0.0% | 11.7% | 0.2% |

urban median drops by 40.2 points (expected about 40). figures: `results/figures/urban_index_before_after.png`, `highway_index_before_after.png`.

consequences:
- wave is exactly 0 on every SUMO row: LC2013 changes lanes in one step, so no car ever sits off the centre. the index runs on three terms in SUMO until the sublane model (part 2).
- urban is 99.6% conservative: default SUMO cars are calm; the 40 points came from the wave bug. planted aggressive types (part 2) are needed to test the index there.
- remaining highway scores >= 70: hard braking (median -4.9 m/s2) at about 8.6 m gaps, about 85 km/h. real events.
- row counts differ (urban 2,903 before, 96,005 after): the old CSVs came from a shorter run; compared as distributions, not row by row.

## hand checks (3 real windows)

page: `docs/hand_checks/hand_check_3_windows.md`, script `scripts/hand_check.py` (plain python on the raw files, no pipeline code), tests `tests/test_hand_checks.py`.

| window | hand | pipeline | label | main term |
|---|---|---|---|---|
| UAH D1 normal motorway, t0 306.88 s | 42.32 | 42.32 | normal | speed 25.3 (107 km/h) |
| UAH D1 aggressive motorway, t0 688.94 s | 86.87 | 86.87 | aggressive | prox 45.9 (12 m at 125 km/h) |
| NGSIM US-101 car 983, t 259 s | 52.50 | 52.50 | normal | prox 37.4 (15.8 m at 34 km/h) |

all three agree to about 1e-14. the code computes what the formula says; what remains is whether the formula measures the right thing:
- UAH aggressive: 12 m at 125 km/h is a 0.34 s time headway, real tailgating, the high score is earned
- NGSIM car 983: 15.8 m at 34 km/h is a 1.7 s time headway, ordinary for dense traffic, and still gets 37 points from proximity (bug 4)

## sanity checks on every source (`scripts/sanity_checks.py`, `data/sanity_checks.csv`)

| source | rows | abs(a) > 8 | wave > 2 m | score 100 | abs(a) >= 5 | wave >= 1.5 | leader share | median | p95 |
|---|---|---|---|---|---|---|---|---|---|
| sumo urban | 96,005 | 0.06% | 0% | 0% | 0.09% | 0% | 0.49 | 6.5 | 12.4 |
| sumo highway | 348,448 | 0.09% | 0% | 0.03% | 0.19% | 0% | 0.71 | 25.9 | 34.1 |
| uah (per second) | 30,960 | 0.03% | 0% | 0.14% | 0.09% | 0.08% | 0.47 | 33.9 | 75.5 |
| ngsim us-101 (10 Hz) | 333,065 | 0% | 0.06% | 0.77% | 0% | 1.45% | 0.94 | 47.9 | 81.4 |
| legacy highway-env | 72 | 50.0% | 2.8% | 0% | 72.2% | 5.6% | 1.00 | 32.0 | 64.1 |

- [x] every current source passes (abs(a) > 8 under 0.1%, every clip share under 5%); NGSIM wave > 2 m counted outside +-3 s of a lane change
- [x] feature histograms side by side: `results/figures/feature_check_{speed_kmh,accel,gap_m,wave_m,score}.png`
- legacy file (bug 7) traced to `legacy/data_collector.py`: accel divided by dt = 1/15 s (simulation frequency) while each `env.step` advances one policy step (highway-env default 1 Hz), so accel is 15x too large (min -70.96 m/s2 / 15 = -4.7 m/s2); wave is `v.velocity[1]`, a lateral speed in m/s, not an offset. only legacy code reads it; not used anywhere.

## why the high scores are high (top 5% per source, `results/figures/top5_contributions.png`)

mean points per term:

| source | rows | score | speed | accel | prox | wave |
|---|---|---|---|---|---|---|
| uah | all | 38.8 | 20.2 | 0.8 | 10.0 | 7.8 |
| uah | top 5% | 82.4 | 23.4 | 1.2 | 46.6 | 11.5 |
| ngsim us-101 | all | 47.9 | 5.4 | 2.0 | 29.6 | 11.1 |
| ngsim us-101 | top 5% | 89.8 | 4.3 | 2.7 | 58.1 | 26.2 |
| sumo highway | top 5% | 40.8 | 20.6 | 7.9 | 12.2 | 0 |

proximity carries the high scores on real data (57% of the points in the UAH top 5%, 65% on NGSIM). on NGSIM the speed term is only 4 points: the high scores are slow, close, dense traffic, which is the metres vs time problem (bug 4), not aggressive driving. wave is the second term on NGSIM (26 points): lane changes plus a lane centre taken as the median of all cars in the lane.

## assessor (GT vs agent), `scripts/sumo_runner.py`

changes: reference formula in both; agent accel = slope + signed noise (was abs(slope) + abs(noise)); agent wave = |noisy lane offset| averaged over 5 readings, the same quantity as GT (was std of noisy global y); both gaps add minGap back, no leader = 0 (was 150 m); lateral noise 0.2 m (the `model/noise.py` value, was the 0.8 m range noise); bias per feature now logged.

reproduction of the old code at HEAD: 92.8 / 82.8 / 89.0%, not the reported 91.0 / 81.6 / 88.3%. the numpy noise was never seeded, so every run differs.

5 seeds each (cut offs 35 / 70; with the fitted 29 / 42 see `fall_2026_10_04_open_items.md`):

| scenario | n | agreement | majority baseline | GT conservative / normal / aggressive | bias abs(a) (m/s2) old rule | new rule | bias wave (m) |
|---|---|---|---|---|---|---|---|
| highway | 1328 | 93.8% (sd 0.2) | 62.3% | 62.3 / 36.4 / 1.2% | +0.433 | +0.297 | +0.091 |
| urban | 2284 | 83.1% (sd 0.3) | 68.8% | 68.8 / 25.4 / 5.8% | +0.290 | +0.184 | +0.073 |
| weather | 1825 | 94.3% (sd 0.1) | 92.8% | 92.8 / 6.8 / 0.4% | +0.291 | +0.206 | +0.062 |

- signed noise removes about a third of the accel bias. the rest is not a sign error: the magnitude of a noisy value is larger on average than the magnitude of the true value when the truth is near 0. speed and gap bias are below 0.02.
- the agreement numbers look higher but mean less: under the reference index the GT almost never says aggressive (0.4 to 5.8%), even though the scenarios plant aggressive drivers. weather 94.3% is only 1.5 points above always answering conservative. this is the same problem the May speed_ref change worked around (the squared speed term cannot reach aggressive at realistic speeds); planted driver types with true labels (part 2) replace GT vs agent agreement as the test.
- highway-env assessors (`model/aggressiveness_core.py`): same accel and wave fixes applied, not rerun (`highway_env` is not installed in the .venv).

## uah coverage per trip (`data/uah_trip_coverage.csv`)

| road, behaviour | lane coverage (loader) | lane rows in state 2 | leader coverage |
|---|---|---|---|
| motorway aggressive | 0.994 | 0.788 | 0.639 |
| motorway drowsy | 0.996 | 0.832 | 0.483 |
| motorway normal | 0.991 | 0.791 | 0.538 |
| secondary aggressive | 0.998 | 0.958 | 0.660 |
| secondary drowsy | 1.000 | 0.927 | 0.217 |
| secondary normal | 0.997 | 0.949 | 0.341 |

- leader coverage is confounded with the label: on secondary roads aggressive trips have a car ahead 66% of the time, normal trips 34%. with proximity the dominant term, part of what separates aggressive from normal in UAH may be "a car was detected ahead" (aggressive drivers close up, so the phone camera sees the car). 10 of 40 trips have leader coverage under 0.2.
- `train_uah.py` now prints and saves coverage per trip.

## new findings

10. `datasets/uah.py` line 74 ignores column 4 of PROC_LANE_DETECTION ("state of lane estimator" in the official reader). 12% of the rows it uses are not in state 2 (state -1: 8,560 rows, 0: 22,635, 1: 58,648); states -1 and 0 read about 0 m offset, so wave is slightly understated and lane coverage overstated (99% reported, about 80% state 2 on motorways). the meaning of each state (2 = lane detected) is inferred from the y axis range in the reader, not documented.
11. the validation slides (`scripts/build_validation_ppt.py`) show a "normal" example with wave 4.2 m and an "aggressive" one with wave 33.7 m: both are bug 1 (travel distance), not driver behaviour.
12. the RL ego in both SUMO agent scripts brakes at 30 m/s2 (about 3 g): the brake action drops the target speed by 3 m/s in one 0.1 s step (`execute_action`, urban line 340) with `setSpeedMode(EGO_ID, 0)` (line 402) disabling every limit. not part of the NPC CSVs, but the RL policy is trained on a car that can stop instantly, and NPCs behind it brake hard because of it. fix in part 2 (`slowDown` over the policy step, or speed mode 6).
13. `getLeader(vid, 50)` returns leaders beyond 50 m (57% of highway rows, up to 114 m); harmless, the normaliser treats > 50 m as no proximity.
14. numpy 2.0 on macOS warns "divide by zero in matmul" on clean input; `original_score` now uses an explicit weighted sum.

## context score (25 Sep), still needed?

the context score subtracts the mean score of the cars within 100 m. today's numbers name the effect it was compensating for: proximity in metres reads dense traffic as tailgating (58 of 90 points in the NGSIM top 5%). the measurement fix is time headway or TTC (part 2), tested on UAH LODO AUC. until then, report the absolute and the context score side by side and do not use the context score as the headline number. after time headway is in, rerun NGSIM and drop the context score if the aggressive share no longer depends on density. traffic state as context belongs in part 4 (RAG), where it is explicit instead of a subtraction.

## status

- [x] fixes 1 to 3, reruns, before/after histograms
- [x] one formula
- [x] hand checks on 3 windows, `tests/test_hand_checks.py` (51 tests pass)
- [x] sanity checks and feature histograms
- [x] assessor bias, same wave definition, agreement rerun
- [x] leader and lane coverage per trip
- [x] top 5% contribution plot
- [x] context score: keep side by side until time headway is tested (decision for Dr. Daher)
- [ ] finding 10: filter lane rows to state 2, retrain UAH, compare AUC
- [ ] finding 12: ego braking
