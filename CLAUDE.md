# claude.md: vipp 301a, driver aggressiveness index

Context file for Claude Code. Read fully before doing anything in this repo.

## who and what

- Student: Hadi Al Shmaissani, third year Electrical and Computer Engineering (ECE), American University of Beirut.
- Supervisor: always write "Dr. Daher" (Dr. Naseem Daher). Never "Daher" alone, never "Prof. Daher" in new text.
- Course history: VIPP 201A (spring 2026, finished, final report submitted 10 May 2026, 43 pages). Now VIPP 301A (fall 2026), same project, meetings with Dr. Daher.
- Project: a closed form Aggressiveness Index (AI) that scores a driver from four kinematic signals (speed, acceleration, proximity to the car ahead, lateral waviness). Used to label NPC traffic in SUMO, as an RL reward, trained per environment by the DynamicWeightAgent on real drivers (UAH-DriveSet), and applied to real traffic (NGSIM US-101), with an SUMO replay that draws every car as a body ellipse plus an influence ellipse.
- Longer term (not today): paper for IEEE ITSC or T-ITS, CARLA co-simulation, learnable saturation points, multi agent Social Latency RL, on vehicle ROS2.

## working style (important)

- Hadi wants to learn, not to have it done silently. Explain each change before making it: the file, the line number, what the current code does, why it is wrong, what the new code does. Show a worked number when it helps.
- Claude Code runs commands itself (pytest, SUMO runs, analysis scripts) and reports the numbers (Hadi asked on 3 Oct 2026). Ask before git commit and git push.
- Edits to files are fine after the explanation, unless Hadi says he will type them.
- Point to file and line for "what calls what", not just a description.
- Speed matters: all four parts of the plan below are to be done today (3 Oct 2026), in order, in one session. There are no "weeks" in practice; "week 1 to 4" is just the order.
- Report results honestly, including numbers that get worse after a fix (for example the GT vs agent agreement).

## writing style for any document Claude Code produces (notebook pages, reports, readme text)

- No em dashes and no en dashes (they read as AI written). Use commas, colons, "to", or parentheses.
- No first person and no second person in documents. Neutral wording.
- Notebook style: lowercase headings, `[ ]` checkboxes, abbreviations, plain typography. No colored badges, no template fields like priority/effort/done when.
- Documents never mention version numbers or earlier drafts of themselves.
- Major is "Electrical and Computer Engineering" only.

## machine and environment

- MacBook Air (Apple silicon). Repo path:
  `~/Documents/E3/FALL 26-27/VIPP 301A/VIPP-201---SPRING-2025-2026---Dr-Naseem-Daher`
  (the path has spaces: quote it, e.g. `cd ~/Documents/E3/"FALL 26-27"/"VIPP 301A"/VIPP-201---SPRING-2025-2026---Dr-Naseem-Daher`).
- `.venv` in the repo root uses Python 3.9.6 (the README says 3.11; the Mac runs 3.9.6, so avoid 3.10+ syntax such as `match` or `X | Y` type hints). Activate with `source .venv/bin/activate`.
- SUMO installed via pip (`eclipse-sumo`); `main.py` sets `SUMO_HOME` itself in its check stage. When running a SUMO script directly, SUMO_HOME may need exporting, e.g. `export SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"`. Note that `scripts/sumo_urban_agent.py` line 31 falls back to a Windows path.
- XQuartz installed; `sumo-gui` and the ellipse replay run.
- GitHub: https://github.com/xHeedz/VIPP-201---SPRING-2025-2026---Dr-Naseem-Daher, branch `main`.
- Data lives next to the repo, not inside it, in `../28:9:2026/`:
  - `UAH-DRIVESET-v1/` (labelled trips)
  - `Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260921.csv` (about 2 GB, all locations)
  - `ngsim_us-101_5min.csv`, `ngsim_test_extract_2min.csv` (small samples)
  - `Presentation/`, `Claude outputs/`, `files/` (October presentation material and earlier outputs)
  - `main.py` line 40 points `DATA_FOLDER` there.

## repo state

- `main` at `9ac0465` "Add sensor noise on real drivers, context score, NGSIM ellipse replay and main.py" (committed and pushed 3 Oct 2026). Before it: `9ad8b3c`, `6ff1d4a`, `ca0a88c`.
- 62 tests pass (`python -m pytest -q`).
- Gotcha: never leave `.git/index.lock` behind. If git says another process is running, check for a stale `.git/index.lock` and remove it.

## repo map (what matters for today)

- `model/aggressiveness_model.py`: THE reference index (constants, `label`, `index_features`, `original_score`, `breakdown`; `AggressivenessModel` is a thin scalar wrapper). weights 0.5/0.2/0.8/0.4, squared speed and prox, wave / 1.5, labels < 35 / < 70.
- `model/sumo_features.py`: `npc_features(traci, vid)`, the one way SUMO inputs are measured.
- `model/aggressiveness_core.py`: highway-env `GroundTruthAssessor`, `AgentAssessor` (signed accel noise, lane offset wave), `_formula` wraps the reference.
- `model/dynamic_weight_agent.py`: DynamicWeightAgent, one softmax weight set and one learned threshold per environment (highway, urban, weather). `ai()` at line 105.
- `model/noise.py`: nine noise types, `noise_suite`, `sensor_view` (end of file).
- `model/context.py`: context score = normal_mean + (own score minus mean score of others within 100 m).
- `model/ellipses.py`, `model/shockwave.py`.
- `datasets/uah.py` (`load_trip`, `apply_noise`, `windows`), `datasets/ngsim.py` (`read_raw`, `trajectories`, `_accel_1hz`, `environment_for`).
- `scripts/sumo_urban_agent.py` and `scripts/sumo_highway_agent.py`: PyTorch RL ego agent plus `NPCDataCollector.update()` that writes `data/sumo_npc_aggressiveness.csv` and `data/sumo_highway_npc_aggressiveness.csv`. Bugs 1 to 3 live here.
- `scripts/sumo_runner.py`: `SumoGroundTruth` vs `SumoAgentAssessor`, three scenarios, generates its own networks under `env/sumo_scenarios/validation/`.
- `scripts/train_uah.py`, `scripts/noise_uah.py`, `scripts/score_ngsim.py`, `scripts/ngsim_replay_sumo.py`, `main.py` (stages: check, test, train, noise, sample, score, replay).
- `tests/`: `test_context.py`, `test_ellipses_and_sensors.py`, `test_real_data_pipeline.py`, `fixtures.py`.
- `docs/`: full spring history as markdown: weekly reports (`docs/reports/weekly/`), monthly reports, research notebook per week (`docs/research_notebook/`), bibliography A to I (`docs/bibliography/`), final 4 day report, speaker script. Read these for any history question instead of guessing.
- `what_was_missing/`: components the spring reports describe but the files lacked (ROS2 nodes, SumoEnv, social latency), recreated in September 2026. Not original work, produced no reported result.
- `legacy/`, `experiments/`: older phase code, synthetic weight agent experiments.

## history in short

- Spring 2026 (VIPP 201A): index designed, Q-learner and PyTorch policies in highway-env, SUMO port, shockwave analysis, Social Latency reward, ROS2 wrapper, final report. On 2 May a 100% GT vs agent accuracy turned out to be an info leak (both paths read the same TraCI values); split into SumoGroundTruth and SumoAgentAssessor with Gaussian noise. Final agreement: highway 91.0% (1328 samples), urban 81.6% (2284), weather 88.3% (1825). Calibration then: highway speed_ref 38 m/s, gap_ref 25 m, thresh 55%, noise_scale 1.5; urban 22 m/s, 17 m, 50%, 1.0; weather 25 m/s, 35 m, 55%, 1.0.
- Summer and September 2026: real data step. DynamicWeightAgent trained on UAH-DriveSet with leave one driver out (LODO). Nine noise types. Shockwave factor for aggressive and conservative drivers. NGSIM US-101 scoring and SUMO replay with ellipses (Dr. Daher asked for vehicles as ellipsoids, not particles).
- 25 Sep finding: the UAH-trained agent put about 99% of US-101 cars in aggressive. Causes: NGSIM accel from 10 Hz video positions (fixed with `accel_1hz`, measured the UAH way) and stop and go traffic (context score). On 2 min of US-101: aggressive share 0.99 raw accel, 0.68 with accel_1hz, 0.35 with context. Agent margin is narrow: UAH normal mean 11.0 against highway threshold 12.5.

## october presentation feedback from Dr. Daher (early Oct 2026)

Figures good, model good. Next: verify and hard check everything, hand check every number, run better simulations, train on more and better datasets, explain the very high aggressiveness index values, and start RAG so the model has context (traffic type, road type and more), not just the driving state.

Rule from this: nothing new gets presented until every number in the pipeline has been checked by hand at least once and the high scores are explained. Measurement first, context last: a RAG layer on an unverified index only hides bugs.

Open question for Dr. Daher: redo the old presentation figures with the fixed features, or only new ones going forward?

Plan doc (Claude Doc, web): https://claude.ai/code/artifact/9e03b474-0ddc-404a-9dce-c350cf97107d. Its full content is reproduced below.

## bugs found by reading the code (commit 9ad8b3c, still present)

1. Urban wave = std of global y over 30 samples. `scripts/sumo_urban_agent.py` lines 93 and 96 (`h['ys'].append(vy)`, `np.std(ys)`); highway lines 90 and 93. A car driving north to south gets its travel distance as wave. Worked number: 13.9 m/s, samples every 0.3 s, 30 samples span about 8.7 s, about 121 m of travel, std about 35 m, / 1.5 clips to 1, 0.4 x 1 x 100 = 40 points from travel alone. 99.4% of urban rows in `data/sumo_npc_aggressiveness.csv` have wave above 1.5 m; urban median score 46.7 at about 50 km/h where the speed term is only about 5.6.
   Fix: `wave_m = abs(traci.vehicle.getLateralLanePosition(vid))` (offset from lane centre, same quantity as UAH and NGSIM wave). Consequence: with the default LC2013 model (no sublane model) cars sit on the lane centre, so wave reads about 0 except during lane changes. Real lane wobble comes later via the sublane model (`--lateral-resolution`) for the planted driver types.
2. Accel divides by 0.1 s: urban line 91, highway line 88. `update()` only runs every `POLICY_FREQ` steps (urban line 45: 3, highway line 45: 4), so the real dt is 0.3 s and 0.4 s and accel is 3x and 4x too big. Worked number: 10.0 to 10.6 m/s over 0.3 s is 2.0 m/s², code gives 6.0, clipped to 5, accel term maxed.
   Fix: `accel_ms2 = traci.vehicle.getAcceleration(vid)` (SUMO's accel over the last 0.1 s step, independent of how often update runs). Alternative: divide by `0.1 * POLICY_FREQ`.
3. Proximity = straight line distance to the ego car: urban line 81, highway line 78 (`np.sqrt((vx - ego_x) ** 2 + ...)`). UAH and NGSIM use the gap to the car ahead in the same lane. Same name, different quantity, so weights do not transfer.
   Fix: `leader = traci.vehicle.getLeader(vid, 50.0)`. It returns `(leader_id, dist)` or None (older SUMO: `('', -1)`), so test `if leader and leader[0]`. `dist` is measured from the follower's front bumper PLUS minGap to the leader's rear bumper, so the true gap is `dist + traci.vehicle.getMinGap(vid)` (default minGap 2.5 m). No leader: `prox_m = 0.0`, which the normaliser already treats as "no proximity term" (urban line 128: `if 0 < prox_m <= 50 else 0.0`), the same convention as UAH.
   After the fix `ego_x, ego_y` are no longer needed by `update()`; keep the signature or update the call sites (highway around line 202, urban similar).
4. Proximity is in metres, not time, so dense traffic reads as tailgating (the US-101 problem): a 10 m gap at 20 km/h gives (1 - 10/50)² = 0.64. Try time headway (gap / speed) or TTC.
5. UAH sets gap = 0 when no car is detected ahead; NGSIM almost always has a leader. The prox weight is learned on mostly zeros, then fires on NGSIM. Report leader_coverage, train with a "leader present" flag.
6. UAH labels are per trip: every window of an aggressive trip is labelled aggressive, even calm cruising. Squashes the scale (normal mean 11.0 vs threshold 12.5). Relabel with UAH event files or keep only the top windows per trip.
7. 47% of rows in `data/driving_behaviors_dataset.csv` have |accel| above 9 m/s² (min -35.5). Drop or recompute; trace where it was generated.
8. Three definitions of the index: `aggressiveness_core._formula` (linear prox, wave / 2.0, weight 0.3, cut offs 30/65), `aggressiveness_model.get_ai_score` (squared prox, wave / 1.5, weight 0.4, cut offs 35/70), and the UAH trained agent (learned weights and threshold).
9. `AgentAssessor` adds `abs(np.random.normal(0, 0.6))` to accel: mean of a half normal is 0.6 x sqrt(2/pi) = about 0.48 m/s², so the noisy agent always sees more accel than the truth. `GroundTruthAssessor` uses offset from lane centre for wave, `AgentAssessor` uses std of y. Make them the same.
- Highway SUMO run: 2.3% of rows sit exactly at the cap of 100 and 11.7% are above 70.

## the plan (from the plan doc), do in this order today

### part 1: fixes 1 to 3 and hand checks
- [x] fix bugs 1, 2, 3 in both SUMO agent scripts (explain first, then edit)
- [x] rerun both SUMO scripts (`--no-gui`), redo the score histograms, before/after histogram of the urban index (expect the urban median to drop by roughly 40 points)
- [x] one formula, not three: pick one definition, delete or wrap the others, write it once in the notebook as the reference; every script imports it from one place (grep for `/ 150.0`, `/ 50.0`, `/ 1.5`, `/ 2.0` to find copies)
- [x] hand check on 3 real windows (one UAH normal, one UAH aggressive, one NGSIM US-101 car), per feature:
  - speed: raw value, km/h conversion, / 150, squared
  - accel: the two speeds used, the real dt between them, the derivative, clip, / 5
  - proximity: who the gap is measured to, units, / 50, squared
  - wave: what the lateral reference is, units, / 1.5
  - weighted sum, x 100, threshold, label; must match the code to 2 decimals
- [x] `tests/test_hand_checks.py` with the hand numbers hard coded
- [x] sanity checks on every CSV the pipeline writes: no |accel| above about 8 m/s²; no wave above about 2 m outside a lane change; share of rows exactly at a clip limit (score 100, accel 5, wave 1.5) under 5%; histogram of each feature per dataset side by side saved to `results/figures/feature_check_*.png`
- [x] assessor: signed noise before the abs in AgentAssessor, measure the bias again; same wave definition in GT and agent; rerun GT vs agent agreement and report the new number honestly
- [x] print leader_coverage and lane_coverage per UAH trip (train_uah.py already computes it)
- [x] per feature contribution plot for the top 5% scores (which term pushes them up)
- [x] then decide whether the 25 Sep context score is still needed (it shifts everything toward the UAH normal mean and can hide a broken feature)

### part 2: better simulations
Planted driver types, one vType each in the .rou.xml, carFollowModel IDM (starting guess, tune so speed and headway spread look like NGSIM):

| type | tau (s) | accel (m/s²) | decel (m/s²) | speedFactor | lcAssertive | sigma |
|---|---|---|---|---|---|---|
| conservative | 1.8 | 1.5 | 3.0 | 0.9 | 0.5 | 0.2 |
| normal | 1.2 | 2.6 | 4.5 | 1.0 | 1.0 | 0.5 |
| aggressive | 0.6 | 3.5 | 6.0 | 1.2 | 3.0 | 0.7 |

- [x] three vTypes, mix 20 / 60 / 20, label carried in the vehicle id; every NPC now has a true label, report AUC and confusion matrices
- [x] log raw traci values at 10 Hz to CSV, compute features offline with the same `index_features` used for UAH and NGSIM (one code path for every source)
- [x] at least 10 seeds per scenario (`--seed`), mean and 95% CI
- [x] density sweep: low, medium, jammed flow on the same network; the index should not climb just because traffic is dense
- [x] time headway vs metres: compare AUC on UAH leave one driver out
- [x] new scenarios: on ramp merge, stop and go jam, roundabout
- [x] weather: lower speedFactor and decel for all types plus larger sensor noise
- [x] calibration check: speed and time headway histograms of sim vs NGSIM US-101, KS test
- [x] keep the NGSIM ellipse replay as the visual for Dr. Daher, driven by the fixed features

### part 3: more and better datasets
Labelled: UAH-DriveSet (in use; check its per event files for window level labels), Kaggle driving behavior (smartphone IMU, no speed or gap), smartphone sensor dataset PMC 2022, multi class driver behaviour dataset PMC 2025 (check labels, sensors, license, access).
Unlabelled trajectories: NGSIM US-101 and I-80 (in use, jam case), NGSIM Lankershim and Peachtree (arterial, `datasets/ngsim.py` already supports), highD (German highway), exiD (entries and exits), rounD (roundabouts), INTERACTION (merges, roundabouts, intersections, several countries), pNEUMA (dense urban Athens, closest to Beirut), MiTra (freeway with ramps, all traffic states).
- [ ] apply for highD and exiD access (same form covers rounD); approval takes days
- [x] download pNEUMA and NGSIM arterial first (open)
- [x] one loader per dataset in `datasets/`, all returning the same per second table as `uah.py` (t, speed_kmh, accel, gap_m, wave_m)
- [x] leave one driver out for UAH; any new labelled set split by driver, never by window
- [x] for unlabelled sets: score histogram per road type and per density bin
- [x] write down the label definition of each dataset (trip labels vs event labels)

### part 4: RAG (context for the model)
Before scoring a window, retrieve what is known about its situation and score relative to that. The same 1.0 s headway is aggressive at 120 km/h on an empty motorway and ordinary in a Beirut jam.

Context fields (retrieval key): road type (motorway, arterial, residential, roundabout, ramp: SUMO edge type / OSM highway tag), speed limit (`traci.lane.getMaxSpeed` / OSM maxspeed), traffic state (density, mean speed, free flow vs jam), manoeuvre (cruise, lane change, merge, turn), weather and light, region (driving culture: dataset country), vehicle class (vType / NGSIM v_Class).

Two things get retrieved:
1. similar labelled windows: FAISS index over UAH (and later labelled) windows keyed by context + features. Score = percentile among the k nearest normal windows with matching road type and traffic state; threshold from the neighbours instead of one number per environment.
2. text knowledge: short markdown files with speed limits, following distance rules, regional norms (Lebanon, Germany, US). Used only for the explanation ("flagged: 0.6 s headway at 110 km/h, local rule 2 s"), never inside the score.
An LLM is optional and only for the explanation; it cannot sit in the 10 Hz ROS2 loop.

- [x] `Context` dataclass with the fields above, filled in the SUMO scripts and every dataset loader
- [x] window index (features + one hot context) with FAISS, UAH only first
- [x] context aware score = percentile among k nearest normal windows with matching road type and traffic state
- [x] ablation table: no context / 3 environments / RAG context, AUC on UAH LODO and on the planted SUMO drivers
- [x] knowledge folder (markdown) and an explanation function that cites which file it used
- [x] 3 to 5 papers on context aware driving behaviour and retrieval augmented trajectory models for annotated bibliography J (`docs/bibliography/`)

Diagram: window + context -> retriever -> similar labelled windows -> scorer (percentile) -> score. Rules folder -> explanation only.

## for the next meeting with Dr. Daher
- [x] before / after histogram of the urban index (99.4% wave saturation gone)
- [x] hand check page for one window, paper number next to code number (`docs/hand_checks/hand_check_3_windows.md`)
- [x] AUC on the planted SUMO drivers, mean and CI over seeds
- [x] RAG design diagram with the first ablation number if ready

## where things stood (3 Oct 2026, end of part 1)
- part 1 done, NOT committed yet. Full write-up with every number: `docs/research_notebook/fall_2026_10_03_part1_verification.md`.
- One reference index: `model/aggressiveness_model.py` (constants, `index_features`, `original_score`, `label`, `breakdown`). SUMO inputs: `model/sumo_features.py` `npc_features`. `sumo_runner._agg_formula` and the per scenario speed_ref/gap_ref calibration are gone (Hadi chose the reference on 3 Oct).
- Bugs 1 to 3 were also in `scripts/collect_data.py` (fixed). Before-fix CSVs kept as `data/*_before_fix.csv`.
- GT vs agent, reference index, 5 seeds: highway 93.8% (majority baseline 62.3%), urban 83.1% (68.8%), weather 94.3% (92.8%). GT almost never says aggressive (0.4 to 5.8%), so agreement means little; planted driver types (part 2) replace it.
- High scores on real data come from proximity in metres: 58 of 90 points in the NGSIM top 5%, 47 of 82 in UAH. Bug 4 (time headway) is the next measurement fix.
- New findings 10 to 14 in the notebook page: UAH lane estimator state ignored (`datasets/uah.py` line 74), leader coverage confounded with the UAH label, validation slide examples were bug 1 artifacts, RL ego brakes at 30 m/s2 (`execute_action` + `setSpeedMode(EGO_ID, 0)`), getLeader beyond 50 m.
- Next: part 3. Commit everything together at the end (Hadi, 4 Oct).

## where things stood (4 Oct 2026, end of part 2)
- write-up: `docs/research_notebook/fall_2026_10_04_part2_planted_drivers.md`. Scripts: `scripts/planted_drivers.py` (70 runs, 7 settings x 10 seeds, logs in `data/sumo_planted/`, git-ignored), `scripts/planted_eval.py`, `scripts/headway_vs_metres.py`, `datasets/sumo_log.py`, `model.aggressiveness_model.headway_features` (candidate, not reference).
- AUC aggressive vs normal (reference): 0.72 (jam) to 0.92 (roundabout); headway variant better in 6 of 7 SUMO settings but worse on UAH LODO (0.795 vs 0.829). Index in metres climbs with density (normal drivers +14 points low to jam), headway does not.
- Cut-offs 35 / 70 label only 0.9% of planted aggressive vehicles aggressive: uncalibrated.
- Sim not calibrated to NGSIM US-101 (KS D 0.3 to 0.99); tau too large.
- Ego speed mode 0 -> 6: ego emergency braking 5328 -> 0 (urban), 4781 -> 0 (highway); collisions 7 -> 14, 149 -> 176.

## where things stood (4 Oct 2026, end of part 3)
- write-up: `docs/research_notebook/fall_2026_10_04_part3_datasets.md`; dataset reference `docs/datasets.md`.
- new data in `../28:9:2026/`: `ngsim_i-80.csv`, `ngsim_lankershim.csv`, `ngsim_peachtree.csv`, `pneuma/20181024_d1_0830_0900.csv` (pNEUMA terms accepted by Hadi 4 Oct, CC BY-NC 4.0).
- new code: `datasets/pneuma.py`, `datasets/ngsim.py` (`planar`, `per_second`, `_split_periods`), `scripts/score_unlabelled.py`, `scripts/uah_relabel.py`.
- finding 15: combined NGSIM file reuses Vehicle_ID per recording period; fixed by `_split_periods`.
- In metres, congested NGSIM traffic outscores UAH aggressive drivers (I-80 median 57.8 vs 54.5); with headway every unlabelled site sits near/below UAH normal. Index in metres climbs with density on real traffic too.
- highD/exiD application is Hadi's to do.

## where things stood (4 Oct 2026, end of part 4, all four parts done)
- write-up: `docs/research_notebook/fall_2026_10_04_part4_rag.md`. Code: `model/rag.py`, `datasets/contexts.py`, `knowledge/`, `scripts/rag_ablation.py`, bibliography J.
- RAG helps only where situations are mixed (pooled SUMO 0.751 -> 0.778, headway 0.830 -> 0.849), loses 0.028 on UAH. Headway adds more than context. Uncalibrated SUMO is a bad reference set for real traffic.
- Open items for next session: tune SUMO tau to NGSIM, fit or replace the 35/70 cut-offs, decide metres vs headway (Dr. Daher), labelled dense-traffic reference set, highD/exiD application (Hadi), hand check one arterial and one pNEUMA window, UAH lane estimator state (finding 10), ego right of way.
