# next session: where to resume

read `CLAUDE.md` first (working style, writing style, repo map, history), then this list. everything below is open; everything else from the October plan is done and written up in `docs/research_notebook/fall_2026_10_0*.md`.

## state at the end of 4 oct 2026

- repo `main` on GitHub (`github.com/xHeedz/VIPP-201-SPRING-2025-2026-Dr-Naseem-Daher`), 68 tests pass (`python -m pytest -q`).
- reference index: `model/aggressiveness_model.py`, gap in metres, cut offs 29 / 42 (fitted). time headway variant: `headway_features`.
- October deck: `../28:9:2026/Presentation/VIPP 301A Session Results October 2026.pptx` (18 slides), rebuilt by `python scripts/build_october_deck.py` (reads `data/`, so rerun after any new result).
- full deck to present: `../28:9:2026/Presentation/VIPP 301A Project Recap and October 2026 Update.pptx` (45 slides: project recap 1 to 25, September session 26 to 29, October update 30 to 43, glossary, thank you), rebuilt by `python scripts/merge_recap_deck.py` after the October deck. the recap's own 9 charts do not render in Keynote (also in the original recap file); check them in PowerPoint.
- data next to the repo in `../28:9:2026/`: UAH, NGSIM (US-101 5 min, I-80, Lankershim, Peachtree), pNEUMA (10 drones, 24 Oct 2018 08:30), DriveDNA-Sample (signal CSVs only).
- Hugging Face: logged in as xHeedz (`hf auth login`); DriveDNA-Sample granted; full DriveDNA not yet.

## 0. RL on the index side (5 oct 2026, `docs/rl_pivot.md`)

- decision: option a, sequential labeling (Hadi). `agent/driver_q_learner.py` renamed to `agent/score_bucket_labeler.py` (label counting, not RL).
- built: `env/labeling_env.py` (gymnasium), `tests/test_labeling_env.py`, `scripts/train_labeling_rl.py` (PPO, UAH leave one driver out). write-up: `docs/research_notebook/fall_2026_10_05_rl_labeling.md`.
- result: PPO 0.78 balanced acc in 5 to 9 s, top reward at wait cost 0.002 and 0.005 by a margin inside the spread across drivers; less accurate than the DynamicWeightAgent at 10 s (0.80) and 30 s (0.83); misses about a third of aggressive episodes. not a win yet.
- 5 oct follow up: missed aggressive cost 2 + weight agent score in the observation: 0.792, 75% of aggressive caught, 12 s (still below the weight agent).
- 7 oct, planted SUMO drivers (three classes, split by seed, `scripts/train_labeling_rl_planted.py`): PPO 0.666 to 0.676 balanced acc in 7 to 12 s, equal to the per environment index at 30 s (0.667), 0.015 below it at 60 s; catches 86 to 90% of aggressive drivers. gain is speed. write-up `docs/research_notebook/fall_2026_10_07_rl_planted_highd.md`.
- 7 oct, later: gamma 1 + wait bias 4 fixed the early commit; headway features: PPO 0.736 (3 seeds) after about 30 s vs per environment index 0.677 at 30 s, 0.741 at 60 s. hand traced decisions in `docs/hand_checks/rl_decision_trace_*.md`.
- proximity: Dr. Daher wants metres and headway mixed per environment. fitted shares of metres: highway 0.5, urban 0.5, weather 0.2 (`PROX_MIX`, proposal, reference still metres). decision to ask: adopt as reference (then refit cut offs and rerun everything, old 2.5 list).
- US-101: EIDM with a lane drop to 3 lanes at 5,750 veh/h: median 42 km/h, speed KS 0.26, headway KS 0.063 (was 0.43 / 0.14). target 0.2.
- deck for the meeting: `../28:9:2026/Presentation/VIPP 301A Update 7 October 2026.pptx` (`scripts/build_week_deck.py`). paper outline: `docs/paper_outline.md`.
- skipped: 100-DrivingStyle (Hadi, 7 oct).
- 9 oct: highD + exiD in `../12-10-2026/` (also inD, rounD, uniD zips, not unpacked). scored (`scripts/score_german.py`): about half of German motorway windows >= 42 in every proximity variant (speed + ordinary headway + lane offset add up). German reference percentile (`scripts/context_reference.py`): false alarms 1 to 3% but misses most aggressive drivers, because the bin used own speed; next: bin on traffic state. IRL (`model/irl.py`, `scripts/irl_planted.py`): 0.707 AUC on planted drivers vs 0.948 for the index; not run on highD. write-up `docs/research_notebook/fall_2026_10_09_german_data.md`.
- the DriveDNA folder in `../12-10-2026/` is the sample again (plus videos); full release left until Dr. Daher gives access (Hadi, 9 oct).
- 9 oct, later: inD, rounD, uniD unpacked and scored (`datasets/levelx_urban.py`); traffic state context (highway: AUC kept, false alarms about 1%; urban fails on SUMO realism); IRL with desired headway: population fits order the planted types (1.63 / 2.43 / 2.94 s), per vehicle AUC 0.728, German motorway fits not identified. write-up `docs/research_notebook/fall_2026_10_09b_all_german_data.md`.
- disk: about 13 GB free (94%); the five levelX zips in `../12-10-2026/` (3.8 GB) can go once Hadi agrees.
- deck rule (Hadi, 9 oct): never create a presentation without asking; add to the existing deck.
- highD: loader `datasets/highd.py` ready and tested on a fixture; download in progress (Hadi). put the files in `../28:9:2026/highD/` (`XX_tracks.csv`, `XX_tracksMeta.csv`, `XX_recordingMeta.csv`), then run it, check gap against dhw, hand check one window, score in metres and headway.

## 1. waiting on Hadi (check these first)

- [ ] full DriveDNA (465 drivers): access at huggingface.co/datasets/HenryYHW/DriveDNA (not the -Sample page). test: `.venv/bin/hf download HenryYHW/DriveDNA index/drives.parquet --repo-type dataset --local-dir ../28:9:2026/drivedna`
- [ ] 100-DrivingStyle: reply from the authors to the email in `../drafts/email_100_drivingstyle.md`
- [ ] Dr. Daher's decisions: (a) gap in metres or time headway for proximity, (b) redo the September figures with the fixed features or only new ones, (c) report labels with 29 / 42 only after (a)

## 2. technical work, in this order

### 2.1 full DriveDNA (as soon as access is granted)
- download only `index/`, `splits/`, `data/windows.parquet`, `data/maneuver_events.parquet` and `raw_signals/` (7.3 GB); never `videos/`, `embeddings/`, `features/`
- extend `datasets/drivedna.py` to the parquet layout (`raw_signals/driver_XXX/drive_YYY.parquet`, human spans from `index/human_segments.parquet`)
- rerun `scripts/score_drivedna.py` on all drivers: slow car following vs free driving (metres vs headway), driver consistency across cars, density bins
- use the rule labels (close following, hard braking primitives in `windows.parquet`) only as a check, not as aggressiveness labels

### 2.2 100-DrivingStyle (if the authors share it)
- loader returning the per second table (speed, accel; no gap, no lane: two term score)
- leave one driver out AUC of the speed and accel terms against the expert aggressiveness ratings (5 levels, 100 drivers)

### 2.3 US-101 synchronized flow in SUMO (open item 5)
- problem: IDM breaks down into stop and go; US-101 has steady 40 to 60 km/h flow. best so far: `us101` scenario at 7,000 + 1,200 veh/h, median 38 km/h, speed KS 0.43 (`data/us101_sweep.csv`)
- try: carFollowModel `Kerner` or `EIDM` for the planted types (keep the tau ratios), and a lower speed limit on the weaving section; rerun `scripts/us101_sweep.py`; target speed KS < 0.2 with headway KS < 0.15
- if it works: add `us101` to the planted runs and redo the density and calibration slides

### 2.4 RL ego collisions at the junction (open item 6)
- state: 4 of 200 episodes with a junction collision (episodes 1, 39, 85, 134); 39 and 85 are a crossing NPC driving into the ego
- try: end the episode on a collision (not just warn), add a reward for being stopped at the stop line while a crossing car has right of way, log who hits whom per episode; compare collisions and stalled episodes against speed mode 31
- file: `scripts/sumo_urban_agent.py` (`compute_reward`, `run_episode`)

### 2.5 after Dr. Daher's proximity decision
- if headway: make `headway_features` the reference in `model/aggressiveness_model.py`, refit the cut offs (`scripts/fit_cutoffs.py`), rerun every script that writes `data/`, rebuild the deck, update the hand check pages and tests
- redo the NGSIM ellipse replay video with the fixed features (`scripts/ngsim_replay_sumo.py`, needs sumo-gui and a screen recorder or ffmpeg)

### 2.6 smaller items
- percentile thresholds from retrieval with a dense traffic reference (needs 2.1 or a calibrated 2.3)
- paper outline for IEEE ITSC
- validate the deck with the pptx skill validator (needs Python 3.10+; not on this Mac)
