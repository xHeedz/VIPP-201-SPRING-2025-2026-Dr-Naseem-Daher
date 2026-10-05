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
- next: missed aggressive cost 2, weight agent score in the observation, planted SUMO drivers as episodes (three classes); confirm the direction with Dr. Daher before option b (inverse RL on NGSIM).

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
