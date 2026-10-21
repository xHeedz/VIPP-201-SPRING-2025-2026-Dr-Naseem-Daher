# tue 20 to wed 21 oct 2026: Dr. Daher's revision list

requests (meeting, week of 18 oct): add weights to the state and shrink it, square the acceleration and remove the square of the speed, revisit the reward and the action space, clean and deeply analyse every dataset before use, rerun every model on every dataset, backtrack what each model computes and why, explain with a video. decisions by Hadi: state = weighted terms then shrink; refit weights and cut offs; gap = metres and seconds mixed per environment (PROX_MIX) as the reference; video = one driver second by second.

## 1. audit and cleaning

`scripts/audit_datasets.py` (`data/dataset_audit.csv`, `_columns.csv`): same checks on 13 sources, 9.9 M rows, about 258,000 vehicles. clean: no duplicates, no NaN, no speed above 200 km/h, almost no acceleration above 8 m/s2.

| problem | where | size |
|---|---|---|
| gap 0 = "camera saw no car" (read as nobody close) | UAH | 53% of rows |
| lane offset missing stored as 0 | pNEUMA, inD, rounD, uniD (no lanes), DriveDNA (lane lost), UAH | 100%, 50%, 5% |
| impossible lane offset | NGSIM Lankershim (to 31 m), Peachtree (23 m), I-80 (4.6 m); SUMO invalid code 2^30 | 0.1 to 1% (SUMO 11 rows) |
| impossible gap (under 2 m or 0.3 s while moving) | rounD 1.2%, others under 0.4% | |
| short tracks | highD 13 s, exiD 15 s, inD 9 s, rounD 14 s, uniD 17 s median per vehicle | |
| GPS timing jitter (0.8 to 1.2 s) | UAH | harmless, accel uses the real step |

`datasets/cleaning.py`: rule 1 drop |accel| above 8 or at the 9 clip; rule 2 wave unknown (no lanes, lane not valid, above 2 m); rule 3 impossible gap unknown; rule 4 UAH gap 0 = unknown, elsewhere 0 = no car ahead; `leader` flag (1 / 0 / NaN). `scripts/build_clean_tables.py` -> `data/clean/*.csv.gz` (git-ignored), `data/cleaning_log.csv`. UAH: 16,434 gap rows unknown, 836 wave; NGSIM 99 to 1,236 wave rows; planted 5 wave, 368 gap.

## 2. reference formula

`model/aggressiveness_model.py` (`reference_terms`, `reference_score`, `reference_label`; old functions unchanged): speed / 150 (linear), (|a| / 5)^2, gap mix per environment, lane / 1.5; unknown terms left out, weights of the known terms rescaled. windows: `datasets/windows.py` (10 s, step 5 s, term unknown when more than half its seconds are unknown).

`scripts/fit_reference_weights.py`: 1,771 weight vectors (0.05 grid, sum 1), criterion mean of UAH A vs N, planted A vs N, planted N vs C AUC (seeds 0 to 6). weights 0.25 / 0.35 / 0.30 / 0.10 (speed, accel^2, gap, lane), cut offs 19.0 / 27.5 (`data/reference_weights.json`). tests: UAH LODO 0.841 (old 0.829; D1 0.606 -> 0.761), planted A vs N 0.888, N vs C 0.791 (old windows 0.72 to 0.92 and 0.64 to 0.76).

`scripts/rerun_formula_all.py` (`data/rerun_formula_summary.csv`): mean points per window, speed 4 to 20, gap 3 to 12, lane 0 to 3.5, accel^2 0.1 to 1.1 on every source. accel^2 barely counts: (2 / 5)^2 = 0.16. aggressive share: highD 25% (51% before), exiD 18%, US-101 11% (51%), I-80 4%, pNEUMA 5%, inD / rounD / uniD 0.5 to 0.7%, DriveDNA sample 7%; UAH aggressive trips 79% of windows, normal 27%; planted aggressive 53%, normal 2%, conservative 0%.

## 3. supervised model, and a correction

`scripts/rerun_supervised.py`: DynamicWeightAgent with the reference terms (unknown terms rescaled). the agent has one threshold per environment (highway about 8 to 12, urban 22 to 31), so its raw scores of different environments are on different scales; ranking them together (as `scripts/train_uah.py` evaluate did) understated its AUC. measured on the margin (score minus the window's environment threshold):
- old agents (`data/uah_lodo_agents`, old features): 0.897 (reported 0.762), against the old formula 0.829
- new terms: 0.868 (raw pooled 0.556), formula 0.842; accuracy 0.763 vs always normal 0.626; highway weight on accel^2 0.73 to 0.79
- applied to planted test seeds: 0.671 (formula 0.858)
the 18 oct statement "supervised below the formula" was a measurement error.

## 4. state, reward, actions (PPO and DQN)

`env/labeling_ref.py` (configurable state from weighted terms, reward and actions; old env unchanged), `tests/test_labeling_ref.py` (3 tests, hand values), `scripts/state_reward_ablation.py`: planted, seeds 0 to 6 train, 7 to 9 test (8,863 vehicles), 1M steps, gamma 1, wait bias 4, one seed.

| config | inputs | PPO | DQN | normal right (PPO) | seconds (PPO) |
|---|---|---|---|---|---|
| full | 16 | 0.689 | 0.721 | 0.48 | 24.4 |
| weighted points + time + env | 12 | 0.704 | | 0.56 | 26.5 |
| small6: score run / rec, gap while following, car ahead share, time, env | 8 | 0.710 | 0.710 | 0.50 | 29.0 |
| small4: no gap while following, no env | 4 | 0.644 | 0.624 | 0.51 | 21.6 |
| tiny: score run, time | 2 | 0.612 | | 0.49 | 26.1 |
| small6 + growing wait cost (0.0001 per s) | 8 | 0.693 | | 0.57 | 30.0 |
| small6 + ordinal cost (far mistake 2) | 8 | 0.699 | | 0.61 | 30.4 |
| small6 + wait 5 s action | 8 | 0.722 | | 0.55 | 32.7 |
| formula, per environment cut offs, 30 / 60 s | | 0.679 / 0.749 | | 0.49 / 0.62 | |

reading: 16 -> 8 inputs loses nothing; gap while following and environment carry 0.07; no reward, action or algorithm raises the ceiling (all 0.70 to 0.72 in about 30 s, formula 0.75 at 60 s).

## 5. backtrack

- `scripts/backtrace_models.py` -> `docs/hand_checks/backtrace_formula.md` (8 windows, raw seconds to label), `backtrace_supervised.md` (weights per fold, single term AUC per driver: gap 0.69 to 0.87, accel^2 0.60 to 0.83, speed 0.59 to 0.84, lane 0.47 to 0.61), `backtrace_irl.md` (one car, choice probabilities)
- `scripts/backtrace_agents.py` -> `docs/hand_checks/backtrace_agents.md`: PPO small6 + wait 5 s, car never following: conservative 0.95, normal 0.10, aggressive 0.09; following over 75% of the time: 0.60 / 0.74 / 0.95. 76% of PPO mistakes are normal and conservative mixed up. PPO learned "no close car -> conservative"; DQN (full) gets 0.80 of freely driving aggressive cars. missing: how free driving differs by type (speed relative to the limit or the traffic)
- inverse RL (`scripts/rerun_irl.py`): all seconds A vs N 0.728, N vs C 0.815 (formula 0.956 / 0.903); following seconds only 0.727 / 0.853. per car preferred gap about 2.5 s for every type with the pull, 1.77 / 2.54 / 2.82 s without it (lam 0) but AUC 0.690 / 0.652: the limit is information per car (about 100 decisions, mostly keep speed), not the pull

## 6. video, deck, workstation

- `scripts/make_explainer_video.py`: GIF replay of one test driver (speed, gap, the 8 inputs, button probabilities), trace CSV next to it. `../20-10-2026/explainer_correct_follows.gif` (normal, follows at about 30 m, labelled normal after 45 s, 37% sure) and `explainer_wrong_never_follows.gif` (normal, car ahead beyond 100 m, labelled conservative after 31 s, 83% sure)
- deck `../20-10-2026/Weekly Update.pptx` (10 slides, `scripts/build_revision_deck.py`), speaker notes `../20-10-2026/Speaker Notes.pdf` / `.docx`
- full DriveDNA: `scripts/drivedna_full.py`, `docs/workstation_drivedna.md`; tested on two sample drives through a mock of the release layout (parquet read untested here, no pyarrow)

## status

- [x] audit and cleaning of every dataset
- [x] reference formula, weights and cut offs refitted
- [x] formula, supervised, PPO, DQN, inverse RL rerun on the cleaned data
- [x] state shrunk 16 -> 8, reward and action variants
- [x] backtrack pages for every model, replay videos
- [ ] full DriveDNA on the workstation (Hadi)
- [ ] 3 seeds for PPO and DQN; PPO and DQN on UAH with the new state
- [ ] Dr. Daher: squared acceleration (keep or hard braking only), speed relative to limit / traffic, simulated type overlap
