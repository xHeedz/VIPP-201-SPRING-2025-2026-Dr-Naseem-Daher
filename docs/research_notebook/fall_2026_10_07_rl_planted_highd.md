# wed 7 oct 2026: labeling RL on the planted drivers, highD loader

## labeling RL on the planted SUMO drivers

`env/labeling_env.py`: `planted_sequences` (one sequence per planted vehicle, seconds below 5 km/h dropped, at least 10 s), environment one hot extended to highway / urban / weather (UAH observations get one input that is always 0). `scripts/train_labeling_rl_planted.py`: three classes, horizon 60 s, wait cost 0.002, PPO 1M steps. split by seed: seeds 0 to 6 train (22,811 vehicles), 7 to 9 test (8,863 vehicles with at least 60 s: 1,824 conservative, 5,271 normal, 1,768 aggressive), so test drivers and traffic runs are never seen in training.

baselines on the same test vehicles, label from the reference AI of the mean features after a fixed T:
- reference: cut offs 29 / 42
- refit: one pair of cut offs refitted on the training seeds per T
- refit per environment: one pair per environment per T (the PPO observation contains the environment, so this is the fair baseline). cut offs at 60 s: highway 30 / 43, urban 26 / 41, weather 21 / 30

| method | balanced acc | conservative | normal | aggressive | seconds | balanced reward |
|---|---|---|---|---|---|---|
| PPO seed 0 | 0.676 | 0.700 | 0.430 | 0.900 | 11.5 | 0.335 |
| PPO seed 1 | 0.666 | 0.595 | 0.542 | 0.861 | 6.8 | 0.322 |
| PPO seed 2 | 0.667 | 0.601 | 0.538 | 0.863 | 8.1 | 0.322 |
| refit per environment, 10 s | 0.638 | 0.629 | 0.524 | 0.761 | 10 | 0.258 |
| refit per environment, 30 s | 0.667 | 0.738 | 0.483 | 0.779 | 30 | 0.276 |
| refit per environment, 60 s | 0.685 | 0.731 | 0.490 | 0.832 | 60 | 0.251 |
| refit, 60 s | 0.665 | 0.714 | 0.502 | 0.780 | 60 | 0.213 |
| reference 29 / 42, 60 s | 0.663 | 0.748 | 0.460 | 0.780 | 60 | 0.207 |

balanced acc by setting, PPO seed 0 against refit per environment at 60 s: highway low 0.642 / 0.638, highway medium 0.691 / 0.696, jam 0.665 / 0.651, merge 0.687 / 0.727, roundabout 0.704 / 0.717, urban 0.738 / 0.741, weather 0.660 / 0.666.

reading:
- PPO (mean of 3 seeds 0.670) matches the per environment index at 30 s (0.667) after 7 to 12 s, and is 0.015 below it at 60 s. the gain is speed, not accuracy.
- PPO catches more aggressive drivers (0.86 to 0.90 against 0.76 to 0.83) at the cost of conservative and normal ones; normal stays the hardest class for every method (0.43 to 0.54).
- the first comparison against one pair of cut offs (0.676 vs 0.665) overstated the gain: most of it was the environment, as the weather setting shows (0.534 with one pair of cut offs, 0.666 per environment).
- unlike UAH, waiting pays here: the per environment index goes from 0.638 at 10 s to 0.685 at 60 s, and the per vehicle labels mean every second belongs to the labelled driver.

outputs `data/labeling_rl_planted_{summary,by_scenario}_seed{0,1,2}.csv`, `data/labeling_rl_planted_episodes_seed{0,1,2}.csv.gz`.

note: the PPO rows of this table were trained with gamma 0.99 (stable-baselines3 default). the files with these names were later overwritten by the gamma 1 rerun (section below: 0.681 / 0.663 / 0.681); the baselines are unchanged.

## highD loader

`datasets/highd.py` (`recordings`, `read_recording`, `per_second`), from the documented file format; highD download in progress (Hadi). per second table of `datasets/uah.py` plus vehicle id, class, lane, speed limit:
- speed: |velocity| at each whole second (25 Hz, frame % 25 = 0), 3 s mean; accel = its derivative
- gap: own front bumper to the leader's rear bumper (`precedingId`), from the boxes (x, y = top left corner, width = length). assumed here that highD `dhw` is front to front; checked on the real data (9 Oct): `dhw` is bumper to bumper and equals the computed gap to within 1 cm, kept as `dhw_m`
- wave: |box centre y - centre of the lane between consecutive markings|, upper markings for driving direction 1, lower for 2

test `tests/test_new_loaders.py::test_highd_gap_wave_speed` with hand values: gap 25.5 m (right) and 26.0 m (left), lane offset 0.25 m, 108 km/h.

## proximity: metres and headway mixed per environment

Dr. Daher (Oct 2026): proximity as a mix of gap in metres and time headway, depending on the environment. `model/aggressiveness_model.py` `mixed_features`: n_p^2 = alpha x metres term + (1 - alpha) x headway term. `scripts/fit_proximity_mix.py`: alpha 0 to 1 per environment, AUC aggressive vs normal windows (planted seeds 0 to 6 fit, 7 to 9 check, UAH), density effect = median score of normal drivers in the jam minus low density highway. `data/proximity_mix_grid.csv`, `data/proximity_mix.json`.

| environment | metres (alpha 1) | headway (alpha 0) | fitted alpha | AUC at fitted alpha | UAH at fitted alpha (metres) |
|---|---|---|---|---|---|
| highway, SUMO | 0.750 | 0.789 | 0.5 | 0.869 (check 0.866) | 0.768 (0.792) |
| urban, SUMO | 0.889 | 0.936 | 0.5 | 0.938 | 0.883 (0.893) |
| weather, SUMO | 0.796 | 0.816 | 0.2 | 0.817 | none |

density effect (normal drivers, light traffic to jam): metres +16.5, headway -1.6, mix 0.5 +8.7. a mix beats both pure measures in SUMO highway (the two terms carry partly different information); UAH alone prefers metres. `PROX_MIX` saved as a proposal; the reference still uses metres.

## labeling RL: discount and wait bias

- gamma: stable-baselines3 default 0.99 makes a label after n seconds worth 0.99^n (60 s: 0.55, UAH 120 s: 0.30), far more than the wait cost (0.002 x 59 = 0.12). with gamma 0.99, 52 to 56% of PPO decisions came after 1 s at about 50% accuracy. now `--gamma` default 1. effect alone small (metres 3 seeds 0.663 to 0.681, decisions at 1 s 44 to 50%; headway 72% at 1 s).
- wait bias: the untrained policy picks each action with 1/4, three end the episode, so early episodes last about 1.3 s and the policy learns to commit at once. `--wait-bias 4` adds 4 to the wait logit of the untrained policy (wait with 0.95 per second). reward and env unchanged.

with gamma 1 and wait bias 4 (balanced acc, seconds, balanced reward):

| features | seed 0 | seed 1 | seed 2 | fixed index per environment, 30 s / 60 s |
|---|---|---|---|---|
| metres | 0.659, 16.6 s, 0.291 | | | 0.667 / 0.685 |
| headway | 0.742, 31.1 s, 0.429 | 0.729, 29.6 s, 0.406 | 0.736, 28.2 s, 0.423 | 0.677 / 0.741 (reward 0.297 / 0.365) |
| mix | 0.730, 27.7 s, 0.413 | 0.705, 18.8 s, 0.378 | 0.709, 21.4 s, 0.381 | 0.678 / 0.740 |

headway: PPO mean 0.736 after about 30 s, against 0.677 for the per environment index at 30 s and 0.741 at 60 s; aggressive caught 0.83 to 0.88; decisions at 1 s 72% to 12%. metres: still below the index (31% at 1 s).

## hand checks of RL decisions

`scripts/rl_decision_trace.py`, `docs/hand_checks/rl_decision_trace_*.md`; first second recomputed by hand, matches the code to 2 decimals in every trace.
- metres policy (gamma 0.99, no wait bias), aggressive `aggressive_r0.13`: 98 km/h, no car within 50 m, AI 23.96; labelled conservative after 1 s (wrong). the case that exposed the early commit.
- headway policy (wait bias 4), aggressive `aggressive_r0.14`: 104 km/h, 23.9 m gap, headway 0.83 s, AI 70.47; aggressive after 1 s (p 0.99), correct.
- headway policy, normal `normal_r0.35`: AI 43.8 to 49.9, above the 42 cut off (fixed index: aggressive); P(wait) 0.98 to 0.56 over 18 s, P(normal) 0.02 to 0.44; labelled normal after 21 s, correct, reward +0.96.

## US-101 synchronized flow

`scripts/planted_drivers.py`: `CAR_FOLLOW`, `US101_MAIN2_SPEED`, `US101_MAIN2_LANES` (defaults IDM, 29.06 m/s, 5 lanes: planted runs unchanged); `scripts/us101_sweep.py`: `US101_CFM`, `US101_MAIN2_SPEED`, `US101_MAIN2_LANES`. NGSIM US-101: median 48.2 km/h (p25 38.4, p75 57.3), headway median 1.49 s.

| model | after the weave | best demand (main, ramp veh/h) | median km/h | speed KS | headway KS |
|---|---|---|---|---|---|
| IDM | 5 lanes, 65 mph | 6,000, 600 | 44.9 (p25 0) | 0.418 | 0.209 |
| EIDM | 5 lanes, 65 mph | 8,000, 1,200 | 86.4 | 0.821 | 0.085 |
| Krauss | 5 lanes, 65 mph | 7,000, 1,200 | 85.2 | 0.784 | 0.176 |
| Krauss | 5 lanes, 54 km/h | 6,000, 1,200 | 78.5 | 0.554 | 0.210 |
| EIDM | 5 lanes, 54 km/h | 8,000, 1,200 | 83.3 | 0.696 | 0.073 |
| Krauss | 3 lanes | 5,000, 1,200 | 9.9 | 0.632 | 0.211 |
| EIDM | 4 lanes | 7,000, 1,200 | 66.6 | 0.434 | 0.070 |
| EIDM | 3 lanes | 5,750, 1,200 | 41.8 (p25 20.4, p75 70.2) | 0.259 | 0.063 |

reading: EIDM and Krauss stay in free flow at any demand (SUMO inserts a car only when there is room); congestion needs a capacity bottleneck downstream. best: EIDM with a lane drop to 3, speed KS 0.43 to 0.26 and headway KS 0.063; spread still too wide (p25 20 vs 38). target < 0.2 not reached.

## paper outline and deck

- `docs/paper_outline.md`: IEEE ITSC, claims with their numbers, sections, figures, what is missing.
- `scripts/build_week_deck.py` -> `../28:9:2026/Presentation/VIPP 301A Update 7 October 2026.pptx` (8 slides, October style): correction, proximity mix, MDP, RL results, traced decision, US-101, decisions. render checked through Keynote.

## status

- [x] planted drivers as labeling episodes, three classes, 3 PPO seeds
- [x] per environment cut off baseline
- [x] highD loader and test
- [ ] run the highD loader on the real files, check gap against dhw minus the leader length, hand check one window
- [ ] score highD (metres and headway), per lane and per density bin, compare with NGSIM
- [ ] exiD loader once the files are here
- [x] labeling RL with headway and the mix (gamma 1, wait bias 4, 3 seeds)
- [x] proximity mix per environment fitted (proposal)
- [x] hand checks of RL decisions
- [~] US-101: EIDM with lane drop, speed KS 0.26 (target 0.2)
- [x] paper outline, week deck
- [ ] Dr. Daher: adopt the mix as reference (refit cut offs, rerun every result)?
- [ ] gamma and wait bias ablation over 3 seeds each (paper)
- [ ] metres features: why the RL labeler stays below the index
