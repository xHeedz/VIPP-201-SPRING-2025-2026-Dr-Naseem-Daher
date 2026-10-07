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

## highD loader

`datasets/highd.py` (`recordings`, `read_recording`, `per_second`), from the documented file format; highD download in progress (Hadi). per second table of `datasets/uah.py` plus vehicle id, class, lane, speed limit:
- speed: |velocity| at each whole second (25 Hz, frame % 25 = 0), 3 s mean; accel = its derivative
- gap: own front bumper to the leader's rear bumper (`precedingId`), from the boxes (x, y = top left corner, width = length). highD `dhw` is front to front and includes the leader's length, so it is kept as `dhw_m` for a check only
- wave: |box centre y - centre of the lane between consecutive markings|, upper markings for driving direction 1, lower for 2

test `tests/test_new_loaders.py::test_highd_gap_wave_speed` with hand values: gap 25.5 m (right) and 26.0 m (left), lane offset 0.25 m, 108 km/h.

## status

- [x] planted drivers as labeling episodes, three classes, 3 PPO seeds
- [x] per environment cut off baseline
- [x] highD loader and test
- [ ] run the highD loader on the real files, check gap against dhw minus the leader length, hand check one window
- [ ] score highD (metres and headway), per lane and per density bin, compare with NGSIM
- [ ] exiD loader once the files are here
- [ ] labeling RL with headway features (`headway_features`) on the planted drivers
