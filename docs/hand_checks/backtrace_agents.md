# backtrace: where the labeling agents go wrong

planted test vehicles (seeds 7 to 9, at least 60 s). 'following' = share of the watched seconds with a car ahead within 3 s; only then can the type specific following gap (tau 1.26 / 0.84 / 0.42 s) show.

## PPO, state small6_wait5

confusion (rows: true type, share of that type)

| true \ predicted | conservative | normal | aggressive |
|---|---|---|---|
| conservative | 0.82 | 0.13 | 0.06 |
| normal | 0.40 | 0.55 | 0.05 |
| aggressive | 0.07 | 0.12 | 0.81 |

share of all mistakes that are normal and conservative mixed up: 76%

accuracy by how much the car followed another car while watched

| following | episodes | accuracy | normal correct | conservative correct | aggressive correct |
|---|---|---|---|---|---|
| never | 1848 | 0.47 | 0.10 | 0.95 | 0.09 |
| under 25% | 746 | 0.42 | 0.32 | 0.85 | 0.03 |
| 25 to 75% | 1609 | 0.61 | 0.54 | 0.81 | 0.60 |
| over 75% | 4660 | 0.78 | 0.74 | 0.60 | 0.95 |

accuracy per scenario (balanced over the three types)

| scenario | balanced accuracy | following share |
|---|---|---|
| highway_low | 0.55 | 0.21 |
| highway_medium | 0.73 | 0.47 |
| jam_jam | 0.77 | 0.83 |
| merge_medium | 0.74 | 0.61 |
| roundabout_medium | 0.73 | 0.86 |
| urban_medium | 0.77 | 0.65 |
| weather_medium | 0.67 | 0.49 |

## DQN, state full

confusion (rows: true type, share of that type)

| true \ predicted | conservative | normal | aggressive |
|---|---|---|---|
| conservative | 0.84 | 0.15 | 0.01 |
| normal | 0.33 | 0.64 | 0.03 |
| aggressive | 0.04 | 0.16 | 0.80 |

share of all mistakes that are normal and conservative mixed up: 69%

accuracy by how much the car followed another car while watched

| following | episodes | accuracy | normal correct | conservative correct | aggressive correct |
|---|---|---|---|---|---|
| never | 1477 | 0.57 | 0.29 | 0.88 | 0.80 |
| under 25% | 1055 | 0.46 | 0.28 | 0.88 | 0.35 |
| 25 to 75% | 1725 | 0.59 | 0.54 | 0.83 | 0.43 |
| over 75% | 4606 | 0.80 | 0.82 | 0.65 | 0.82 |

accuracy per scenario (balanced over the three types)

| scenario | balanced accuracy | following share |
|---|---|---|
| highway_low | 0.64 | 0.17 |
| highway_medium | 0.71 | 0.44 |
| jam_jam | 0.78 | 0.83 |
| merge_medium | 0.71 | 0.66 |
| roundabout_medium | 0.78 | 0.86 |
| urban_medium | 0.84 | 0.64 |
| weather_medium | 0.63 | 0.48 |
