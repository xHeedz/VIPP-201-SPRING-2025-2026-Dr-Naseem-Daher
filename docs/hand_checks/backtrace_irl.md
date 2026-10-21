# backtrace: inverse RL on one planted aggressive car

car `highway_medium_s9/aggressive_r0.45` (highway_medium), 95 decisions. each second the car picks one of 12 accelerations; the model gives each a probability from theta x features (speed reached, gap in seconds over the next 2 s, discomfort).

- decisions at 'keep speed' (a = 0): 74%; within 0.5 m/s2: 92%
- mean probability the fitted model gives the choice actually made: 0.27 (12 equal choices would give 0.08)
- fitted preferred gap 2.54 s; aggressive cars pooled: see data/rerun_irl_summary.csv

| second | speed km/h | gap m | chosen accel | P(chosen) | P(keep speed) | most likely choice |
|---|---|---|---|---|---|---|
| 0 | 144.5 | 121.4 | -2.0 | 0.02 | 0.26 | -0.5 |
| 1 | 136.3 | 114.8 | -1.5 | 0.06 | 0.26 | +0.0 |
| 2 | 126.9 | 113.1 | +0.5 | 0.21 | 0.27 | +0.0 |
| 3 | 126.6 | 111.2 | +1.5 | 0.02 | 0.28 | +0.0 |
| 4 | 132.3 | 107.1 | +1.0 | 0.07 | 0.27 | +0.0 |
| 5 | 136.3 | 101.5 | +0.5 | 0.15 | 0.25 | -0.5 |
| 6 | 138.7 | 94.7 | +0.0 | 0.24 | 0.24 | -0.5 |
| 7 | 139.8 | 87.1 | +0.0 | 0.23 | 0.23 | -0.5 |
| 8 | 139.7 | 79.2 | -0.5 | 0.28 | 0.22 | -0.5 |
| 9 | 138.5 | 71.3 | -0.5 | 0.27 | 0.22 | -0.5 |
| 10 | 136.5 | 63.4 | -1.0 | 0.22 | 0.22 | -0.5 |
| 11 | 133.8 | 55.9 | -1.0 | 0.21 | 0.22 | -0.5 |
| 12 | 130.4 | 48.9 | -1.0 | 0.21 | 0.22 | -0.5 |
| 13 | 126.6 | 42.5 | -1.0 | 0.16 | 0.26 | +0.0 |
| 14 | 122.6 | 85.4 | -0.5 | 0.27 | 0.24 | -0.5 |
| 15 | 118.5 | 80.9 | +0.0 | 0.25 | 0.25 | -0.5 |
| 16 | 115.0 | 28.1 | +0.0 | 0.26 | 0.26 | -0.5 |
| 17 | 112.7 | 25.4 | +0.0 | 0.26 | 0.26 | -0.5 |
| 18 | 111.5 | 23.5 | -0.5 | 0.29 | 0.26 | -0.5 |
| 19 | 111.0 | 22.1 | -0.5 | 0.27 | 0.28 | +0.0 |
| 20 | 110.5 | 21.0 | -0.5 | 0.26 | 0.29 | +0.0 |
| 21 | 109.3 | 20.1 | +0.0 | 0.29 | 0.29 | +0.0 |
| 22 | 108.1 | 19.6 | +0.0 | 0.30 | 0.30 | +0.0 |
| 23 | 107.0 | 19.4 | +0.0 | 0.30 | 0.30 | +0.0 |
| 24 | 105.9 | 19.3 | +0.5 | 0.15 | 0.25 | -0.5 |
