# datasets: what each one measures and what its labels mean

every source goes through the same table before scoring: one row per second with t, speed_kmh, accel, gap_m, wave_m, then `datasets/uah.py` `windows` (10 s, step 5 s) and `model/aggressiveness_model.py` `index_features`.

| source | loader | labels | label unit | speed | accel | gap | wave |
|---|---|---|---|---|---|---|---|
| UAH-DriveSet | `datasets/uah.py` | normal / drowsy / aggressive, acted on instruction | whole trip | GPS 1 Hz | derivative of 3 s smoothed GPS speed | phone camera, car ahead, 0 when none (47% of seconds have one) | phone camera lane offset (ignores the estimator state column, finding 10) |
| UAH, window labels | `scripts/uah_relabel.py` | DriveSafe "ratio aggressive WINDOW" (last 60 s), SEMANTIC_ONLINE col 14 | second | | | | |
| SUMO planted drivers | `scripts/planted_drivers.py`, `datasets/sumo_log.py` | conservative / normal / aggressive by vType (IDM tau, accel, decel, speedFactor, lcAssertive) | vehicle | TraCI, sampled 1 Hz | derivative of 3 s smoothed speed | getLeader + minGap | lateral lane position (SL2015, lcSigma 0.2 for all) |
| NGSIM US-101, I-80 | `datasets/ngsim.py` | none | | Local_Y derivative | accel_1hz (UAH way) | Space_Headway minus leader length | Local_X minus lane median |
| NGSIM Lankershim, Peachtree | `datasets/ngsim.py` (planar) | none | | 2-D path derivative | accel_1hz | as above | Local_X minus median per (direction, section, lane, 10 m of road: the arterials curve in the Local_X frame); through traffic on sections only |
| pNEUMA (Athens) | `datasets/pneuma.py` | none | | pNEUMA speed, 1 Hz | derivative of 3 s smoothed speed | nearest vehicle ahead in a +-1.6 m, 30 deg cone, minus half lengths | not measurable without a lane map: 0 (three term score) |

## label definitions

- **UAH trip labels**: the driver was told to drive normally, aggressively or drowsily for the whole trip. every window inherits the trip label, including calm cruising inside an aggressive trip. DriveSafe rates only 47% of aggressive trip windows as aggressive (ratio >= 0.5) and 0% of normal trip windows (`data/uah_relabel.csv`).
- **UAH window labels**: DriveSafe's own per minute estimate. an algorithm, not a human; it uses accelerations, brakings, car following and weaving from the same phone, so it partly measures what the index measures.
- **SUMO planted labels**: true by construction, but the types differ mainly in IDM tau (a desired time headway), which favours a time headway proximity term.
- **NGSIM, pNEUMA**: no labels. only distributions (per road type, per density bin) and relative statements.

## recording periods (NGSIM)

I-80 (3 periods), Lankershim (2) and Peachtree (2) restart Frame_ID and reuse Vehicle_ID in each period of the combined data.transportation.gov file. `datasets/ngsim.py` `_split_periods` separates them (Global_Time minus 100 ms x Frame_ID is constant within a period). before this, (Vehicle_ID, Frame_ID) de-duplication spliced two cars into one trajectory (finding 15). the US-101 5 min extract has one period.

## candidates checked, not used

| dataset | what it has | usable for |
|---|---|---|
| Kaggle "Driving Behavior" (outofskills) | phone accelerometer + gyroscope, SLOW / NORMAL / AGGRESSIVE | the accel term only (no speed, gap, lane) |
| smartphone sensor dataset, Indian drivers (Data in Brief 2022, PMC8914310) | accelerometer + gyroscope, driving events | the accel term only |
| multi class driver behaviour dataset (Data in Brief 2025, PMC12019831) | 7,286 in cabin images: safe, phone, texting, turning, other distraction | nothing (distraction from images, no kinematics) |
| highD, exiD, rounD (levelXdata) | drone trajectories, German highway, ramps, roundabouts | full index; access by application form (Hadi) |
| INTERACTION, MiTra | merges, roundabouts, intersections; freeway with ramps | full index; not downloaded yet |

## attribution

pNEUMA: data source pNEUMA, open-traffic.epfl.ch (Barmpounakis and Geroliminis, "On the new era of urban traffic monitoring with massive drone data: The pNEUMA large-scale field experiment", Transportation Research Part C, 2020; DOI 10.5281/zenodo.10491409), CC BY-NC 4.0, non commercial use. slice used: drone 1, 24 Oct 2018, 08:30 to 09:00.
