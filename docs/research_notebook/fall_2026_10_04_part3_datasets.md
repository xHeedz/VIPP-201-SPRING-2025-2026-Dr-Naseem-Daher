# sun 4 oct 2026: part 3, more and better datasets

dataset reference with every label definition: `docs/datasets.md`.

## new data

- NGSIM I-80, Lankershim, Peachtree extracted from the 2 GB combined file (40 s): 4.57 M, 1.61 M and 0.87 M rows, in `../28:9:2026/ngsim_{site}.csv`
- pNEUMA, one slice (drone 1, 24 Oct 2018, 08:30 to 09:00, 86 MB, 922 vehicles over 13.6 min, 25 Hz) in `../28:9:2026/pneuma/`. data source pNEUMA, open-traffic.epfl.ch, CC BY-NC 4.0 (terms accepted 4 Oct 2026, non commercial use, attribution in `docs/datasets.md`)
- [ ] highD / exiD / rounD: access by application form on levelxdata.com (Hadi), approval takes days

## loaders, one table for every source

`datasets/ngsim.py` `per_second`, `datasets/pneuma.py` `per_second`, `datasets/sumo_log.py` `per_second`, `datasets/uah.py` `load_trip`: all give t, speed_kmh, accel, gap_m, wave_m, then the same `windows` and `index_features`.

- NGSIM arterials (`trajectories(..., planar=True)`): cars drive both ways along Local_Y and turn, so speed comes from the 2-D path (with Local_Y only, every southbound car would read speed 0 after the clip), and only through traffic on road sections is scored (Int_ID 0, Movement 1, north or southbound). the lane centre is a median per (direction, section, lane, 10 m of road): the arterials curve in the fixed Local_X / Local_Y frame (within lane spread of Local_X 2.1 m on Lankershim and 2.2 m on Peachtree, against 0.6 m on US-101), and one centre per lane turned that curvature into wave (offset above 1.5 m: Lankershim 24.3% of rows with one centre per lane, 1.9% with the 10 m centre; Peachtree 10.0% and 0.9%). the freeways keep one centre per lane (US-101 median offset 0.34 m, 0.24 m with the 10 m centre).
- pNEUMA: no lanes, no leaders. gap = nearest vehicle ahead whose centre is within +-1.6 m of the heading line and heading within 30 deg, minus half of both lengths (by vehicle type). wave cannot be measured without a lane map and is 0 (pNEUMA scores are three term). motorcycles are leaders but are not scored. leader found 67% of seconds, median gap 11.1 m, median time headway 1.84 s; 13% of gaps under 2 m (queues at lights, plus some cars alongside caught by the cone).

## finding 15: NGSIM recording periods

the combined file reuses Vehicle_ID and restarts Frame_ID in each recording period (I-80: 3, Lankershim: 2, Peachtree: 2). every duplicate (Vehicle_ID, Frame_ID) pair was conflicting (different time and position, 0 exact copies): 405,708 rows in I-80, 115,277 in Lankershim, 53,867 in Peachtree. `read_raw` kept the first of each pair, which splices two different cars into one trajectory. fix: `_split_periods` (Global_Time minus 100 ms x Frame_ID is constant within a period; with more than one period, ids become period x 100000 + id, leaders too). all rows kept now; trajectories: I-80 5,678, Lankershim 2,442, Peachtree 2,337. the US-101 5 min extract has one period and is unchanged (hand check still passes). anything that loaded more than one period of the full file before today was affected.

## unlabelled sets: scores per road type (`scripts/score_unlabelled.py`, `data/unlabelled_summary.csv`, `results/figures/unlabelled_scores.png`)

| site | road | windows | median speed (km/h) | median score | p95 | share >= 70 | median, headway | share >= 70, headway | prox points | wave points |
|---|---|---|---|---|---|---|---|---|---|---|
| NGSIM US-101 (5 min) | freeway | 5,382 | 46.9 | 48.4 | 75.1 | 10.2% | 36.0 | 3.1% | 30.8 | 10.1 |
| NGSIM I-80 (15 min) | freeway | 19,915 | 25.5 | 57.8 | 81.8 | 20.7% | 26.6 | 1.3% | 41.8 | 10.6 |
| NGSIM Lankershim | arterial | 12,788 | 36.8 | 49.1 | 76.7 | 14.3% | 25.0 | 0.8% | 31.6 | 9.1 |
| NGSIM Peachtree | arterial | 7,038 | 31.0 | 34.9 | 73.2 | 7.8% | 18.2 | 0.1% | 23.2 | 9.5 |
| pNEUMA Athens | downtown | 4,810 | 23.2 | 36.8 | 69.0 | 4.0% | 11.3 | 0.1% | 31.8 | 0 |
| UAH normal (labelled) | motorway, secondary | 2,591 | 90.7 | 29.3 | 59.4 | 0.7% | 31.9 | 8.1% | | |
| UAH aggressive (labelled) | motorway, secondary | 1,562 | 104.5 | 54.5 | 80.1 | 23.4% | 66.6 | 46.7% | | |

- with the gap in metres, ordinary congested freeway traffic outscores the UAH aggressive drivers: I-80 (median 57.8 at 26 km/h) is above the UAH aggressive median (54.5). almost all of it is proximity (41.8 of 57.8 points; speed 2.0).
- with time headway the order becomes plausible: every unlabelled site (11 to 36) sits near or below UAH normal (31.9), and UAH aggressive (66.6) is far above. the share of NGSIM windows at 70 or more drops from 8 to 21% to 0.1 to 3%.
- wave gives about 9 to 11 points at every NGSIM site; the arterial value was checked by hand (`docs/hand_checks/hand_check_arterial_pneuma.md`).

## scores per density bin (`data/unlabelled_density.csv`, `results/figures/unlabelled_density.png`)

NGSIM, vehicles within +-100 m in the same direction, per km per lane:

| road | density | windows | median speed | median score | median, headway | prox points | share >= 70 |
|---|---|---|---|---|---|---|---|
| freeway | 20 to 40 | 7,621 | 35.5 | 50.1 | 30.6 | 33.4 | 11.6% |
| freeway | 40 to 60 | 15,630 | 26.8 | 57.2 | 28.7 | 41.1 | 19.6% |
| freeway | > 60 | 1,982 | 14.8 | 65.7 | 21.5 | 50.0 | 36.5% |
| arterial | < 10 | 1,354 | 39.0 | 24.5 | 19.8 | 13.6 | 6.6% |
| arterial | 10 to 20 | 3,580 | 37.9 | 37.2 | 22.1 | 23.1 | 9.9% |
| arterial | 20 to 40 | 8,690 | 34.8 | 47.7 | 23.4 | 31.0 | 12.7% |
| arterial | 40 to 60 | 3,295 | 23.7 | 54.0 | 19.3 | 36.2 | 18.1% |
| arterial | > 60 | 523 | 19.0 | 59.4 | 19.0 | 41.8 | 24.9% |

(freeway bins below 20 veh/km/lane hold only 6 and 58 windows.) pNEUMA quartiles of vehicles within 50 m: median score 20.2 / 34.0 / 40.9 / 48.1 in metres, 5.4 / 11.8 / 14.3 / 17.9 with headway.

the planted SUMO result holds on real traffic: in metres the index climbs with density at every site (freeway +16 points from 20 to 40 to above 60 veh/km/lane, arterial +35 from below 10 to above 60), driven by proximity; with headway it is flat or falls.

## UAH window labels (bug 6, `scripts/uah_relabel.py`, `data/uah_relabel.csv`)

SEMANTIC_ONLINE column 14, DriveSafe's "ratio aggressive" over the last minute, averaged per window: aggressive trips 0.524, normal trips 0.137. only 46.9% of aggressive trip windows reach 0.5; 0% of normal trip windows do.

leave one driver out, AUC (window labels: aggressive trip windows with ratio >= 0.5 against normal trip windows with ratio < 0.5):

| proximity | labels | aggressive / normal windows per driver | hand-set index | trained agent |
|---|---|---|---|---|
| metres | trip | 260 / 432 | 0.829 | 0.761 |
| metres | window | 122 / 431 | 0.826 | 0.803 |
| headway | trip | 260 / 432 | 0.795 | 0.744 |
| headway | window | 122 / 431 | 0.789 | 0.765 |

the hand-set index does not change; the trained agent gains 0.04 with cleaner labels (it learns weights and a threshold from them). the window labels come from DriveSafe's own algorithm, which uses the same phone signals, so the gain is an upper bound on what better labels give.

## other labelled sets (checked, `docs/datasets.md`)

- Kaggle "Driving Behavior" and the 2022 smartphone set (Data in Brief, PMC8914310): accelerometer and gyroscope only, no speed, gap or lane offset; usable for the accel term only.
- "multi class driver behaviour dataset" (Data in Brief 2025, PMC12019831): in cabin images of distraction, no kinematics; not usable.
- no labelled set with dense traffic exists in the repo yet. that is what decides metres against headway (UAH says metres by 0.03 AUC in free flow; SUMO and the NGSIM density sweep say headway). highD/exiD have no labels either; a labelled dense set remains open.

## status

- [ ] apply for highD and exiD access (Hadi)
- [x] pNEUMA (one slice) and NGSIM arterials
- [x] one loader per dataset, same per second table
- [x] leave one driver out for UAH (no new labelled set to split)
- [x] unlabelled sets: score per road type and per density bin
- [x] label definition of each dataset (`docs/datasets.md`)
- [x] hand check one arterial row and one pNEUMA second (`docs/hand_checks/hand_check_arterial_pneuma.md`)
- [ ] more pNEUMA slices (all 10 drones of one slot) once the loader is checked by hand
