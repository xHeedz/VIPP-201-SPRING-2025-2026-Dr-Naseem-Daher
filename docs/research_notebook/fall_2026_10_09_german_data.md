# fri 9 oct 2026: highD and exiD scored

data in `../12-10-2026/` (Hadi, levelXdata licence, research only, not redistributed): highD, exiD, inD, rounD, uniD (zips), DriveDNA. highD and exiD unpacked to `../12-10-2026/{highD,exiD}/` (4.8 and 6.5 GB). the DriveDNA folder is the sample again (same 63 drives, 5 drivers as `../28:9:2026/drivedna_sample/`, plus 63 dashcam videos): the full release (`HenryYHW/DriveDNA`, 465 drivers) is still not downloaded.

## loaders, checked on the real files

- `datasets/highd.py`: gap from the boxes equals highD `dhw` to within 1 cm on all 10,481 vehicle-seconds with a leader of recording 01 (the loader docstring had assumed `dhw` is front to front: wrong, it is bumper to bumper). `direction` added for density per carriageway. recording 01: 1,047 vehicles, median 109 km/h, leader present 75%, lane offset median 0.26 m.
- `datasets/exid.py` (new): `lonVelocity` per second, 3 s mean, derivative as accel (correlation 0.95 with exiD `lonAcceleration`); gap = `leadDHW` (equals centre distance minus half both lengths: median difference -0.5 cm, sd 8 cm, recording 00); wave = |`latLaneCenterOffset`|; cars, vans, trucks, buses only. recording 00: 1,404 vehicles, median 94 km/h, limit 100 km/h.
- tests: `tests/test_new_loaders.py::test_exid_reads_lead_gap_and_lane_offset` (30 m gap, 0.4 m offset, 90 km/h, pedestrian dropped).

## scores

`scripts/score_german.py`: per second (cached in `../12-10-2026/<set>/per_second/`) to 10 s windows, scores in metres, headway and the highway mix (`PROX_MIX` 0.5); density = scored vehicles in the same recording and second (same direction for highD), quartile bins per dataset. `data/german_windows.csv.gz`, `data/german_summary.csv`.

| | windows | vehicles | median speed | median score metres / headway / mix | labelled aggressive (>= 42) metres / headway / mix |
|---|---|---|---|---|---|
| highD | 149,028 | 103,913 | 92.9 | 39.2 / 44.9 / 42.3 | 43.1% / 54.9% / 50.8% |
| exiD | 103,652 | 64,474 | 86.3 | 37.6 / 40.4 / 39.3 | 39.5% / 46.7% / 44.0% |

by density quartile (median score metres / headway): highD light 38.1 / 42.5, q2 39.2 / 47.0, q3 39.1 / 51.1, dense 41.0 / 39.5 (dense quartile median speed 75 km/h); exiD light 36.5 / 39.0, dense 41.2 / 42.5.

mean points per term:

| | speed | accel | prox metres | prox headway | wave |
|---|---|---|---|---|---|
| highD | 21.6 | 0.8 | 10.1 | 16.8 | 8.2 |
| exiD | 18.5 | 1.0 | 9.3 | 13.1 | 11.3 |

## reading

- about half of ordinary German motorway driving is labelled aggressive with the 42 cut off, in every proximity variant. no single broken feature: normal terms add up. 93 km/h gives (93/150)^2 x 50 = 19 points; a 0.3 m lane offset 8; a leader is present 80 to 91% of the time and an ordinary 1.0 to 1.5 s headway gives 17 to 27 points (the headway term reaches 0 only at 3 s).
- UAH motorway trips drive as fast (median 97.8 km/h) but rarely have a detected leader (finding 5), so the proximity term is mostly 0 there and the cut off fitted on UAH is too low for traffic with a car ahead.
- on German motorways headway scores higher than metres (at 100 km/h a 30 m gap is 1.1 s: metres term (1 - 30/50)^2 = 0.16, headway term (1 - 1.1/3)^2 = 0.40). the US-101 result (metres inflates jams) and this one (headway inflates fast following) point the same way: one absolute cut off cannot serve every context.
- the density effect in metres is small here (+3 to +5 points light to dense) because German dense traffic still moves at 75 to 85 km/h; US-101 congestion is slower.
- consequence: score relative to normal drivers in the same context (percentile among highD / exiD windows with the same speed band and density), or fit cut offs per context. highD and exiD are large enough to serve as the normal reference for motorways.

## status

- [x] highD and exiD loaders checked on the real data, scored
- [ ] DriveDNA full release (not downloaded; folder is the sample)
- [ ] context reference: percentile score among highD / exiD windows in the same speed band and density
- [ ] inD, rounD, uniD (urban, roundabout, campus): unzip and score
- [ ] SUMO planted types calibrated against highD (speed and headway distributions)
- [ ] option b: per driver reward weights from highD trajectories (inverse RL)
