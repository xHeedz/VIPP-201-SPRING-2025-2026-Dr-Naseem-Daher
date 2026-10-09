# fri 9 oct 2026 (continued): all five German datasets, traffic state context, IRL with desired headway

## inD, rounD, uniD

unpacked to `../12-10-2026/{inD,rounD,uniD}/` (levelXdata, research only). no leader ids and no lane positions: `datasets/levelx_urban.py` measures the inputs the pNEUMA way (`datasets/pneuma.py` per_second: cone gap +-1.6 m, heading within 30 deg; wave 0, three term score). pedestrians dropped; bicycles and motorcycles can lead but are not scored; parked vehicles (top speed below 2 km/h over the whole track) dropped: before this, inD traffic speed had a median of 2.8 km/h (parked cars counted as traffic), after it 18.6 km/h.

`scripts/score_german.py` now covers all five (highD and exiD: highway; inD, rounD, uniD: urban, mix alpha 0.5) and stores the traffic speed (mean speed of the other vehicles at the same second, same direction on highD) per window.

| | windows | vehicles | median speed | traffic speed | median score metres / headway / mix | aggressive (>= 42) metres / headway / mix |
|---|---|---|---|---|---|---|
| highD | 149,028 | 103,913 | 92.9 | 104.3 | 39.2 / 44.9 / 42.3 | 43.1% / 54.9% / 50.8% |
| exiD | 103,652 | 64,474 | 86.3 | 94.2 | 37.6 / 40.4 / 39.3 | 39.5% / 46.7% / 44.0% |
| inD | 3,901 | 2,716 | 13.7 | 18.6 | 8.0 / 4.3 / 6.8 | 19.7% / 0.7% / 5.6% |
| rounD | 17,256 | 10,164 | 21.0 | 22.2 | 11.1 / 5.0 / 8.3 | 12.4% / 2.5% / 4.6% |
| uniD | 2,648 | 1,273 | 19.2 | 13.4 | 37.9 / 4.2 / 22.2 | 46.0% / 1.0% / 8.0% |

urban, metres score by density quartile (vehicles within 50 m): uniD 7.5 / 34.8 / 46.6 / 55.4, inD 6.0 / 7.1 / 16.3 / 16.6; headway stays at 3 to 7. the density effect of metres seen in SUMO, NGSIM and pNEUMA, now on German urban data.

## context reference: traffic state instead of own speed

`scripts/context_reference.py`, `data/context_reference.csv`. bin = environment x traffic speed band (20 km/h) x car ahead; reference highway = highD + exiD, urban = inD + rounD + uniD (276,485 windows, 16 full bins). own speed version kept for comparison. test: planted SUMO, test seeds 7 to 9.

| setting (mix) | AUC fixed / own speed context / traffic context | fixed 42: caught / normal flagged | traffic context 90: caught / normal flagged |
|---|---|---|---|
| highway low | 0.873 / 0.629 / 0.816 | 0.771 / 0.185 | 0.218 / 0.011 |
| highway medium | 0.869 / 0.719 / 0.829 | 0.830 / 0.279 | 0.161 / 0.010 |
| jam | 0.882 / 0.845 / 0.879 | 0.916 / 0.624 | 0.535 / 0.015 |
| merge | 0.859 / 0.827 / 0.859 | 0.815 / 0.253 | 0.218 / 0.002 |
| weather | 0.842 / 0.784 / 0.863 | 0.715 / 0.189 | 0.227 / 0.002 |
| urban junction | 0.901 / none / 0.711 | 0.762 / 0.047 | 0.877 / 0.451 |
| roundabout | 0.987 / none / 0.827 | 0.964 / 0.099 | 0.974 / 0.417 |

NGSIM US-101 share flagged (mix): 50.8% with 42, 14.4% with traffic context 90.

reading: on highway the traffic state context keeps the ranking (AUC within 0.05, better in weather) and cuts false alarms to 0.2 to 1.5%; the 90th percentile catches 16 to 54% of aggressive drivers (the threshold is a trade off still to choose). urban fails the other way: SUMO urban drivers, even normal ones, are faster and closer than real German intersection and roundabout traffic, so 42 to 45% of normal SUMO drivers are flagged. a simulation realism gap, not a context gap.

## IRL with desired headway

`model/irl.py`: risk replaced by headway and headway squared (smallest time headway over 2 s / 4 s, no leader = 4 s): a per driver desired headway h* = -theta_h / (2 theta_h2) x 4 s, like the desired speed. tests updated (recovery of a 2 s desired headway).

per vehicle, planted test seeds (`scripts/irl_planted.py`): aggressive vs normal AUC 0.707 to 0.728 (index mix 0.948); median desired headway aggressive 2.51, normal 2.55, conservative 2.62 s.

population fits (`scripts/irl_population.py`, `data/irl_population.csv`, up to 400,000 decisions each):

| group | decisions | desired headway s | discomfort weight | desired speed km/h |
|---|---|---|---|---|
| planted conservative (tau 1.26 s) | 369,617 | 2.94 | -10.1 | 93.0 |
| planted normal (tau 0.84 s) | 400,000 | 2.43 | -5.9 | 98.3 |
| planted aggressive (tau 0.42 s) | 332,628 | 1.63 | -4.4 | 91.3 |
| highD | 400,000 | 4.08 | -41.3 | 148.2 |
| exiD | 400,000 | 4.34 | -27.6 | 127.2 |
| NGSIM US-101 | 30,644 | 3.52 | -4.4 | 74.4 |

highD desired headway by density quartile 4.81 / 4.67 / 4.46 / 3.88 s; exiD 5.83 / 5.17 / 4.68 / 3.78 s.

reading: pooled by type the model orders the planted drivers correctly in desired headway and discomfort weight, so the weak per vehicle result is a data problem (one vehicle gives few informative choices; most seconds are a = 0), not a model problem. desired speed does not separate the types (traffic limits the speed). on German motorways the model is not identified: drivers barely accelerate (discomfort -27 to -41), desired speed (127 to 256 km/h) and desired headway (beyond the 4 s cap) are extrapolations; only the trend with density (shorter desired headway in denser traffic) is usable.

## status

- [x] inD, rounD, uniD loaded and scored; all five German datasets in one script
- [x] traffic state context: highway works (AUC kept, false alarms about 1%); urban fails on SUMO realism
- [x] IRL with desired headway; population fits order the planted types; German motorway fits not identified
- [ ] IRL per driver with pooling across similar drivers (hierarchical) or longer observation
- [ ] context threshold: choose the percentile (90 is strict) on labelled data
- [ ] SUMO urban calibration against inD / rounD (speed and gap distributions)
- [ ] DriveDNA full release (after Dr. Daher's access)
