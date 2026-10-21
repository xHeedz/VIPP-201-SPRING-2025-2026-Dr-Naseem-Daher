# sun 18 oct 2026: why each model stops where it does

request from Dr. Daher: understand in depth why every model reaches the number it reaches (93%, 75%, 0.73, ...) and why none reaches 99%. no new models in this page: only measurements on the existing outputs in `data/`. diagnostic scripts are throwaway (not in the repo); every number below can be recomputed from the files named next to it.

## summary: four reasons, in order of size

1. the labels do not say what the windows show (UAH trip labels, 6 drivers). the ceiling is in the data, not in the model.
2. the classes overlap by construction (planted SUMO drivers: desired speed is drawn at random per vehicle, so a fast normal driver and a slow aggressive driver are the same driver for 10 s). the measured accuracy sits on the theoretical ceiling.
3. the observation is too short or too uninformative (10 s windows, IRL decisions where 74 to 90% of seconds are "hold speed"). more time or more informative moments raise every number.
4. some numbers are not accuracy at all (GT vs agent agreement, shares on German and NGSIM traffic). they cannot reach 99% because they do not measure correctness.

## 1. numbers that are not accuracy

### GT vs agent agreement (93.8 / 83.1 / 94.3%)

`scripts/sumo_runner.py`: both sides compute the same index, one from clean TraCI values, one with sensor noise. agreement = share of windows where the noise did not move the score across a cut off. it measures noise robustness, not correctness against a true driver type.

the majority baseline (always predict the most common label) is high because ground truth almost never says aggressive (0.4 to 5.8%). share of the gap from baseline to perfect that is actually closed, (agreement minus baseline) / (100 minus baseline):

| scenario | agreement | majority baseline | gap closed |
|---|---|---|---|
| highway | 93.8% | 62.3% | 84% |
| urban | 83.1% | 68.8% | 46% |
| weather | 94.3% | 92.8% | 21% |

reading: weather has the highest agreement and the worst result. 92.8% is reached by predicting one label for everything. urban is lower because more urban windows sit close to a cut off, where small noise flips the label. this is why the planted drivers (true labels) replaced this test on 4 Oct.

### shares on unlabelled traffic (German 51%, US-101 51% to 14%, DriveDNA 77% vs 0.3%)

highD, exiD, inD, rounD, uniD, NGSIM, pNEUMA and DriveDNA carry no aggressiveness labels. "51% of highD windows aggressive" is a calibration statement (where the 42 cut off falls on that road), not an error rate. worked example from `data/german_windows.csv.gz`, one highD window at exactly 42.0 (mix): 88 km/h gives 17.4 points, a 1.63 s headway 10.0 points (mix), 0.54 m lane offset 14.4 points, accel 0.3. ordinary driving, 42 points. the cut off was fitted on UAH and planted SUMO drivers, never on German motorways. no amount of model work makes these shares "correct" without labels for these roads.

### window accuracy against its baseline (UAH 77% / 81%)

UAH leave one driver out has 432 normal and 260 aggressive windows per held out driver on average: always "normal" scores 62.4%. the index (76.9%) and the DynamicWeightAgent (81.4%) are 14.5 and 19.0 points above that, not 77 and 81 points above zero.

## 2. UAH: the labels set the ceiling

data: 6 drivers, 28 normal or aggressive trips (17 normal, 11 aggressive), 4,153 windows of 10 s (`data/uah_windows.csv`).

### trip labels on window decisions

UAH labels whole trips. every 10 s window of an aggressive trip is labelled aggressive, including straight cruising. DriveSafe's own per minute judgement (`SEMANTIC_ONLINE`, `scripts/uah_relabel.py`) rates only 46.9% of aggressive trip windows as aggressive (ratio at least 0.5), and 0% of normal trip windows. so about half of the "aggressive" windows are, by the dataset's own algorithm, not aggressive.

measured with the reference index (metres, cut off 42):
- 26% of aggressive trip windows score below 42 (missed by construction of the labels, not only by the formula)
- 21% of normal trip windows score 42 or more (a fast overtake on a motorway inside a normal trip)

### more time, higher AUC

same index, same drivers, the score averaged over longer stretches of each trip (mean over the 6 held out drivers, AUC aggressive vs normal):

| observation | 10 s | 30 s | 60 s | 120 s | whole trip |
|---|---|---|---|---|---|
| AUC | 0.831 | 0.849 | 0.866 | 0.897 | 0.944 |

trip accuracy of the index is 93.3% for the same reason: averaging a whole trip removes the calm stretches. the 93% and the 77% are the same model, measured on a trip and on 10 s.

### one driver pulls the mean down

| driver | AUC (10 s) | aggressive windows below the normal median | prox points normal / aggressive | speed normal / aggressive |
|---|---|---|---|---|
| D1 | 0.606 | 32% | 11.8 / 7.2 | 100 / 117 |
| D2 | 0.923 | 3% | 5.5 / 28.9 | 94 / 103 |
| D3 | 0.834 | 11% | 7.1 / 23.4 | 93 / 101 |
| D4 | 0.881 | 7% | 6.7 / 27.5 | 88 / 95 |
| D5 | 0.884 | 1% | 6.5 / 21.6 | 92 / 110 |
| D6 | 0.857 | 7% | 3.1 / 17.0 | 84 / 102 |

without D1 the mean is 0.876. D1 is the only driver whose aggressive trips have fewer proximity points than the normal ones. cause: the phone camera found a car ahead in 10% of D1's aggressive secondary road trip against 59% of the two normal ones (65% and 53%) (`data/uah_trip_coverage.csv`), so the heaviest weighted term (0.8) reads 0 on that trip. D1's mean trip scores on the secondary road are 40.9 aggressive against 43.2 normal: reversed. this is a sensor coverage problem, not a driving style.

the confound also runs the other way for every other driver: aggressive trips have a car detected ahead more often (0.64 to 0.66 against 0.34 to 0.54), partly because tailgating keeps the leader in view. part of the proximity signal on UAH is "camera saw a car", not "driver was close".

### why the DynamicWeightAgent has a lower AUC (0.762) but a higher accuracy (0.814) than the index (0.831 / 0.769)

- AUC measures ranking, accuracy measures one threshold. the agent fits its threshold on the training drivers' label balance, which moves accuracy; the index threshold (42) was not tuned per environment on UAH.
- the agent learns from 5 drivers, 22 trips per fold. its highway weights end at 0.15 / 0.69 / 0.12 / 0.04 (speed, accel, prox, wave): it moved most weight to acceleration and away from proximity, the term that D1 inverts. on a held out driver whose proximity is informative, that ranking is worse.
- trained on window labels instead of trip labels (cleaner targets) the agent gains 0.04 AUC (0.761 to 0.803); the hand-set index does not move (0.829 to 0.826), because it learns nothing from labels. the agent is limited by the labels; the index by its fixed weights.
- weather weights stay at 0.25 each: UAH has no weather trips, so that part of the agent is untrained.

### small sample

with 6 drivers the spread of per driver AUC is 0.11 (sd). the 95% interval of the mean is roughly plus or minus 0.12, so differences of 0.01 to 0.03 between methods on UAH (index vs agent vs PPO vs RAG) are inside the noise.

## 3. planted SUMO drivers: the ceiling is how they were planted

### desired speed is random per vehicle

`scripts/planted_drivers.py` line 198: every vType has `speedDev="0.1"`. SUMO draws each vehicle's speedFactor from a normal distribution: conservative 0.9, normal 1.0, aggressive 1.2, each with sd 0.1 (absolute). on an open road a driver's speed is the clearest sign of type, and these distributions overlap.

best possible separation from desired speed alone (two normal distributions d standard deviations apart: AUC = Phi(d / sqrt 2)):
- normal vs conservative: d = 1, AUC at most 0.760
- aggressive vs normal: d = 2, AUC at most 0.921

measured, per vehicle mean speed on the free highway: 0.744 and 0.881. the data sit on the ceiling.

three classes, best boundaries 0.95 and 1.10 (equal class weight, as in balanced accuracy):

| | conservative | normal | aggressive | balanced |
|---|---|---|---|---|
| ceiling from desired speed | 0.69 | 0.53 | 0.84 | 0.69 |
| index per environment, 60 s (metres) | 0.73 | 0.49 | 0.83 | 0.685 |
| index per environment, 60 s (headway) | 0.83 | 0.58 | 0.81 | 0.741 |
| PPO, headway features (seed 0, about 30 s) | 0.85 | 0.52 | 0.85 | 0.742 |

the metres index reaches the speed ceiling almost exactly. normal is the worst class for every method because it sits between two neighbours and loses drivers on both sides. the headway versions go above the speed ceiling because headway carries the other planted parameter (tau), see below. PPO equals the headway index at 60 s after about half the time: its gain is speed, not information.

### which signal carries the type depends on traffic

per vehicle AUC of single signals over the whole trajectory (all 10 seeds):

| scenario | share of seconds following (headway under 3 s) | aggressive vs normal, mean speed | same, median headway | same, index | normal vs conservative, index |
|---|---|---|---|---|---|
| highway low | 27% | 0.881 | 0.600 | 0.941 | 0.775 |
| highway medium | 58% | 0.742 | 0.785 | 0.973 | 0.836 |
| jam | 87% | 0.544 | 0.957 | 0.931 | 0.819 |
| merge | 75% | 0.585 | 0.924 | 0.937 | 0.826 |
| urban | 41% | 0.497 | 0.990 | 0.966 | 0.728 |
| roundabout | 52% | 0.320 | 0.996 | 0.999 | 0.703 |
| weather | 62% | 0.729 | 0.824 | 0.940 | 0.834 |

- in free flow the type shows in speed; in a jam everyone drives the leader's speed (0.544, a coin flip) and the type shows only in headway (0.957). a single fixed weighted sum cannot weight both right in every scenario. this is the measured reason for the proximity mix and for context.
- over the whole trajectory the index separates aggressive from normal at 0.93 to 0.999. the 10 s window numbers (0.72 to 0.92) are lower only because 10 s is short.
- normal vs conservative stays at 0.70 to 0.84 even with everything: tau 1.26 vs 0.84 s, accel 1.5 vs 2.6, speedFactor 0.9 vs 1.0. the planted gap between those two types is small, so the ceiling is low.
- `sigma` (driver imperfection) is set per type, but SUMO's IDM appears not to read it (its parameter list has no sigma). if so, one of the six planted differences does nothing. to verify in the SUMO source before the paper.

## 4. PPO labeler

### UAH: 0.79 balanced accuracy, 75% of aggressive drivers caught

best variant (`data/labeling_rl_*_wait0.002_miss2_agentobs.csv`):
- confusion over all held out episodes: aggressive 175 caught, 49 missed; normal 315 correct, 62 called aggressive
- per held out driver: D1 0.727, D2 0.872, D3 0.834, D4 0.849, D5 0.828, D6 0.639. D6 drove only one aggressive trip (motorway), so all its aggressive episodes come from one trip
- 61% of decisions come after 1 s (accuracy 0.84 there)

why it cannot pass about 0.83 to 0.86:
- it sees the same four features as the index. at 120 s the best fixed method reaches 0.861; that is roughly the information ceiling of these features on UAH. PPO only chooses when to stop; it adds no information.
- the trip labels make waiting nearly useless: the reference index gains only 0.08 balanced accuracy from 10 to 120 s, while 120 s of waiting costs 0.24 reward. the reward says "decide early", and PPO does.
- 22 training trips per fold: PPO learns from very few independent drivers.

### planted: 0.736 balanced accuracy (headway, mean of 3 seeds), normal at 0.52

- confusion (seed 0, rows true): normal goes to conservative in 37% of episodes. conservative and normal are the overlapping pair from section 3.
- 60% of decisions are made between 30 and 59 s, at 0.61 accuracy; decisions between 1 and 30 s are 0.82 to 0.86 correct. the agent decides fast on clear drivers and waits on unclear ones, then still guesses. this is the right behaviour: the late, wrong decisions are drivers that are not separable in 60 s.
- 12% of decisions after 1 s are only 0.52 correct (still some early commits).
- PPO with metres stays below the index (0.659): with metres the jam and merge scenarios lose their main signal (headway), see section 3.

## 5. inverse RL: 0.73 per vehicle, while the pooled fit is right

### most seconds carry no information

the model learns from which acceleration a driver picks. share of observed decisions (`model/irl.py` `decisions`, test seeds 7 to 9) that fall on the "hold speed" choice:

| scenario | decisions per vehicle | a = 0 | abs(a) at most 0.5 | abs(a) at least 1.5 |
|---|---|---|---|---|
| highway low | 85 | 90% | 98% | 0.5% |
| highway medium | 94 | 74% | 91% | 4.3% |
| jam | 137 | 35% | 64% | 15.3% |

on the free highway 90% of seconds say "keep the speed". a driver reveals a desired headway only when the gap changes and the driver reacts, which is a few seconds per vehicle.

### the pull toward the population hides the differences

with so few informative decisions the per vehicle fit is pulled toward one population theta fitted on all types together (`scripts/irl_planted.py`, lam). result: per vehicle desired headway medians 2.51 (aggressive), 2.55 (normal), 2.62 s (conservative), interquartile ranges about 0.1 s. pooled by type the same model gives 1.63 / 2.43 / 2.94 s. the model is right; one vehicle does not carry enough data to show it.

### German motorways not identified

highD drivers barely accelerate (discomfort weight -41), so desired speed (127 to 256 km/h) and desired headway (beyond the 4 s cap) come out as extrapolations. same cause as above, stronger.

## what would move each number (not done, for discussion)

| number | main limit | what would raise it |
|---|---|---|
| UAH AUC 0.83, accuracy 77 to 81% | trip labels, 6 drivers, D1 sensor coverage | window or event labels; multiple instance learning (a trip is aggressive if some windows are); a "leader detected" flag; more labelled drivers (DriveDNA, 100-DrivingStyle) |
| GT vs agent 93.8% | not an accuracy | report against true labels only (done with planted drivers) |
| planted balanced 0.69 to 0.74 | speedDev overlap, close conservative and normal parameters | longer observation; a cleaner planted design (speedDev 0.05 or wider type gaps) if the goal is a method test; keep the overlap if the goal is realism |
| PPO 0.79 (UAH), 0.74 (planted) | same features as the index, wait cost, labels | richer observation (headway, leader flag, traffic state); compare against an optimal stopping rule (SPRT) |
| IRL 0.73 | 74 to 90% uninformative decisions | fit only on informative seconds (gap or leader speed changes); hierarchical prior per type instead of one population theta |
| German 51% | no labels, cut off from other roads | context score (done), a labelled motorway set |

## remaining gap of the best number per slide (19 oct)

- UAH whole trip AUC 0.944: drivers D2 to D6 order every aggressive trip above every normal trip (1.00); D1 reaches 0.667 because its aggressive secondary road trip (40.9) scores below two of its normal trips (45.5, 41.0). the whole missing 0.056 is that one trip (car ahead detected 10% of the time).
- planted, PPO headway 0.742: 66% of the remaining errors are normal and conservative confused. normal accuracy by setting: highway low 0.37, weather 0.34, jam 0.61, roundabout 0.72 (the following gap only shows while following).

## dqn, step 1 (19 oct)

`scripts/train_labeling_dqn_planted.py`: same environment, reward, split and evaluation as `scripts/train_labeling_rl_planted.py` (functions imported from it); only the learner changes (stable-baselines3 DQN, gamma 1, epsilon 1.0 to 0.05 over 30% of steps, buffer 200k, target update 2k).

first trial, headway features, 200k steps (34 s of training), seed 0: balanced acc 0.677 (conservative 0.715, normal 0.483, aggressive 0.834), 26.4 s to decision, 10% of decisions after 1 s. PPO with the same features: 0.742 after 1M steps. not comparable yet (5x fewer steps, no tuning). output `data/labeling_dqn_planted_summary_trial_headway_seed0.csv`.

- [x] script reusing the PPO setup
- [x] first trial run
- [ ] early commits: check per episode length during training
- [ ] 1M steps, 3 seeds, headway features
- [ ] PPO vs DQN vs formula on the same test vehicles
- [ ] UAH leave one driver out

## status

- [x] every headline number traced to its cause, with a measurement
- [ ] Dr. Daher: which limits to attack first (labels, observation length, planted design)
- [ ] verify whether SUMO IDM reads sigma

## correction (20 oct)

the supervised AUC above (0.762, "below the formula") was measured on raw scores pooled over environments; the agent has one threshold per environment, so its scores must be ranked by the margin (score minus threshold). measured that way the same fold models reach 0.897, above the formula (0.829). details: `fall_2026_10_20_revision.md` section 3.
