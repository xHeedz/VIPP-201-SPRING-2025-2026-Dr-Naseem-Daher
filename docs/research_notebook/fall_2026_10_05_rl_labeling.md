# mon 5 oct 2026: index side as RL, sequential labeling (option a)

handoff: `docs/rl_pivot.md`. option a chosen by Hadi; `agent/driver_q_learner.py` renamed to `agent/score_bucket_labeler.py` after the results.

## audit, confirmed in code

- `model/dynamic_weight_agent.py`: lines 37 to 42 are the whole model (3 x 4 raw weights, softmax per environment, one threshold per environment, shared `log_tau`; no MLP). `fit_labels` (line 63) is balanced binary cross entropy with Adam (line 87). no reward, no sampled action, no KL term, no annealing.
- `agent/driver_q_learner.py`: state = score bucket (line 23), updated action = the correct label (line 24), reward always 15 (line 31), only that cell updated (line 32), the max in line 32 reads the same state (no next state). counts the most frequent label per bucket. circular: `Category` is a threshold on `AI_Score`.
- no `log_prob`, `REINFORCE`, `KL` or curiosity term in `model/`, `agent/`, `experiments/weight_agent/`.

## formulation

`env/labeling_env.py`, `LabelingEnv` (gymnasium):
- episode: one UAH trip from a random start (class drawn first, balanced), one step per second, stopped seconds (< 5 km/h) dropped
- actions: wait, label normal, label aggressive (UAH has no conservative class; the class list is a parameter)
- reward: +1 correct, -1 wrong (per class cost optional), -wait_cost per wait, waiting at the horizon (120 s) or the end of the drive counts as wrong
- observation (13): running mean of the 4 index features, mean of the last 10 s, reference AI of both / 100, elapsed / horizon, environment one hot
- tests: `tests/test_labeling_env.py` (6, hand values: reference AI 0.46, running and recent means, reward sequence, timeout -1.04)

## experiment

`scripts/train_labeling_rl.py`: PPO (stable-baselines3, 300k steps, ent_coef 0.01), leave one driver out over the 6 UAH drivers (28 trips, 11 aggressive, 21,005 s). test episodes: every held-out trip, start every 30 s with a full horizon left (603 episodes). baselines on the same episodes: reference index after fixed T (AI >= 42), DynamicWeightAgent of the same fold after fixed T, two-threshold stopping rule on the running AI (lo, hi fitted on the training drivers, same reward). outputs `data/labeling_rl_{lodo,summary}_wait<cost>.csv`.

mean over the 6 held-out drivers:

| wait cost | method | balanced acc | acc normal | acc aggressive | seconds | reward |
|---|---|---|---|---|---|---|
| 0.001 | PPO | 0.773 | 0.899 | 0.648 | 8.6 | 0.611 |
| 0.001 | weight agent T30 | 0.827 | 0.816 | 0.837 | 30 | 0.623 |
| 0.001 | stopping rule | 0.826 | 0.872 | 0.779 | 101.4 | 0.576 |
| 0.002 | PPO | 0.776 | 0.899 | 0.654 | 9.1 | 0.605 |
| 0.002 | weight agent T30 | 0.827 | 0.816 | 0.837 | 30 | 0.594 |
| 0.002 | stopping rule | 0.777 | 0.840 | 0.714 | 33.4 | 0.523 |
| 0.005 | PPO | 0.779 | 0.889 | 0.670 | 5.5 | 0.598 |
| 0.005 | weight agent T10 | 0.804 | 0.798 | 0.809 | 10 | 0.564 |
| 0.005 | stopping rule | 0.699 | 0.726 | 0.673 | 12.3 | 0.367 |

fixed T, independent of wait cost (balanced acc): reference 0.749 / 0.772 / 0.786 / 0.826 at 10 / 30 / 60 / 120 s; weight agent 0.804 / 0.827 / 0.831 / 0.861.

first run with wait cost 0.01: waiting 120 s costs 1.19 reward, more than the accuracy gain (about +0.08 balanced acc from 10 to 120 s, worth about +0.16); PPO and the stopping rule both decide at second 1. the wait cost has to sit near 0.001 to 0.005 for the trade off to exist on UAH.

## reading

- PPO has the highest reward at wait cost 0.002 and 0.005, by 0.011 and 0.034; the spread across drivers is about 0.12. not a significant win.
- PPO is less accurate than the DynamicWeightAgent (0.78 vs 0.80 at 10 s, 0.83 at 30 s); its reward comes from deciding in 5 to 9 s.
- PPO is biased toward normal (0.89 to 0.90 vs 0.65 to 0.67): about a third of aggressive episodes are missed.
- PPO does not learn to wait long even when waiting is cheap (0.001: 9 s, while the stopping rule waits 100 s for 0.83).
- likely causes: 22 training trips per fold (few independent drivers), trip level labels (finding 6: calm cruising inside an aggressive trip is labelled aggressive, so waiting adds little), observation built on the hand-set features while the weight agent uses learned weights.

## follow up: missed aggressive cost and learned index in the observation

`--miss-aggressive-cost 2` (wrong label on an aggressive driver costs 2, baselines scored with the same costs) and `--agent-obs` (observation adds (learned score - learned threshold) / 100 of the running and recent means, from the fold's DynamicWeightAgent, never the held-out driver). wait cost 0.002, PPO, mean over the 6 held-out drivers:

| variant | balanced acc | acc normal | acc aggressive | seconds |
|---|---|---|---|---|
| as before | 0.776 | 0.899 | 0.654 | 9.1 |
| learned index in obs | 0.780 | 0.867 | 0.694 | 8.9 |
| missed aggressive costs 2 | 0.743 | 0.794 | 0.691 | 17.1 |
| both | 0.792 | 0.836 | 0.747 | 12.4 |

reference: DynamicWeightAgent after 10 s 0.804 (0.798 / 0.809), after 30 s 0.827 (0.816 / 0.837). with both changes PPO misses 25% of aggressive episodes instead of 35% and is the best RL variant, still 0.012 below the weight agent at 10 s and 0.035 below it at 30 s. balanced reward with miss cost 2: PPO 0.434, weight agent at 30 s 0.515. outputs `data/labeling_rl_*_wait0.002{,_miss2}{,_agentobs}.csv`.

## status

- [x] audit confirmed in code
- [x] option a chosen (Hadi)
- [x] `env/labeling_env.py`, tests
- [x] PPO with leave one driver out on UAH, compared with fixed T and stopping rule baselines
- [x] asymmetric cost (missed aggressive driver costs 2)
- [x] weight agent score in the observation (best together: 0.792, still below the weight agent)
- [ ] planted SUMO drivers as episodes (per vehicle labels, three classes, many more episodes)
- [x] `agent/driver_q_learner.py` renamed to `agent/score_bucket_labeler.py`, docstring says what it does
- [ ] confirm the RL direction with Dr. Daher
