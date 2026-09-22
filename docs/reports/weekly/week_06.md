# **Weekly Report — Week 6**

**Period: **Tuesday Feb 24 – Monday Mar 2, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Implement a count-based exploration bonus on the existing DriverQLearner. Discretize 4D normalized telemetry into a 10⁴ grid, maintain visitation counts N(s), add β/sqrt(N(s)+1) to the reward. Sweep β over {0.01, 0.05, 0.1, 0.5, 1.0}. Pass criteria: re-trained agent's state-visitation heatmap is visibly more uniform; Bellman errors in the high-prox column drop by ≥30%; per-class accuracy on a held-out perturbed dataset improves by ≥10 percentage points on the under-represented classes.

## **Work completed**

Wrote agent/curiosity.py implementing CountBasedExplorer: discretizes the state, maintains counts in a default-dict, returns β / sqrt(count) on each query. Wired into the Q-learner so total_reward = env_reward + bonus(state).

Ran the β-sweep over 1500 episodes per condition. β = 0.01 had little effect. β = 0.05 reduced the easy-corner visitation from 70% to 55%. β = 0.1 produced ~40% concentration in the easy corner — a clean uniform-ish spread. β = 0.5 was over-exploration: the agent thrashes through bad noisy states for the bonus and converges worse than the no-curiosity baseline. β = 1.0 fails to converge.

Locked β = 0.1 as the operating point and re-ran the diagnostic suite.

Verified the curiosity bonus is transient: bonus dominates env_reward for the first ~30 episodes (~0.2 vs ~0.05), then env_reward dominates from episode ~80 onward (env ~0.07, bonus ~0.005). By the end of training the bonus is essentially negligible. Evaluation comparisons (where the bonus is removed) are therefore fair.

## **Key results**

Bellman-error reduction: high-prox column mean |δ| dropped from 0.18 to 0.09, a 50% reduction (criterion was ≥30%). The right tail of the histogram shrank visibly — ~95% of |δ| values now under 0.03, with the tail going out to 0.4 instead of 0.6.

Per-class accuracy improvements on the held-out perturbed dataset: Conservative 67% → 89% (+22), Normal 81% → 86% (+5), Aggressive 95% → 94% (essentially unchanged, already saturated). Average over the under-represented classes: 74% → 87.5% (+13.5).

State-visitation distribution noticeably more uniform across the (n_speed, n_prox) marginal.

## **Plan for next week**

Long-horizon stability check (5000 episodes with curiosity, weights tracked over time, drift bounded against targets). In parallel: install SUMO on the lab machine, get hello-world running, write a thin Python wrapper around the TraCI API exposing the same step / reset / get-vehicles interface as the highway-env wrapper.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
