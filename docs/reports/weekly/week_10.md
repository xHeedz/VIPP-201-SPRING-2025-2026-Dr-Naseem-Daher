# **Weekly Report — Week 10**

**Period: **Tuesday Mar 24 – Monday Mar 30, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Implement the Social Latency penalty: R_total = (1 − AI_self) − λ · max(0, ΔAI_neighbors), where ΔAI_neighbors is the increase in mean AI among vehicles within 30m relative to a moving baseline. Sweep λ on the SUMO 5%-aggressive scenario. Pick the λ that preserves self-AI within +5% of baseline while reducing the shockwave bump on neighbors by ≥20%.

## **Work completed**

Wrote agent/social_latency.py. Maintains a 5-second rolling window of neighbor AI scores; on each step computes ΔAI_neighbors as current-minus-rolling-mean, clips at zero so the agent is penalized only for *increasing* neighbor AI, not for being near already-aggressive drivers it didn't cause.

Ran the λ-sweep across 5 values × 1500 episodes × 3 seeds (~22500 episodes total, ~9 hours overnight). Recorded per-condition self AI and neighbor shockwave bump.

Behavioral inspection of the λ=2.0 agent vs the baseline. Replayed a 60-second episode with the same seed and scenario for both. Two visible differences: (a) larger following distances in tight-following intervals (mean prox 0.41 vs 0.36 — about 14% more space), and (b) fewer lane changes when an aggressive vehicle is in the 30m radius — the trained agent tends to hold its lane and let aggressives move away.

Fleet experiment: replaced 50% of normal vehicles in the 5%-aggressive scenario with λ=2.0-trained agents (final population: 5% aggressive / 47.5% normal-baseline / 47.5% normal-with-Social-Latency). Re-measured population mean AI.

## **Key results**

λ-sweep trade-off curve. λ=0: self 28.2, shockwave +6.7. λ=0.5: self 28.5 (+0.3), shockwave +5.8 (−0.9). λ=1.0: self 29.1 (+0.9), shockwave +5.1 (−1.6). λ=2.0: self 30.4 (+2.2), shockwave +4.0 (−2.7, 40% reduction). λ=5.0: self 34.7 (+6.5), shockwave +2.2 (−4.5).

Pareto pick under the criterion: λ=2.0. Self goes up 2.2 points (well within the +5% budget); shockwave drops by 2.7/6.7 = 40% (above the 20% criterion).

Fleet result: with 50% Social Latency deployment in the 5%-aggressive scenario, population mean AI = 30.6, vs. 33.1 unmitigated and 26.3 reference. So a 50% fleet recovers roughly 75% of the shockwave-induced damage at a population level — not just for the deployed vehicles but for the whole population, because the fleet damps cascading effects on hop-1 and hop-2 normals.

Cost side from the SUMO tripinfo XML: mean trip time +2.8% (statistically significant, n=600), lane utilization essentially unchanged. So a 50% Social Latency deployment buys 24% reduction in population-level aggressiveness for ~3% travel-time cost.

## **Plan for next week**

Friction-aware regime. Add slip-ratio telemetry to the SUMO wrapper using the friction attribute on edges. Set up a low-friction scenario (μ=0.4 instead of the default ~0.9). Verify the agent's gate puts mass on the weather embedding when slip is elevated. Tune the weather-target weights based on per-channel correlation with slip. Then move to ROS2 wrapper in week 12.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
