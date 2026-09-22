# **Weekly Report — Week 9**

**Period: **Tuesday Mar 17 – Monday Mar 23, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Run the two-hop propagation experiment and the aggressive-fraction sweep. Make the final go/no-go decision on CARLA based on install footprint vs. expected term-2 payoff.

## **Work completed**

Built a proximity graph from the heterogeneous-driver run. Nodes are vehicles, edges connect any pair that came within 30m at any point. Aggressive vehicles are hop-0; their direct neighbors are hop-1; normals connected to hop-1 normals (but not to any hop-0) are hop-2; further out is hop-3+.

Ran the aggressive-fraction sweep: 0%, 5%, 10%, 15%, 20% with 3 random seeds per condition for variance estimation (15 runs total, ~62 minutes of compute).

CARLA decision: spent 2 hours doing the honest accounting on the trade-off. Pros: real urban telemetry with occlusion, more realistic perception, the simulator the AD industry uses. Cons: ~30 GB install, dedicated GPU needed, heavier Python API, no clear path to use CARLA scenarios for the population-level reward-shaping experiments planned for weeks 10–11. Trade-off summary: 2 weeks on CARLA costs us the multi-objective reward shaping, friction-aware experiments, and likely the ROS2 wrapper.

Decision: descope CARLA to a stretch goal that probably will not land this term. Documented the trade-off as a deliberate decision rather than a failure, for the final report's limitations section.

## **Key results**

Two-hop propagation: hop-1 mean AI = 38.7 (+12.4 vs reference, replicating week 8), hop-2 = 28.2 (+1.9), hop-3+ = 26.5 (+0.2). Two-hop effect is small (~15% of direct-proximity magnitude) and the effect is essentially gone by three hops out.

Dose-response sweep, mean across seeds: 0% → 26.3 (sd 0.4); 5% → 33.1 (sd 0.6, +6.8); 10% → 35.7 (sd 0.8, +9.4); 15% → 37.5 (sd 0.9, +11.2); 20% → 38.3 (sd 1.1, +12.0). Curve is concave with diminishing marginal effect: +6.8 from 0→5%, +2.6 from 5→10%, +1.8 from 10→15%, +0.8 from 15→20%.

Saturating-exponential fit AI(f) = 26.3 + 12.5·(1 − exp(−f/0.06)) gives R² = 0.998 over the 5 conditions. Saturation constant 0.06 means the curve reaches 63% of its asymptote at f=6% and 95% at f=18%. Practical implication: small fractions of aggressive drivers cause most of the population-level damage.

## **Plan for next week**

Multi-objective reward: R = (1 − AI_self) − λ · max(0, ΔAI_neighbors). Sweep λ ∈ {0, 0.5, 1.0, 2.0, 5.0}. Train the gate-head agent for 1500 episodes per condition in the SUMO 5%-aggressive scenario; evaluate on a held-out scenario. Pick the λ that preserves the agent's own AI within +5% of baseline while reducing the neighboring-AI shockwave bump by at least 20%.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
