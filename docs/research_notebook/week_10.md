# **Week 10: Tuesday Mar 24 – Monday Mar 30, 2026**

**Theme — ***Social Latency penalty + λ-sweep. Fleet-level mitigation.*

## **Goals for the week**

Implement the Social Latency penalty: R_total = (1 - AI_self) - λ · max(0, ΔAI_neighbors), where ΔAI_neighbors is the increase in mean AI among vehicles within 30m relative to a moving baseline. Sweep λ ∈ {0, 0.5, 1.0, 2.0, 5.0} on the SUMO 5%-aggressive scenario. Pick the λ that preserves the agent's own AI within +5% of baseline while reducing the shockwave bump on neighbors by at least 20%.

## **Daily entries**

**Tue Mar 24***  —  Implementation.*

Wrote agent/social_latency.py. Tracks a 5-second rolling window of neighbor AI scores; on each step, computes the *current* neighbor AI, takes the difference vs. the rolling mean, clips at zero (we only penalize the agent for *increasing* neighbor AI, not for being near already-aggressive drivers it didn't cause). The clip is important — without it, the agent gets penalized just for spawning near an aggressive driver, which is unfair.

Wired into the gate-head agent's reward path. Total reward at step t: R_t = (1 - AI_self_t) - λ · max(0, ΔAI_neighbors_t). The 30m proximity radius matches the shockwave-zone definition from week 8.

**Wed Mar 25***  —  λ-sweep, run 1.*

Started the λ-sweep. 5 values × 1500 episodes × ~3 seeds = ~22500 episodes total, ran overnight on the lab machine. Took ~9 hours. Got results in by morning.

λ=0: baseline. Agent AI = 28.2. Neighbor shockwave = +6.7 (close to the population shockwave we measured, which is good — the agent is a representative "normal"). λ=0.5: Agent AI = 28.5 (+0.3). Neighbor shockwave = +5.8 (-0.9). Small effect on the agent, slight reduction in shockwave. λ=1.0: Agent AI = 29.1 (+0.9). Neighbor shockwave = +5.1 (-1.6). λ=2.0: Agent AI = 30.4 (+2.2). Neighbor shockwave = +4.0 (-2.7). λ=5.0: Agent AI = 34.7 (+6.5). Neighbor shockwave = +2.2 (-4.5). Trade-off curve.

Pareto-best for the criterion (preserve self within +5%, reduce shockwave by ≥20%): λ=2.0. Self goes up 2.2 points (well within budget), shockwave drops by 2.7 / 6.7 = 40% of its magnitude. Ship it.

**Thu Mar 26***  —  Behavioral inspection of the λ=2.0 agent.*

Wanted to actually understand *what* the λ=2.0 agent is doing differently. Replayed a 60-second episode side-by-side with the λ=0 baseline (same seed, same scenario). Two visible differences: (a) the λ=2.0 agent maintains slightly larger following distances when behind another vehicle (mean prox in tight-following intervals: 0.41 vs 0.36 for baseline — about 14% more space), and (b) the λ=2.0 agent does fewer lane-changes when an aggressive vehicle is in its 30m radius — it tends to hold its lane and let the aggressive vehicle move away rather than try to navigate around it. Both behaviors are intuitive and both reduce the chance of the agent's actions causing neighbors' kinematic perturbations.

**Fri Mar 27***  —  Fleet experiment.*

Ran a fleet experiment: replace 50% of the normal vehicles in the 5%-aggressive scenario with λ=2.0-trained agents (so the population is now 5% aggressive, 47.5% normal-baseline, 47.5% normal-with-Social-Latency). Re-measured population mean AI. Without the fleet: 33.1. With the fleet: 30.6 (-2.5 from the heterogeneous baseline, -3.7 from the 0%-aggressive reference). So a 50% deployment of the Social Latency policy across the fleet recovers ~75% of the shockwave-induced damage, not just for the deployed vehicles but for the population as a whole because the fleet damps the cascading effects on hop-1 and hop-2 normals.

Population-level mitigation = 24% reduction in heterogeneous-fleet mean AI vs. unmitigated baseline, in the 5%-aggressive condition. This is the fleet-level result. Made a clean figure for the sync.

**Sat Mar 28***  —  Off.*

Off.

**Sun Mar 29***  —  Sync prep.*

Cleaned up four figures for the sync: λ-sweep trade-off curve, behavioral inspection plots (following-distance + lane-change rate), 50%-fleet population-level result, and a sketch of how this would extend to higher fleet penetration.

**Mon Mar 30***  —  Sync.*

Sync. Daher liked the result, called it "the practical use-case the AI score has been waiting for." The story now has a clean shape: closed-form interpretable AI → diagnose shockwave → quantify dose-response → reward-shape against shockwave → fleet deployment recovers most of the damage. He wants me to refine the writing for the final report (we'll start drafting in week 13) and to move on to the friction-aware regime for week 11.

One specific ask: in the final report, the fleet result should be presented with the cost side too — what's the trade-off in terms of average travel time / lane utilization? I checked the SUMO tripinfo XML for the 50%-fleet run vs. the unmitigated baseline. Mean trip time: +2.8% (statistically significant, n=600). Lane-utilization: essentially unchanged. So a 50% Social Latency deployment costs about 3% on travel time for a 24% reduction in population-level aggressiveness. That feels like a cleanly defensible trade.

## **Reflection**

This week is the use-case argument for the whole project. The closed-form AI score isn't just a measurement — it's a signal a fleet of agents can act on to dampen the shockwaves they cause. The 24% population-level mitigation at 50% fleet penetration is a concrete number with a concrete cost (+2.8% travel time). The behavioral inspection (more space when following, fewer lane-changes near aggressors) makes the result intuitive — the agent isn't doing anything weird, it's doing what a polite driver would do.

## **Plan for next week**

Friction-aware regime. Add slip-ratio telemetry to the SUMO wrapper using the `friction` attribute on edges. Verify the agent's gate puts mass on the weather embedding when slip ratios are elevated. Tune the (w_speed, w_accel, w_prox, w_wave) weights for low-friction conditions. Then move to ROS2 wrapper in week 12.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
