# **Week 9: Tuesday Mar 17 – Monday Mar 23, 2026**

**Theme — ***Two-hop propagation + dose-response sweep. Also: CARLA gets descoped, and that's the right call.*

## **Goals for the week**

Two follow-ups from last week's sync. (1) Two-hop propagation: identify normal vehicles that never came into proximity with an aggressive driver but did come into proximity with a normal that did. Compare their mean AI to (a) the reference normals and (b) the directly-affected normals. (2) Dose-response sweep on aggressive fraction: 0%, 5%, 10%, 15%, 20%, 5 minutes per condition, plot population-level mean AI. Stretch: identify a critical fraction if one exists. Also, separately: make a final go/no-go call on CARLA — install footprint vs. expected term-2 payoff.

## **Daily entries**

**Tue Mar 17***  —  Two-hop.*

Two-hop graph. Built a proximity graph for the 5%-aggressive scenario from last week: one node per vehicle, an edge between any two vehicles that came within 30m at any point. Marked the aggressive vehicles as "hop-0". Their direct neighbors (ever within 30m) are "hop-1" — these are the directly-affected normals from last week. Normals connected to hop-1 normals (but not to any hop-0) are "hop-2". Normals further out are "hop-3+".

Mean AI by hop level, with reference normal mean = 26.3 and standard deviation = 6.7: hop-1 = 38.7 (+12.4), hop-2 = 28.2 (+1.9), hop-3+ = 26.5 (+0.2). So two-hop propagation does exist but it's small — about 15% of the magnitude of the direct-proximity bump. By three hops out, the effect is essentially gone. This is consistent with a fast-decaying shockwave. Visually, the histogram for hop-2 overlaps the reference distribution heavily but has a slightly thicker right tail.

**Wed Mar 18***  —  Dose-response sweep, run 1.*

Started the dose-response sweep. 5 conditions (0%, 5%, 10%, 15%, 20%), each 5 minutes of sim time, plus 3 random seeds per condition for variance estimation. 15 runs total. Each runs in real-time at about 0.6× wall-clock so the whole sweep is ~62 minutes of compute, but I'm doing other work during it.

Initial results, mean across seeds for population-level mean AI: 0% → 26.3 (sd 0.4). 5% → 33.1 (+6.8, sd 0.6). 10% → 35.7 (+9.4, sd 0.8). 15% → 37.5 (+11.2, sd 0.9). 20% → 38.3 (+12.0, sd 1.1). So the curve flattens — the marginal effect of going from 0→5% is +6.8, from 5→10% is +2.6, from 10→15% is +1.8, from 15→20% is +0.8. Concave dose-response. No sharp tipping point in this range; just diminishing returns.

The interpretation is interesting: a few aggressive drivers in a fleet do most of the damage, and adding more doesn't proportionally amplify it. Consistent with the spatial-decay finding from week 8 — once you're already affecting your local neighborhood, adding more aggressors mostly affects already-affected neighborhoods rather than reaching new vehicles.

**Thu Mar 19***  —  CARLA: a hard look at the trade-off.*

Spent 2 hours doing the honest accounting on CARLA. The pros: real urban telemetry with actual occlusion, much more realistic perception challenges, the simulator the AD industry actually uses. The cons, and they are substantial: ~30 GB install footprint, requires a dedicated GPU which the lab machine doesn't have available consistently, the Python API is heavier than TraCI and has its own learning curve, and (critically) there's no clear path to use CARLA scenarios for the reward-shaping experiments I want to run in weeks 10-11 because CARLA's traffic management is designed for ego-vehicle scenarios, not population-level sweeps like the dose-response curve.

The realistic version of the term plan: if I spend 2 weeks on CARLA, I lose the multi-objective reward shaping (Social Latency penalty, λ-tuning), the friction-aware regime experiments, and probably the ROS2 wrapper. CARLA is one piece in exchange for three. Bad trade.

Decision: descope CARLA to a stretch goal that probably won't land. Wrote a one-paragraph memo on the trade-off so when I write the final report, the descoping is documented as a deliberate decision rather than a failure. Will discuss with Daher Monday.

**Fri Mar 20***  —  Dose-response figure + write-up.*

Made the dose-response figure. Aggressive fraction on x-axis, population-level mean AI on y, error bars from the 3-seed variance, individual seed dots overlaid. Curve is visibly concave — fits a saturating exponential nicely (AI(f) ≈ 26.3 + 12.5·(1 - exp(-f/0.06))) with R² = 0.998 over the 5 points. The 0.06 saturation constant means the curve gets to ~63% of its asymptote by f=6%, and ~95% by f=18%. So in policy-relevance terms: even a small fraction of aggressive drivers (single-digit percentages) does most of the population-level damage.

**Sat Mar 21***  —  Off.*

Off — went to a wedding.

**Sun Mar 22***  —  Light prep for sync.*

Cleaned up the figures and wrote a 2-page summary of weeks 8-9 for the sync. Three figures: spatial-bin shockwave from week 8, two-hop bar chart, dose-response curve with the saturating-exponential fit overlaid. Plus the CARLA descoping memo as an appendix.

**Mon Mar 23***  —  Sync.*

Sync. Daher liked the dose-response curve a lot — said the saturating-exponential fit is a clean way to summarize the result. He agreed with the CARLA descoping after about 5 minutes of pushback, in which he pointed out that we should at least keep CARLA in the future-work section of the final report so reviewers don't ask why we didn't do it. Fair. Recorded.

Next-step ask: now that we've quantified the shockwave, design a reward-shaping experiment to mitigate it. The idea is to add a Social Latency penalty to the agent's reward that penalizes the agent if its actions cause neighboring vehicles' AI scores to spike. Sweep the penalty weight λ. Show that for some λ the agent's own performance is preserved while neighboring AI is reduced. This is the multi-objective reward shaping from the future-work section.

## **Reflection**

Two-hop shows the shockwave decays fast, dose-response shows it saturates fast. Both findings tell the same story: aggressive driving has strong local effects with diminishing marginal cost as more aggressors are added. The CARLA descoping is, honestly, a relief. Decisions made in week 9 are easier to defend than decisions made in panic at the end of the term.

## **Plan for next week**

Multi-objective reward: R = (1 - AI_self) - λ · max(0, ΔAI_neighbors). Sweep λ ∈ {0, 0.5, 1.0, 2.0, 5.0}. For each, train the gate-head agent for 1500 episodes in the SUMO 5%-aggressive scenario, then evaluate on a held-out scenario. Pick the λ that preserves the agent's own AI score within +5% of baseline while reducing neighboring AI by at least 20% of the shockwave bump.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
