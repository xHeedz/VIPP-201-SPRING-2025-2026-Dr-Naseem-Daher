# **Week 6: Tuesday Feb 24 – Monday Mar 2, 2026**

**Theme — ***Count-based curiosity bonus on the Q-learner. The fix actually works.*

## **Goals for the week**

Implement a count-based exploration bonus on the existing DriverQLearner. Discretize 4D normalized telemetry into a 10⁴ grid, maintain visitation counts N(s), add β/sqrt(N(s)+1) to the reward. Sweep β in {0.01, 0.05, 0.1, 0.5, 1.0}. Pass: re-trained agent's state-visitation heatmap is visibly more uniform across the (n_speed, n_prox) marginal; Bellman errors in the high-prox column drop by at least 30%; per-class accuracy on a held-out perturbed dataset improves by at least 10 percentage points on the under-represented classes.

## **Daily entries**

**Tue Feb 24***  —  Curiosity implementation.*

Wrote agent/curiosity.py — a small CountBasedExplorer class. Constructor takes the number of bins per dim (10 by default) and the dim count (4). Maintains a default-dict from discretized-state-tuple to int. On bonus(state) it discretizes, increments the count, returns β / sqrt(count). On save() it writes to a pickle. About 40 lines. Wired into the Q-learner's reward path: total_reward = env_reward + bonus(state).

Quick sanity: ran with β=0.1 for 200 episodes. The first few episodes get huge bonuses (everything is novel), but by episode 50 the bonuses settle into the 0.001-0.05 range, which is the same order as the env reward (typically 0.01-0.1 on highway-env). So β=0.1 gives roughly equal weight to env reward and exploration bonus on average. Reasonable starting point.

**Wed Feb 25***  —  β-sweep, run 1.*

Started the sweep over β ∈ {0.01, 0.05, 0.1, 0.5, 1.0}. Each run is 1500 episodes, so total is ~7500 episodes. Took ~5 hours total on the lab CPU (I batched it overnight). Re-ran the state-visitation heatmap on each.

β=0.01: barely changes anything from the no-curiosity baseline. β=0.05: visitation distribution starts to spread out, the (n_speed, n_prox) marginal is now ~55% concentrated in the easy corner instead of 70%. β=0.1: nice spread, ~40% in the easy corner. β=0.5: actually too uniform — the agent is exploring noisy bad states for the bonus and not converging on a good policy (test-time AI score worse than no-curiosity baseline). β=1.0: total exploration thrash, training never converges.

β=0.1 is the sweet spot. Going to lock it in for the rest of the term.

**Thu Feb 26***  —  Bellman-error re-measurement + per-class accuracy.*

Re-ran the Bellman-error diagnostic on the β=0.1 agent. The high-prox column's mean |δ| dropped from 0.18 to 0.09 — 50% reduction, which exceeds the 30% pass criterion. The right tail of the histogram shrinks visibly; now ~95% of |δ| < 0.03 and the long tail goes out to 0.4 instead of 0.6.

Per-class accuracy on the held-out perturbed dataset (computed by perturbing the Phase-1 CSV with random context noise and re-bucketing): Conservative class went from 67% to 89%, Normal from 81% to 86%, Aggressive from 95% to 94% (basically unchanged — already saturated). Conservative + Normal averaged ~74% before, ~87% after. So the under-represented classes pulled up significantly without sacrificing the dominant class. This is the result.

Made an updated state-visitation figure. Side-by-side: "before curiosity" and "after curiosity, β=0.1". Striking visually. Saved both.

**Fri Feb 27***  —  Sanity sweep on environment reward.*

Daher pre-emptively asked on Monday whether the curiosity bonus is artificially inflating the agent's effective reward and therefore making policy comparisons unfair. Wrote a small check: for the β=0.1 agent, plot env_reward and bonus separately over the course of training. Bonus dominates for the first ~30 episodes (~0.2 vs ~0.05 env), then env_reward dominates from episode ~80 onward (env ~0.07, bonus ~0.005). By the end of training, bonus is essentially negligible — curiosity is a transient driver, not a permanent reward distortion. So the comparisons at evaluation time (where we don't add the bonus) are fair. Documented this in the diagnostic memo.

**Sat Feb 28***  —  Off.*

Off. Worked on a problem set for another class.

**Sun Mar 1***  —  Off.*

Off.

**Mon Mar 2***  —  Sync.*

Sync. Walked through the β-sweep, the Bellman-error reduction, and the per-class accuracy improvement. Daher was visibly pleased — said this is the kind of result he'd wanted to see in the midterm. Asked one new question: does the curiosity-augmented agent generalize to states outside the training distribution? Specifically, if I run the trained agent on a longer episode (5x the training horizon) does it stay in well-visited states or does it drift into the under-explored corners and produce wild predictions? Recorded as a question for week 7. Scope for week 7: long-horizon stability check (5000-episode training, weights logged over time, drift bounded against targets); in parallel, start the SUMO setup.

## **Reflection**

Curiosity is one of those ideas that's clean in theory and risky in practice — there are a lot of moving parts (β, the discretization, the interaction with env reward). The fact that we got it to work in two days without much pain is partly luck, partly the unit-test discipline I started in week 5. The infrastructure is paying for itself in compounding ways now.

## **Plan for next week**

Long-horizon stability check (5000 episodes with curiosity, weights tracked over time, drift bounded against targets). In parallel: install SUMO on the lab machine, get hello-world running, read enough of the TraCI Python API to know what to call.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
