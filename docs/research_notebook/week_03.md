# **Week 3: Tuesday Feb 3 – Monday Feb 9, 2026**

**Theme — ***Wire the agent into the sim loop. Bug hunt. Add jerk to telemetry.*

## **Goals for the week**

Plug the DynamicWeightAgent into the highway-env step loop so each step produces predicted weights → AI score, log everything to a per-episode CSV, plot AI vs. fixed-weight baseline on a controlled scenario. Add a jerk channel to the telemetry — Daher's been asking for it since last term and it should be a 30-minute change.

## **Daily entries**

**Tue Feb 3***  —  Wiring.*

Started wiring agent into the highway-env step loop. The complication is that the agent expects an 8-dim input but highway-env hands us its native obs (a (V, 5) array of vehicle features). So we need a small adapter that pulls the ego vehicle's row, computes (n_speed, n_accel, n_prox, n_wave) using the same normalization as the AggressivenessModel, attaches the current context (we are on highway, so context = [1, 0, 0, density_norm, 0]), and hands the 8-dim vector to the agent. The wave term is the one with state — we accumulate steering/heading deltas over a short window — so I had to add a small ring buffer to the adapter. Did all this in env/highway_adapter.py. About 80 lines.

Got an end-to-end run working by EOD. Episode of 200 steps on highway-env, AI score logged at every step. Mean AI = 19.7 with predicted weights, vs. 22.4 with fixed weights. Lower is better here, so the agent is — at least nominally — choosing weights that pull the AI lower than the fixed baseline. Could be that the agent is learning to game the score. Worth investigating.

**Wed Feb 4***  —  BatchNorm bug — mostly self-inflicted.*

Wasted half the day on a bug. The wired-in agent was producing identical predicted weights every step regardless of state. Spent like an hour and a half thinking the input adapter was broken. It wasn't. The bug was that I'd switched on BatchNorm in the agent at some point on a branch I forgot about, and during inference at batch size 1, BN's running statistics make every input look the same once the mean/var saturate. Removed BN, predicted weights now vary cleanly across the episode.

This is the second time in the project I've gotten bitten by BN at inference time. Adding `assert not any(isinstance(m, nn.BatchNorm1d) for m in self.modules())` to the agent's __init__ as a guard. Petty but I don't want to lose a third afternoon to it.

**Thu Feb 5***  —  Jerk + figure.*

Added a jerk channel (numerical derivative of acceleration over a 3-step window) to the highway adapter. Five lines of code. Confirmed it produces sensible values on a controlled lane-change scenario — peaks at ~3.2 m/s³ during the change, near zero during steady cruise. Updated the AI formula's input tuple to be (n_speed, n_accel, n_prox, n_wave, n_jerk) — five terms now. The agent's input dim grows from 8 to 9. Retrained briefly to absorb the change, took 4 min.

Made a figure for the Monday sync: AI score vs. step index for both the fixed-weight baseline and the predicted-weight agent on the same controlled lane-change scenario. The two lines are nearly identical except in the lane-change region, where the predicted-weight version drops about 4 points lower because the agent reduces w_wave on impact. Visually persuasive. Saved as eval/lane_change_compare.png.

**Fri Feb 6***  —  Per-context profiling.*

Profiled the predicted weights across 50 random episodes per context (highway-only seeds). Mean predicted weights on highway: (0.31, 0.09, 0.44, 0.16). Hand-picked highway target: (0.30, 0.10, 0.45, 0.15). Within 0.02 L1, which is the noise floor of the run. Good.

Then I tried switching the context input to urban while keeping highway-env as the actual sim — basically asking the agent "if this were urban, what would the weights be?" Mean predicted weights flip to (0.20, 0.15, 0.35, 0.30) — higher on proximity, much lower on speed. That's exactly the urban target. So the agent is responding to the context input even when the underlying physics aren't matching. Good. Also a little dangerous, because nothing prevents us from feeding it nonsense context. Adding context-validation to the adapter as a TODO.

**Sat Feb 7***  —  Off.*

Power cuts most of the day, did some reading at the apartment but no real work. Helped my brother fix something on his laptop.

**Sun Feb 8***  —  Quick gradient-norm instrumentation.*

Did the gradient-norm thing Daher asked for last week. Hooked into the training loop, recorded total grad norm + per-layer norm every 10 epochs, plotted. Norms are healthy: total grad norm sits around 0.4 at the start, decays cleanly to about 0.05 by epoch 200, no spikes, no flat zones. No layer is starving. Saved the plot.

**Mon Feb 9***  —  Sync + scope for week 4.*

Sync. Walked Daher through the wiring, the BN bug story (he laughed), the lane-change figure, and the per-context profiling. He's happy with the wiring and wants me to start building the gating layer next. Scope for week 4: implement the gate as a 3-regime softmax attention over learnable regime embeddings, train it against the same KL loss as before, and demonstrate that it produces a sensible smooth transition when we feed it a gradually changing context vector (e.g. linearly interpolating from highway to urban over 30 simulated seconds).

## **Reflection**

The wiring landed. The BN bug cost me half a day but the rest of the week was clean. The fact that the agent responds correctly to a synthetic urban context while the underlying sim is highway is the most useful diagnostic I've gotten so far — it confirms that the supervised signal really is binding the predicted weights to the context vector, not to the physics.

## **Plan for next week**

Build the gating network. Three learnable regime embeddings, a small MLP gate that maps context → 3-way mixture weights, the predicted weight vector is the convex combination. Boundary test on a slowly transitioning synthetic context. Plot the mixture weights over time so we can see the soft handoff.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
