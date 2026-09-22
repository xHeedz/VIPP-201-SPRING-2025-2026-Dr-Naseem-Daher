# **Preface**

This is my running log for the second term of the VIP project on driver aggressiveness. The first term ended with the Milestone-1 report submitted in mid-January, where we got the static rule-based pipeline to work and trained the first Q-learner on highway-env, plus the closed-form Aggressiveness Index for highway, urban, and the friction-aware regime. This notebook picks up from there.

I'm keeping it week by week, starting Tuesday Jan 20 (first day back from break) through Friday May 1 (last working day before the final report is due May 11). Each week has a Monday or Tuesday goals block, then daily entries that try to be honest about what I actually did vs. what I planned to do, and a short "plan for next week" line. Where I quote code or numbers, those are what I committed and what I read off the screen at the time. The point of this thing isn't to look polished — it's to be a real trace of the work.

The four big tracks the midterm report flagged as future work, ranked roughly: (1) the PyTorch DynamicWeightAgent that replaces the discrete Q-table; (2) the gating / attention layer for soft transitions between regimes; (3) moving from highway-env to SUMO for scale, and ideally a CARLA prototype for real urban telemetry; (4) wrapping everything in a ROS2 publisher node. Spoiler: not all four landed cleanly. CARLA got descoped. That's recorded in here too.

# **Week 1: Tuesday Jan 20 – Monday Jan 26, 2026**

**Theme — ***Re-grounding after break + a real roadmap for the term.*

## **Goals for the week**

Get my head back into the project after three weeks off, re-read Milestone-1 cover to cover, and turn Prof. Daher's mid-January feedback into something actionable. The main thing he hammered on: the Q-learner is "memorizing not learning." The accuracy curve looks great because the labels (Conservative / Normal / Aggressive) come from the same threshold rule that produces the AI score the agent is supposed to predict. Circular. The agent never sees a real counterexample. Fixing that is basically the term.

## **Daily entries**

**Tue Jan 20***  —  First day back. Re-read midterm + comments.*

Morning was mostly re-reading the submitted PDF and Daher's annotations. The most painful comment is on Section 5 — the rolling-average accuracy plot looks artificially clean precisely because labels and predictions both come from the same equation. The Q-learner is essentially a lookup table for a function we already wrote. To make term-2 actually mean anything, the agent has to predict something the closed-form math can't trivially produce. Which means: agent learns the weights themselves, conditioned on environment, instead of mapping AI → category.

Spent the afternoon doing a what-actually-exists-in-the-repo audit vs. what the report implied existed. The midterm casually mentioned a "DynamicWeightAgent in PyTorch" — but the only thing actually committed is the numpy DriverQLearner in learning_agent.py. So the PyTorch agent is net-new code, not a refactor. Useful to know upfront — term-2 starts smaller than I thought.

**Wed Jan 21***  —  Whiteboard roadmap.*

Got to the lab around 10. Sketched the whole term on the whiteboard, took a phone pic, transcribed it into the repo's TODO.md before someone erased it (which has happened before). Four tracks ranked roughly by importance: (1) PyTorch DynamicWeightAgent that takes telemetry + environmental context and outputs the four AI weights; (2) a gating / attention layer so the same agent can shift smoothly between highway, urban, and weather regimes instead of switching abruptly; (3) SUMO transition so we can simulate hundreds of vehicles instead of the 15 we get in highway-env; (4) a thin ROS2 wrapper that publishes the AI score on a topic.

Stretch column — probably won't all land: CARLA for real occlusion-aware urban telemetry, slip-ratio telemetry for the friction track, and a multi-objective reward with a Social Latency penalty so the agent doesn't get steamrolled by an aggressive neighbor. I'm leaving CARLA in the stretch column because the install footprint alone (~30 GB plus a dedicated GPU) is a lot of friction on the lab machine, and we already lost half a Saturday to it last term.

**Thu Jan 22***  —  Lit triage.*

Pulled up the references I cited in the midterm and re-read with sharper attention this time around. Kiran et al. 2022 (the DRL-for-AD survey) — page 10 has a clean taxonomy of state representations that I want to lift for the design memo. Wang et al. 2017 on driving styles — useful framing for why "aggressiveness" is better treated as a continuous latent than a 3-way bucket. Zhu et al. 2021 on reward design for AD — they argue against piecewise-constant rewards because they create exploration dead zones, which I should worry about given our current Reward = 1.0 - AI structure.

Added to the queue: the Shalev-Shwartz RSS paper (already cited but not really read), to use as the safety floor for any multi-objective reward; and the original Vaswani "Attention is All You Need" paper, because the gating layer in track (2) is basically a small attention head over environmental context vectors. Will read it properly over the weekend.

**Fri Jan 23***  —  Design memo.*

Wrote a one-page memo for the new agent. Inputs at each step: a kinematic tuple (n_speed, n_accel, n_prox, n_wave) plus an environmental context vector (one-hot road type, normalized traffic density, normalized friction coeff, weather flag). Output: four weights (w_speed, w_accel, w_prox, w_wave) that get fed into the existing AI formula. Loss: a combination of supervised (match a hand-picked target weight vector for each regime) and RL (maximize 1.0 - AI on episodes where the ego car doesn't crash). Will tune as we go.

Action space is no longer LANE_LEFT / IDLE / etc. — those don't fit any more. The agent isn't driving the ego car, it's predicting the right weights for the AI calculation given context. Closer to a regression policy than a Q-learner. Might end up with a small MLP with two heads (mean and log-variance) trained against an ELBO, but for the first cut the simplest thing is a 2-layer MLP with a softmax over the four outputs so they sum to 1.

**Sat Jan 24***  —  Repo cleanup + env setup.*

Power was out from like 9 to 1 so I worked at home off the laptop on battery, then headed to the lab. Cleaned up the repo. Made a `legacy/` folder and moved the Phase-1 scripts (aggressiveness_index.py, gym_simulation.py, main.py, simple_env.py) into it. New layout: `model/` for AI math, `agent/` for RL agents, `env/` for simulator wrappers, `notebooks/` for ad hoc Jupyter work, `scripts/` for entrypoints. Wrote a placeholder pyproject.toml with pinned versions: Python 3.11, PyTorch 2.2, gymnasium 1.0.0, highway-env 1.10.1, pandas 2.2, numpy 1.26.

Hit one minor breakage worth flagging: the new highway-env release uses `env.unwrapped.configure(dict)` instead of the old `env.unwrapped.config.update(dict)` pattern that's in our data_collector.py from last term. Updated it in place so the Phase-1 pipeline still runs end-to-end if we need to regenerate the CSV. Verified by running data_collector.py and getting `driving_behaviors_dataset.csv` written out with ~1800 rows. Same as before.

**Sun Jan 25***  —  Quiet day, reading.*

Light day. Read the first half of Vaswani 2017 at the apartment with coffee. The full cross-attention math is overkill for what we need here, but the queries / keys / values framing is the right vocabulary for the gating idea — query is the environmental context, keys are learned regime embeddings (highway, urban, weather), values are the corresponding weight vectors. So in practice this becomes a 1-head softmax attention over three learnable regime vectors. Tiny. Trainable on the data we already have.

**Mon Jan 26***  —  Sync with Prof. Daher.*

Caught him for ten minutes between his classes. Walked him through the design memo. Two pieces of pushback I want to record: (1) he asked whether the normalization functions themselves should be learnable too — i.e. should the saturation point of N_speed be a parameter rather than the constant 150 km/h. Conceded the point but argued for keeping it constant for the term to bound scope. He agreed. (2) He flagged that the supervised target weights I'm proposing as regression targets are themselves hand-picked, so they're not really ground truth. His suggestion: bootstrap the targets with hand-picked values, then let an RL fine-tune phase shift them. Recorded as a TODO.

## **Reflection**

Slow week, on purpose. The honest framing is that the midterm work was wider than it was deep — lots of infrastructure but the agent itself is a thin wrapper around a closed-form score. Term-2 trades breadth for depth: one well-trained PyTorch agent that does a real prediction job, with the gating layer and SUMO transition supporting it. CARLA stays out of scope unless something gives me unexpected GPU time.

## **Plan for next week**

Stand up the PyTorch DynamicWeightAgent skeleton: 2-layer MLP, 8-dim input (4 telemetry + 4 context), 4-dim weight output via softmax. Get a single forward + backward pass running end-to-end on the cached Phase-1 CSV. No environment integration yet — just shape-check and gradient flow.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
