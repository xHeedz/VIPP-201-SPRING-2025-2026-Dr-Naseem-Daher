# **Weekly Report — Week 3**

**Period: **Tuesday Feb 3 – Monday Feb 9, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Plug the DynamicWeightAgent into the highway-env step loop so each step produces predicted weights → AI score, and log everything to a per-episode CSV. Compare against the fixed-weight baseline. Add a jerk channel to the telemetry, which Prof. Daher has been asking for since last term.

## **Work completed**

Wrote env/highway_adapter.py that pulls the ego vehicle's row from the highway-env native observation, computes (n_speed, n_accel, n_prox, n_wave) using the same normalization functions as the AggressivenessModel, attaches the current context vector, and feeds the 8-dim vector to the agent each step. The waviness term is stateful (it accumulates steering and heading deltas over a short window), so the adapter maintains a small ring buffer.

Achieved the first end-to-end run on a controlled 200-step highway-env episode: mean AI score = 19.7 with predicted weights, vs. 22.4 with fixed weights. The predicted-weight version is ~12% lower, which raised an immediate question of whether the agent is gaming the score; this is logged as a follow-up.

Added a jerk channel to the telemetry (numerical derivative of acceleration over a 3-step window). Verified on a controlled lane-change scenario: peaks at ~3.2 m/s³ during the change, near zero during steady cruise. Updated the AI formula's input tuple to include n_jerk, retrained the agent.

Per-context profiling across 50 random highway-env episodes: mean predicted highway weights (0.31, 0.09, 0.44, 0.16), within 0.02 L1 of the target. Forcing the context input to urban while keeping the underlying physics highway flips the predicted weights to (0.20, 0.15, 0.35, 0.30) — the agent responds to the context input even when the physics doesn't match. This confirms the supervised signal is binding to context as designed.

Implemented gradient-norm instrumentation per Daher's earlier request: total grad norm decays cleanly from ~0.4 (epoch 0) to ~0.05 (epoch 200), no spikes or flat zones. No layer is starving.

## **Key results**

End-to-end pipeline working: highway-env → adapter → agent → AI score, with logging at 15 Hz. Predicted weights track the highway target on highway-only seeds. Adversarial context-flip test confirms responsiveness to the context vector.

## **Issues encountered**

Lost approximately half a day to a BatchNorm-related bug. The agent had been producing identical predicted weights every step regardless of state. Root cause: BatchNorm had been re-enabled on a stale branch that I had merged in by mistake; with batch size 1 at inference, BN's running statistics make every input look the same once the mean/var saturate. Removed BN and added an init-time assertion that no nn.BatchNorm1d module is in the agent.

## **Plan for next week**

Build the gating network. Three learnable regime embeddings (highway, urban, weather), a small MLP gate that maps the context vector to a 3-way mixture, predicted weight vector is the convex combination. Boundary test on a slowly transitioning synthetic context to demonstrate the soft handoff.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
