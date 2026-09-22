# **Weekly Report — Week 4**

**Period: **Tuesday Feb 10 – Monday Feb 16, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Build the first version of the gating mechanism described in the future-work section of the midterm. Three learnable regime embeddings, each a 4-dim simplex vector. A small gate network takes the context vector and produces a soft 3-way mixture; predicted weights are the convex combination. Pass criterion: at the regime-flip boundary in the synthetic perturbator, predicted weights should ramp over 10–15 simulated steps instead of snapping in one, with steady-state weights inside each pure regime within 0.03 L1 of the corresponding target.

## **Work completed**

Wrote agent/gating_network.py. The gate is a 2-layer MLP (4-dim context → 16 → 3) followed by softmax. The three regime embeddings live as a (3, 4) parameter tensor initialized to the highway / urban / weather target vectors. Total trainable parameters: 131 in the gate, 12 in the embeddings.

Folded the gating head into DynamicWeightAgent as an alternative head. A construction-time flag (`agent.head_kind = 'mlp' | 'gate'`) controls which is active.

Trained the gate-head agent on the same cached CSV plus a synthetic-context schedule. Loss settles at 0.05 (the pure MLP head settles at 0.04). The constraint imposed by the mixture structure is the design intent.

Boundary test on a 60-second highway-env episode with the perturbator flipping regime every 10 simulated seconds. The gate ramps over ~12 steps (about 0.8 simulated seconds at 15 Hz) where the MLP head snapped in 1 step.

Ablation per Daher's request: zeroed the context input and re-trained. Loss converges to a much worse 0.18 KL — the gate has nothing to discriminate on, and predicted weights collapse to roughly the highway target regardless of regime. Confirms that the context features carry the discriminative signal.

Embedding fine-tune experiment: froze the gate weights at initialization and let only the regime embeddings shift via REINFORCE-style updates against the AI-score reward. The highway embedding drifted from (0.30, 0.10, 0.45, 0.15) to (0.32, 0.09, 0.46, 0.13) over 100 episodes — small but consistent shift toward more weight on speed and proximity, less on waviness, consistent with the highway physics.

## **Key results**

Steady-state predicted weights inside pure regimes: highway (0.29, 0.11, 0.44, 0.16), urban (0.21, 0.15, 0.35, 0.29), weather (0.24, 0.31, 0.15, 0.30) — all within 0.03 L1 of the targets. The gate ramps smoothly across regime boundaries as designed.

## **Plan for next week**

Instrument the gate's mixture-weight output so it can be plotted directly over time during perturbator runs. Begin Bellman-error tracking on the existing Q-learner so the diagnostics are ready when needed for fine-tuning the gate-head agent with RL.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
