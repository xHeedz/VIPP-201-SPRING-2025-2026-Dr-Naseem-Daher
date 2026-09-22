# **Weekly Report — Week 2**

**Period: **Tuesday Jan 27 – Monday Feb 2, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Build the minimum viable PyTorch agent: an 8-dim → 64 → 64 → 4-dim MLP with softmax output, a supervised KL loss against hand-picked per-regime target weights, and a training loop that runs against the Phase-1 telemetry CSV.

## **Work completed**

Wrote agent/dynamic_weight_agent.py implementing the 2-layer MLP described above. Decided against BatchNorm: inputs are already normalized to (0, 1] by construction, and BN with batch size 1 (the deployment case) introduces inference-time noise.

Augmented the Phase-1 CSV with synthetic per-row context tags. The Phase-1 data has no explicit context column — every sample is implicitly highway. A small heuristic loader tags rows as urban (high n_prox), highway (high n_speed, low n_prox), or weather-affected (high n_wave, moderate n_speed) so the agent has something to discriminate on. After tagging the dataset has roughly 1100 highway, 500 urban, and 200 weather samples.

Built a training script using AdamW (lr=1e-3, weight_decay=1e-4) and KL divergence loss against the regime-appropriate target weight vectors. Training runs in approximately 4 minutes on the lab CPU (the lab GPU was occupied by another VIP team's vision project).

## **Key results**

Loss curve drops cleanly from 0.32 to 0.04 over 200 epochs with no oscillation. A held-out highway sample produces predicted weights of (0.30, 0.10, 0.45, 0.15), within 0.01 L1 of the highway target. A perturbation eval (1000 small noise perturbations on the context vector, single telemetry row, urban sample) produces a predicted-weight distribution that is tight in (w_prox, w_wave) and broader in (w_speed, w_accel) — physically sensible for an urban regime.

## **Plan for next week**

Wire the agent into the highway-env simulation loop so it produces predicted weights at each step, feed those into the AggressivenessModel, and log the resulting AI score over a full episode. Compare against the Phase-1 fixed-weight baseline on a controlled scenario (same seed). Per Daher's request, also instrument gradient norms during training.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
