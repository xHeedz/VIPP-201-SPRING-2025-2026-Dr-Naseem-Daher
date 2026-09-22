# **Monthly Progress Report — February 2026**

*Hadi Al Shmaissani · Project 3 · Supervisor: Prof. Naseem Daher*

**Abstract — **February covered four substantial pieces of work. First, wiring the PyTorch DynamicWeightAgent into the highway-env simulation loop, including a small adapter that builds the agent's input vector each step and a jerk channel added to the telemetry. Second, building the gating layer that replaces the hard regime switch with a soft mixture over three learnable regime embeddings; an ablation in which the context input is zeroed confirms that the context features carry the discriminative signal. Third, a diagnostic suite — Bellman-error tracking on the Q-learner plus a state-visitation heatmap — that exposed a conservative-bias problem the midterm's accuracy curve was hiding: the Q-learner had implicitly learned a policy that minimizes its chance of finding out it is wrong by staying in the easy region of state space. Fourth, a count-based exploration bonus that fixes the conservative-bias problem: at β = 0.1 the high-prox-column Bellman error drops by 50% and per-class accuracy on under-represented classes improves from a 74% average to 87.5%. February closes with the diagnostic-and-fix loop completed and the agent stack ready for the SUMO transition planned for March.

## **1. Introduction**

January closed with the minimum-viable PyTorch DynamicWeightAgent training to convergence on cached telemetry. February was scheduled to take the agent from a standalone training script to a wired-in component of the simulation loop, then to extend it with the gating layer described in the midterm's future-work section. A parallel track — the diagnostic infrastructure that addresses Prof. Daher's memorizing-not-learning critique — was originally scheduled for March but moved up after week 4's results showed that the diagnostics could be implemented quickly, and that the conservative-bias problem they exposed had to be fixed before any further claims about the Q-learner could be supported.

## **2. Work Completed This Month**

### **2.1 Wiring the Agent into highway-env**

An adapter (env/highway_adapter.py) was written to bridge the highway-env native observation format and the agent's expected 8-dimensional input. The adapter pulls the ego vehicle's row from the (V, 5) observation array, computes (n_speed, n_accel, n_prox, n_wave) using the same normalization functions as the AggressivenessModel, attaches the current context vector, and feeds the resulting vector into the agent each step. Because the waviness term is stateful — accumulating steering and heading deltas over a short window — the adapter maintains a small ring buffer.

A jerk channel (numerical derivative of acceleration over a 3-step window) was added to the telemetry per Prof. Daher's standing request from the midterm. On a controlled lane-change scenario, jerk peaks at ~3.2 m/s³ during the change and is near zero during steady cruise. The agent's input dimension grew from 8 to 9 to absorb the new channel.

Per-context profiling across 50 random highway-env episodes produced mean predicted weights of (0.31, 0.09, 0.44, 0.16), within 0.02 L1 of the highway target. A diagnostic test in which the context input is forcibly switched to urban while the underlying physics remain highway flips the predicted weights to (0.20, 0.15, 0.35, 0.30) — exactly the urban target. This confirms the supervised signal is binding to the context vector rather than to the physics, which is the design intent.

### **2.2 Gating Network**

The gating head (agent/gating_network.py) is a 2-layer MLP that maps the context vector to a 3-dimensional simplex via softmax. Three learnable regime embeddings (a (3, 4) parameter tensor initialized to the highway / urban / weather targets) are combined as a convex combination weighted by the gate output. Total trainable parameters: 131 in the gate and 12 in the embeddings — small by design. The gate-head agent reaches a KL loss of 0.05 (vs. 0.04 for the pure MLP head); the small loss penalty is the price of the architectural constraint and is the design intent.

On a 60-second highway-env episode with a synthetic perturbator flipping regime every 10 simulated seconds, the gate ramps over approximately 12 simulation steps (about 0.8 simulated seconds at 15 Hz) where the unconstrained MLP head snapped in 1 step. Steady-state weights inside each pure regime are within 0.03 L1 of the targets: highway (0.29, 0.11, 0.44, 0.16), urban (0.21, 0.15, 0.35, 0.29), weather (0.24, 0.31, 0.15, 0.30).

An ablation requested by Prof. Daher zeroed the context input and re-trained the gate. Loss converges to a much worse 0.18 KL — the gate has nothing to discriminate on, and the regime embeddings collapse toward the highway target (the dominant regime in the training set). At evaluation the predicted weights are roughly (0.27, 0.13, 0.41, 0.19) regardless of regime. The ablation confirms that the context features carry the discriminative signal.

### **2.3 Diagnostic Suite: Bellman Error and State Visitation**

Bellman-error logging was added to the existing DriverQLearner: for each (s, a) update, δ = target − Q(s, a) is appended to a CSV alongside the discretized state. After re-running the Q-learning experiment from the midterm with logging on, the histogram of |δ| is heavily right-skewed: 95% of values fall below 0.05, but the long tail extends out to 0.6, and the high-|δ| samples concentrate in the (low n_speed, high n_prox) corner of state space.

A state-visitation heatmap (four 2D marginals of the normalized telemetry space, each a 32×32 grid binning visitation counts over 1000 training episodes) reveals the structural problem: roughly 70% of all visitation concentrates in the (n_speed > 0.7, n_prox < 0.3) corner — fast and unobstructed driving — and almost no visitation in the high-n_prox column where the Bellman error is also worst. The interpretation is direct: the Q-learner has implicitly learned a policy that minimizes its chance of finding out it is wrong, by avoiding exactly the under-explored states where its predictions would fail. The midterm's clean accuracy curve is therefore an artifact of the agent choosing its own evaluation distribution.

### **2.4 Count-Based Curiosity Bonus**

The fix for the conservative-bias problem is intrinsic motivation. A count-based exploration bonus (Bellemare et al. 2016) was selected over alternatives such as ICM (Pathak et al. 2017) on the grounds that the Q-learner is already tabular over a discretized state, so count-based does not require a learned predictor. The 4D normalized telemetry is discretized into a 10⁴ grid; visitation counts N(s) are maintained per cell; the bonus reward is β / sqrt(N(s) + 1) added to the environment reward.

R_total(s) = R_env(s) + β / sqrt(N(s) + 1)

A sweep over β ∈ {0.01, 0.05, 0.1, 0.5, 1.0} found β = 0.1 to be the operating point. Below 0.05, the bonus is too small to matter; above 0.5, the agent over-explores noisy bad states for the bonus and converges worse than the no-curiosity baseline. The bonus is verified to be transient: it dominates env_reward for the first ~30 episodes, but by episode 80 it is essentially negligible, so evaluation comparisons (where the bonus is removed) are fair.

## **3. Results**

Predicted-weight tracking. The wired-in agent produces predicted weights consistent with the regime-appropriate hand-picked targets across 50 random highway-env episodes. The gate-head version smoothly handles regime transitions (a 12-step ramp vs. a 1-step snap) while preserving the steady-state target match within 0.03 L1.

Bellman-error reduction at β = 0.1. The high-prox-column mean |δ| drops from 0.18 to 0.09 — a 50% reduction, well above the 30% target. The right tail of the histogram shrinks accordingly, with 95% of |δ| under 0.03 and the tail going out to 0.4 instead of 0.6.

Per-class accuracy improvements on a held-out perturbed dataset (computed by perturbing the Phase-1 CSV with random context noise and re-bucketing): Conservative class went from 67% to 89% (+22 percentage points), Normal from 81% to 86% (+5), Aggressive from 95% to 94% (essentially unchanged, already saturated). Average over the under-represented classes: 74% → 87.5%.

State-visitation distribution after training with curiosity is visibly more uniform across the (n_speed, n_prox) marginal. The easy-corner concentration drops from 70% to roughly 40% — the agent now explores enough of the state space that its predictions can be trusted on under-represented states.

## **4. Discussion and Limitations**

The core conceptual move this month is the explicit acknowledgment that the midterm's accuracy curve was a function of where the agent chose to drive, not of how well its predictions actually generalize. The diagnostic-and-fix loop in weeks 5–6 was the most important sequence of work in the term so far. The fix — count-based curiosity at β = 0.1 — is mathematically uncomplicated; what mattered was having the diagnostics in place to surface the problem in the first place.

An unresolved-but-intriguing observation: in the gate-head's embedding fine-tune experiment, the highway embedding drifted in a small but consistent direction (toward more weight on speed and proximity, less on waviness) — suggesting the hand-picked target weights are not exactly optimal even on the highway regime. The drift is small enough not to chase further this term, but it is a useful motivator for a longer RL fine-tune phase in future work.

## **5. Plan for Next Month**

March will execute the SUMO transition. The first step is the long-horizon stability check requested by Prof. Daher: 5000 episodes of training with the curiosity-augmented agent, with predicted weights tracked over time and drift bounded against the targets. In parallel, SUMO will be installed and a Python wrapper around the TraCI API written, exposing the same interface as the existing highway-env wrapper so that scenarios can be swapped via a config flag. The wrapper will be tested on a 3-lane highway-equivalent SUMO scenario, and the heterogeneous-driver experiment from the midterm's future-work section will be set up: a fraction of aggressive vehicles seeded into a normal-driver population, with the AI score field used to detect them and to characterize any shockwave propagation through neighboring vehicles.

## **References**

[1] M. G. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos, "Unifying count-based exploration and intrinsic motivation," Adv. Neural Inf. Process. Syst., 2016.

[2] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell, "Curiosity-driven exploration by self-supervised prediction," Int. Conf. Mach. Learn., 2017.

[3] N. Shazeer et al., "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," Int. Conf. Learn. Represent., 2017.

[4] A. Vaswani et al., "Attention is all you need," Adv. Neural Inf. Process. Syst., 2017.

[5] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed., MIT Press, 2018.

VIPP 201A Monthly Reports  —  Hadi Al Shmaissani  —  Page  of
