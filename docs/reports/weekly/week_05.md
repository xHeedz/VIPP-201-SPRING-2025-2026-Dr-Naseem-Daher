# **Weekly Report — Week 5**

**Period: **Tuesday Feb 17 – Monday Feb 23, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Two diagnostics. (1) Mixture-weight panel: plot the gate's 3-way mixture over time during a regime-changing episode to confirm the soft handoff is smooth and not noisy. (2) Bellman-error tracking on the Q-learner: log |target − Q(s,a)| per update and plot its distribution over training. The point of the second diagnostic is to identify states where the agent is consistently confidently wrong — the textbook memorizing-not-learning signature Daher flagged in January. As a bonus, produce a state-visitation heatmap over the 4D normalized telemetry space.

## **Work completed**

Wrote eval/plot_mixture_weights.py producing a 3-panel figure: input context channels, gate mixture weights, predicted AI weights — all sharing an x-axis. Ran on a 90-second synthetic episode cycling highway → urban → weather → highway. Mixture-weight panel shows clean sigmoid handoffs centered on the regime-flip events with crossover behavior between highway and urban.

Instrumented the Q-learner's update path to log Bellman error δ = target − Q(s, a) per (s, a) update along with the corresponding state. Re-ran the Q-learning experiment from the midterm with logging on.

Produced the state-visitation heatmap: four 2D marginals of the normalized telemetry space, each a 32×32 grid binning the agent's visitation counts over 1000 training episodes.

## **Key results**

Bellman-error histogram is heavily right-skewed: 95% of |δ| values are below 0.05, but a long tail extends to 0.6. The high-|δ| samples concentrate in the (low n_speed, high n_prox) corner of state space — exactly where the Phase-1 dataset is sparsest.

State-visitation heatmap shows ~70% of all visitation concentrated in the (n_speed > 0.7, n_prox < 0.3) corner — fast and unobstructed driving — and almost no visitation in the high-n_prox column where the Bellman error is also worst. Translation: the Q-learner has implicitly learned a policy that minimizes its chance of finding out it is wrong, by staying in the easy region of state space.

This is the conservative-bias problem and it is exactly what Prof. Daher flagged in January. The midterm's clean accuracy curve was a function of where the agent chose to drive, not of how well its predictions actually generalize.

## **Issues encountered**

The diagnostic results are uncomfortable but they confirm the central critique. The fix has to involve some form of intrinsic motivation that pulls the Q-learner into under-visited states — count-based or curiosity-driven exploration. Acceptable approach for next week.

## **Plan for next week**

Implement count-based curiosity bonus on the Q-learner. Discretize the 4D telemetry into a 10⁴ grid, maintain visitation counts N(s), add β / sqrt(N(s) + 1) to the reward. Sweep β ∈ {0.01, 0.05, 0.1, 0.5, 1.0}. Pass criterion: state-visitation distribution becomes more uniform across the (n_speed, n_prox) marginal; Bellman error in the high-prox column drops by at least 30%.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
