# **B. Reinforcement learning, theory and methods**

The textbook plus five method-specific papers. Sutton and Barto's textbook is the canonical reference; the others are the specific algorithmic contributions I used directly. Bellemare 2016 (count-based exploration) and Pathak 2017 (curiosity by self-supervised prediction) are the two competing approaches I evaluated for the conservative-bias fix in week 6 — chose Bellemare because the Q-learner state space is small and discrete enough that count-based suffices without needing a learned predictor.

**[6] **R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA: MIT Press, 2018.

***Used in this project: ****[seminal] The textbook. Cited for the foundational treatment of Q-learning, value functions, exploration-exploitation, and reward shaping. The week-5 diagnostic instrumentation (Bellman-error tracking, state-visitation heatmaps) is essentially textbook RL practice applied to our specific case.*

**[7] **M. G. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos, "Unifying count-based exploration and intrinsic motivation," in Advances in Neural Information Processing Systems, 2016.

***Used in this project: ****[seminal] The basis for the count-based curiosity bonus implemented in week 6. The 4D normalized telemetry is discretized into a 10⁴ grid; bonus reward is β / sqrt(N(s) + 1). Operating point β = 0.1. Result: high-prox-column Bellman error dropped 50%, per-class accuracy on under-represented classes jumped 13.5 percentage points. Cited in the February Monthly Report.*

**[8] **D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell, "Curiosity-driven exploration by self-supervised prediction," in International Conference on Machine Learning, 2017.

***Used in this project: ****Evaluated as the alternative to Bellemare's count-based approach. Pathak's ICM is designed for high-dimensional observations like pixels, where you need a learned forward model to provide the curiosity signal. Our state space is 4D and tabular, so count-based is the right pick. Documented in week 5's lit triage.*

**[9] **A. Y. Ng, D. Harada, and S. Russell, "Policy invariance under reward transformations: Theory and application to reward shaping," in International Conference on Machine Learning, vol. 99, 1999, pp. 278-287.

***Used in this project: ****The theoretical reference for reward shaping. Used to validate that the Social Latency penalty (a difference of post-action and pre-action neighbor AI) is a potential-based shaping that preserves the optimal policy structure of the underlying problem. Cited in the Final Report's Methods section.*

**[10] **T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, "Continuous control with deep reinforcement learning," arXiv preprint arXiv:1509.02971, 2015.

***Used in this project: ****Cited as the canonical DDPG reference for continuous control in RL. Read but not directly used — the project's agent operates in a continuous output space (the 4-dim weight vector via softmax) but the training loss is supervised KL plus REINFORCE-style updates, not actor-critic. Cited in the Milestone-1 references.*

**[11] **W. B. Knox and P. Stone, "Interactively shaping agents via human reinforcement learning," in Proceedings of the Fifth International Conference on Knowledge Capture, 2009, pp. 9-16.

***Used in this project: ****Cited as background for the principle that human-shaped rewards (in our case, the hand-picked target weights for each regime) can serve as anchors for learned weight predictors. The embedding fine-tune experiment in week 4 is conceptually adjacent — start from human-picked weights, let the agent drift slightly via RL signal.*

VIPP 201A Annotated Bibliography  —  Hadi Al Shmaissani  —  Page  of
