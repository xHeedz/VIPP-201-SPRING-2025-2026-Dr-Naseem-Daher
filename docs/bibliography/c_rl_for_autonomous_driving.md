# **C. Reinforcement learning for autonomous driving**

Two AD-specific RL references. Kiran et al. is the most useful single reference for orienting in this subfield; Zhu et al. specifically informs the reward-design choices.

**[12] **B. R. Kiran, I. Sobh, V. Talpaert, P. Mannion, A. A. Al Sallab, S. Yogamani, and P. Pérez, "Deep reinforcement learning for autonomous driving: A survey," IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 6, pp. 4909-4926, 2022.

***Used in this project: ****[seminal] The survey paper for RL applied to AD. Section 3 (state representations) directly informed how I designed the agent's input vector — kinematic tuple plus environmental context. Page 10's taxonomy of state representations was lifted (with attribution) for the design memo in week 1.*

**[13] **M. Zhu, X. Wang, and Y. Wang, "Design of reward function on reinforcement learning for automated driving," Transportation Research Part C: Emerging Technologies, vol. 125, p. 103001, 2021.

***Used in this project: ****The most directly useful reward-design reference. Their argument against piecewise-constant rewards (because they create exploration dead zones) is exactly the failure mode our Reward = 1 − AI structure could have suffered without the smooth gating layer. Read in week 1 and re-read after the week-5 diagnostic.*

VIPP 201A Annotated Bibliography  —  Hadi Al Shmaissani  —  Page  of
