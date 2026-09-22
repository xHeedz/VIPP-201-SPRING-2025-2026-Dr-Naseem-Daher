# **A. Driver behavior and aggressiveness**

These five references frame the central problem the project addresses: how to identify and quantify aggressive driving from telemetry. The split between classical statistical approaches (Murphey, Kesting) and modern deep-learning approaches (Deng, Li) is what motivates the project's main architectural choice — a closed-form interpretable index that can compose with downstream planners, rather than a deep classifier whose outputs are opaque.

**[1] **Y. L. Murphey, R. Milton, and L. Kiliaris, "Driver's aggressiveness classification using vehicle signals," in IEEE Conf. Intelligent Transportation Systems, 2009, pp. 1-6.

***Used in this project: ****[seminal] The closest classical-era neighbor to this project's closed-form Aggressiveness Index. Murphey et al. classify driver aggressiveness from vehicle signals using a small fixed feature set; the project's AI score is essentially a normalized, weighted version of the same idea, but with per-regime weight vectors and a context-aware gating layer. Cited in Section 2 of the Final Report and in the March Monthly Report's references.*

**[2] **W. Wang, J. Xi, A. Chong, and L. Li, "A review of research on driving styles and their applications in self-driving vehicles," IEEE Access, vol. 5, pp. 22713-22723, 2017.

***Used in this project: ****[seminal] The survey paper that frames **"**driving style**"** as a continuous latent rather than a 3-way category. Used as the conceptual basis for treating Conservative/Normal/Aggressive as discretized buckets over a continuous score, rather than as ground-truth labels. Read carefully in week 1's literature triage; cited in the Final Report's related-work section.*

**[3] **A. Kesting, M. Treiber, and D. Helbing, "Agents for traffic simulation: Modeling and detecting driver behavior," Transportation Research Part C: Emerging Technologies, vol. 18, no. 3, pp. 289-304, 2010.

***Used in this project: ****Used for the framing of how heterogeneous driver populations interact in traffic-flow simulation. The 2010 paper's results on aggressive-driver impact on local accelerations and braking events are consistent with — and helped me interpret — the +6.8 AI shockwave bump measured in week 8's heterogeneous-driver experiment.*

**[4] **J. Deng, K. Yang, T. Sun, and J. Tomizuka, "Modeling and detecting aggressiveness from driving signals using deep learning," IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 10, pp. 6312-6325, 2021.

***Used in this project: ****Cited as the representative modern deep-learning approach to driver-aggressiveness classification. The Final Report contrasts this approach (high accuracy but opaque outputs) with the project's interpretable-formula approach. Read but not deeply leaned on — the project's design choices go in the opposite direction from this work.*

**[5] **N. Li, Y. Yao, I. Kolmanovsky, E. Atkins, and A. Girard, "Formulating vehicle aggressiveness towards social cognitive autonomous driving," Nature Communications, vol. 12, no. 1, pp. 1-11, 2021.

***Used in this project: ****Borrowed the **"**socially cognitive driving**"** framing for the Final Report's discussion section. The Social Latency penalty implemented in week 10 (R = (1 − AI_self) − λ · max(0, ΔAI_neighbors)) is conceptually aligned with the social-cognitive framing this paper develops, though my mechanism is much simpler than what they propose.*

VIPP 201A Annotated Bibliography  —  Hadi Al Shmaissani  —  Page  of
