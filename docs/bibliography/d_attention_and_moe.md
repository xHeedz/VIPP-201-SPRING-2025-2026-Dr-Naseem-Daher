# **D. Attention mechanisms and mixture-of-experts**

Two papers that informed the gating-network architecture in week 4. Vaswani et al. provided the queries/keys/values vocabulary; Shazeer et al. provided the MoE framing that lets the same architecture extend to more regimes if we add them in future work.

**[14] **A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems, 2017.

***Used in this project: ****[seminal] The attention paper. Read carefully over the weekend of January 24-25 for the gating-layer design. The full cross-attention math is overkill for what we need, but the queries / keys / values framing is the right vocabulary: query is the environmental context vector, keys are learned regime embeddings (highway, urban, weather), values are the corresponding weight vectors. The implemented gate is mathematically equivalent to a 1-head attention with three fixed slots.*

**[15] **N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in International Conference on Learning Representations, 2017.

***Used in this project: ****The MoE reference confirming that what I implemented is mathematically the simplest case of a mixture-of-experts gate. Useful for the future-work section: if we ever want to scale to more regimes (rural, on-ramp, intersection, etc.), the same code structure extends naturally to k slots.*

VIPP 201A Annotated Bibliography  —  Hadi Al Shmaissani  —  Page  of
