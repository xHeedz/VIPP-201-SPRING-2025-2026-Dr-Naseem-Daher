# **Week 4: Tuesday Feb 10 – Monday Feb 16, 2026**

**Theme — ***Gating layer: replace the hard regime switch with a soft mixture.*

## **Goals for the week**

Build the first version of the gating idea from the midterm's future-work section. Three learnable regime embeddings (highway, urban, weather), each a 4-dim simplex vector. A small gate MLP takes the context vector and produces a 3-way mixture; predicted weights are the convex combination. Pass criterion: at the regime-flip boundary in the synthetic perturbator, predicted weights ramp over ~10–15 sim steps instead of snapping in one, and steady-state weights inside each pure regime stay within 0.03 L1 of target.

## **Daily entries**

**Tue Feb 10***  —  Gate architecture.*

Wrote agent/gating_network.py. Gate is a 2-layer MLP: 4-dim context → 16 → 3, then softmax. Three regime embeddings live as a (3, 4) parameter tensor, initialized to the highway / urban / weather targets. Forward pass computes mixture via the gate and returns convex combo of embeddings. Trainable params: 131 in the gate, 12 in the embeddings. Tiny on purpose — the layer should be cheap and interpretable.

Folded it into DynamicWeightAgent as an alternative head — flip a flag (`agent.head_kind = 'mlp' | 'gate'`) at construction time. MLP head is the week-3 baseline, gate head is the new thing.

class GatingHead(nn.Module):

    def __init__(self, ctx_dim=4, n_regimes=3, n_weights=4, init=None):

        super().__init__()

        self.gate = nn.Sequential(

            nn.Linear(ctx_dim, 16), nn.ReLU(),

            nn.Linear(16, n_regimes)

        )

        emb = torch.tensor(init, dtype=torch.float32) if init is not None \

              else torch.randn(n_regimes, n_weights)

        self.regime_emb = nn.Parameter(emb)

 

    def forward(self, ctx):

        mix = torch.softmax(self.gate(ctx), dim=-1)        # (B, R)

        emb = torch.softmax(self.regime_emb, dim=-1)        # (R, W)

        return mix @ emb                                    # (B, W)

**Wed Feb 11***  —  Training run + boundary check.*

Trained the gate-head agent on the same cached CSV plus the synthetic-context schedule — 200 epochs, KL loss against per-regime target. Loss settles at 0.05, a hair worse than the pure MLP head's 0.04, which makes sense — the gate is more constrained. The constraint is the point.

Boundary test: 60-second highway-env episode with the synthetic perturbator flipping regime every 10 sim seconds. Plotted predicted (w_speed, w_accel, w_prox, w_wave) over time. Where the MLP head snapped at the flip in 1 step, the gate ramps over ~12 steps (about 0.8 sim seconds at 15 Hz). Ramp shape is sigmoid-ish — the dominant regime score has to climb past the others before its mixture weight saturates. Steady-state weights inside each pure regime: highway (0.29, 0.11, 0.44, 0.16), urban (0.21, 0.15, 0.35, 0.29), weather (0.24, 0.31, 0.15, 0.30). All within 0.03 L1 of targets. Pass.

**Thu Feb 12***  —  Ablation: no context.*

Ran the ablation Daher asked for. Killed the context input by zeroing it on every step and re-trained. Loss converges to a much worse 0.18 KL — the gate has nothing to discriminate on, so the regime embeddings collapse toward whichever regime has the most training samples (highway, in our case). At eval time, predicted weights are ~(0.27, 0.13, 0.41, 0.19) regardless of what the perturbator says. Confirms what we hoped to demonstrate: the context features carry the signal.

Saved the ablation as a side-by-side bar chart for Monday — three bars per regime: target, with-context prediction, no-context prediction. The without-context bars are visibly wrong on urban and weather but coincidentally close on highway, which itself is instructive — it shows the model defaulting to highway when starved of input.

**Fri Feb 13***  —  Embedding fine-tune experiment.*

Tried something a bit speculative: freeze the gate weights at their initialized values (so the gate still produces a sensible mixture) but let only the regime embeddings fine-tune via the AI-score reward. The intuition is that hand-picked targets are anchors, not gospel — if the agent finds slightly different (w_speed, w_accel, w_prox, w_wave) that produces better Reward = 1 - AI on average, that drift is informative.

Result after 100 episodes of REINFORCE-style updates with lr=1e-4 and a baseline subtracted: highway embedding drifted from (0.30, 0.10, 0.45, 0.15) to (0.32, 0.09, 0.46, 0.13). Small drift, but consistent — the agent gives marginally more weight to speed and proximity, less to waviness, which makes sense on a clean highway. Urban and weather embeddings barely moved because the synthetic perturbator doesn't actually generate real urban or weather physics, so the reward signal in those regimes is mostly noise. Good motivator for the SUMO transition.

**Sat Feb 14***  —  Reading: Vaswani + Shazeer 2017 MoE.*

Re-read the relevant bits of Vaswani et al. and the Shazeer 2017 outrageously-large-MoE paper. Confirmed that what I implemented is essentially a 1-head, 1-layer attention with 3 fixed slots, mathematically equivalent to the simplest mixture-of-experts gate. Reassuring — the literature is on our side, and if we ever want to scale to more regimes (rural, on-ramp, intersection, …) the same code structure extends naturally to k slots.

**Sun Feb 15***  —  Off.*

Off.

**Mon Feb 16***  —  Sync + scope for week 5.*

Sync with Daher. Walked through the boundary test, the ablation, the embedding-drift experiment. He liked all three but pushed on the mixture-weight interpretation: he wants me to plot the mixture weights themselves over time during the perturbator runs to confirm the gate is producing meaningful smoothing rather than noise. Agreed scope for week 5: instrument the gate output, produce the plot, start thinking about Bellman-error tracking that will eventually replace the supervised KL loss for fine-tuning.

## **Reflection**

Gate works. Ablation cleanly shows context matters. The embedding fine-tune experiment surfaces a useful asymmetry — we can drift on highway but not on synthetic urban or weather, because the perturbator doesn't produce real reward signal in those regimes. That's the strongest argument so far for moving to SUMO mid-term.

## **Plan for next week**

Instrument the gate so we can directly inspect mixture weights over time, write the matplotlib panel for that, and start the Bellman-error-tracking feature on the existing Q-learner so we have it ready when we need to fine-tune the gate-head agent with RL.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
