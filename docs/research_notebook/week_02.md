# **Week 2: Tuesday Jan 27 – Monday Feb 2, 2026**

**Theme — ***Stand up the PyTorch agent and get gradients flowing.*

## **Goals for the week**

Build the minimum viable PyTorch agent: 8-dim → 64 → 64 → 4-dim MLP with softmax output, supervised loss against hand-picked target weights, a training loop that runs against the Phase-1 telemetry CSV. Pass criterion for the week: training loss drops monotonically over 200 epochs on the cached data, and the predicted weight vector for at least one held-out highway sample is within 0.05 (L1) of the highway target.

## **Daily entries**

**Tue Jan 27***  —  agent/dynamic_weight_agent.py first draft.*

Wrote the first version of agent/dynamic_weight_agent.py. Forward pass works, shapes line up. Used nn.Sequential + a final softmax so the four outputs are always on the simplex. Decided to keep BatchNorm out for now — the inputs are already normalized in (0,1] by construction (n_speed, n_accel, n_prox, n_wave), and adding BN here just adds noise during inference when batch size is 1, which is the deployment case. Tested with a dummy batch of (32, 8) and confirmed (32, 4) output that sums to 1 along axis 1. Good.

class DynamicWeightAgent(nn.Module):

    def __init__(self, in_dim=8, hidden=64, out_dim=4):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(in_dim, hidden), nn.ReLU(),

            nn.Linear(hidden, hidden), nn.ReLU(),

            nn.Linear(hidden, out_dim)

        )

    def forward(self, x):

        # x: (B, 8) = [n_speed, n_accel, n_prox, n_wave, ctx(4)]

        return torch.softmax(self.net(x), dim=-1)

**Wed Jan 28***  —  Synthetic context features + training data.*

The Phase-1 CSV doesn't have an explicit environmental context column — every sample is implicitly highway. To bootstrap, I wrote a small data-augmentation pass: take each row, attach a synthetic context vector based on whether n_prox is high (treat as urban) or n_speed exceeds 0.85 with low n_prox (treat as open highway) or n_wave is high with moderate n_speed (treat as weather-affected). It's a heuristic, not ground truth, but it gives the agent something to learn against. Logged the heuristic in a docstring in the loader so future me knows it's there.

After tagging, the dataset has ~1100 highway samples, ~500 urban, ~200 weather. Imbalanced but workable. Will add class weights to the loss if it becomes a problem.

**Thu Jan 29***  —  Training loop + first run.*

Wrote scripts/train_weight_agent.py. Training loop is nothing exotic: AdamW, lr=1e-3, weight_decay=1e-4, KL divergence between predicted weight vector and the regime-appropriate hand-picked target. Ran 200 epochs in about 4 minutes on the lab machine (CPU only — the lab GPU was tied up by another VIP team's vision project). Loss curve drops cleanly from 0.32 to 0.04 over the run, no oscillation. Held-out highway sample predicts (0.30, 0.10, 0.45, 0.15), which is within 0.01 L1 of the highway target (0.30, 0.10, 0.45, 0.15). Pass criterion met.

One thing I noticed: the predicted weights for borderline samples (where the heuristic context-tagging is shaky) are basically just averages of the three target vectors. That's the model being honest about uncertainty, which is a nice property to have here. I'll come back to this when we plug in real environments.

**Fri Jan 30***  —  Eval on perturbations.*

Wrote a small eval script that takes a single telemetry row and sweeps the context vector through 1000 random perturbations (small noise on each context dim), recording the predicted weight vector for each. Then plot the distribution of predicted weights as a violin plot for a high-prox urban sample. The distribution is tight in the (w_prox, w_wave) directions and broader in the (w_speed, w_accel) directions, which makes physical sense — urban driving is dominated by proximity / waviness considerations and less so by speed.

Saved the figure as eval/violin_perturb_urban.png to use in the next sync. First proper diagnostic plot of the term.

**Sat Jan 31***  —  Other-coursework day.*

Off the project today. EECE 442 has a problem set due Monday and I had to actually focus on it.

**Sun Feb 1***  —  Light reading.*

Picked up where I left off in Vaswani for the gate design. Also skimmed the highway-env source again to make sure I understand the observation format end-to-end before I plug the agent in.

**Mon Feb 2***  —  Sync.*

Sync with Daher. Showed him the loss curve and the violin plot. He liked both. Two new asks: (a) plot the gradient norm during training as a sanity check that no layer is starving, and (b) start thinking about how the agent's softmax output should interact with the AI score during simulation — i.e. the inference path. I had been assuming we'd compute AI from the predicted weights at every step, but he pointed out that recomputing on every step at 15 Hz might be overkill. Suggested computing it on each new state regardless, but caching when state delta is small. Recorded as a TODO for later — premature optimization for now.

## **Reflection**

First week of real coding for term 2 and it landed cleanly. The softmax output is the right design choice for now — interpretable, simplex-constrained, and naturally regularized. The hand-picked targets are the obvious limitation but the violin plot already shows the model behaving sensibly under uncertainty, which is encouraging.

## **Plan for next week**

Wire the agent into the highway-env simulation loop so it produces predicted weights at each step, feed those into the AggressivenessModel, and log the resulting AI score over a full episode. Compare against the Phase-1 fixed-weight baseline on the same episode (replay the same seed). Also: instrument gradient norms during training, per Daher's ask.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
