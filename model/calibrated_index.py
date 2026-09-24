"""
Calibrated Aggressiveness Index.

Same structure as the closed-form index in model/aggressiveness_model.py, with
the four weights and the aggressive threshold learned from labeled real data
instead of chosen by hand:

  features  phi = (n_speed^2, n_accel, n_prox^2, n_wave)     (normalization as in AggressivenessModel)
  index     AI  = 100 * sum_i w_i * phi_i,   w on the simplex (w_i >= 0, sum w_i = 1)
  decision  P(aggressive) = sigmoid((AI - theta) / tau)

One weight vector and one threshold per road context (motorway, secondary),
which is the dynamic-weight idea: the context decides how much each feature
counts. Training is supervised learning (logistic regression through the
index), not reinforcement learning.
"""
import json

import numpy as np

CONTEXTS = ("motorway", "secondary")
ORIGINAL_WEIGHTS = (0.5, 0.2, 0.8, 0.4)       # AggressivenessModel
ORIGINAL_THRESHOLD = 70.0


def phi(speed_kmh, accel_ms2, gap_m, wave_m):
    """Vectorized features, identical to AggressivenessModel.normalize plus its squares.
    gap_m <= 0 means no vehicle ahead."""
    speed_kmh, accel_ms2, gap_m, wave_m = (np.asarray(x, dtype=float) for x in (speed_kmh, accel_ms2, gap_m, wave_m))
    ns = np.minimum(speed_kmh / 150.0, 1.0)
    na = np.minimum(np.abs(accel_ms2) / 5.0, 1.0)
    npx = np.where((gap_m > 0) & (gap_m <= 50.0), 1.0 - gap_m / 50.0, 0.0)
    nw = np.minimum(np.abs(wave_m) / 1.5, 1.0)
    return np.stack([ns ** 2, na, npx ** 2, nw], axis=-1)


def original_ai(features):
    """The closed-form index exactly as AggressivenessModel computes it."""
    return np.minimum(100.0 * features @ np.array(ORIGINAL_WEIGHTS), 100.0)


class CalibratedIndex:
    def __init__(self, weights=None, thresholds=None, tau=5.0):
        base = np.array(ORIGINAL_WEIGHTS) / sum(ORIGINAL_WEIGHTS)
        self.weights = {c: np.array(weights[c]) if weights else base.copy() for c in CONTEXTS}
        self.thresholds = {c: float(thresholds[c]) if thresholds else ORIGINAL_THRESHOLD for c in CONTEXTS}
        self.tau = tau

    def ai(self, features, context):
        return 100.0 * np.asarray(features) @ self.weights[context]

    def prob_aggressive(self, features, context):
        return 1.0 / (1.0 + np.exp(-(self.ai(features, context) - self.thresholds[context]) / self.tau))

    def fit(self, features, labels, contexts, epochs=600, lr=0.05, seed=0):
        """features (N,4), labels (N,) 1 = aggressive, contexts (N,) strings."""
        import torch
        torch.manual_seed(seed)
        X = torch.tensor(np.asarray(features), dtype=torch.float32)
        y = torch.tensor(np.asarray(labels), dtype=torch.float32)
        ctx = torch.tensor([CONTEXTS.index(c) for c in contexts])
        logits = torch.nn.Parameter(torch.tensor(np.log(np.stack([self.weights[c] for c in CONTEXTS]) + 1e-6),
                                                 dtype=torch.float32))
        # start the threshold in the middle of the data and the slope at its spread, so the
        # sigmoid is not saturated at the start and every weight receives a gradient
        with torch.no_grad():
            ai0 = 100.0 * (X * torch.softmax(logits, -1)[ctx]).sum(-1)
        th0 = [float(ai0[ctx == i].median()) if (ctx == i).any() else self.thresholds[c]
               for i, c in enumerate(CONTEXTS)]
        theta = torch.nn.Parameter(torch.tensor(th0, dtype=torch.float32))
        log_tau = torch.nn.Parameter(torch.tensor(np.log(max(float(ai0.std()) / 2, 1.0)), dtype=torch.float32))
        opt = torch.optim.Adam([logits, theta, log_tau], lr=lr)
        pos = y.mean().clamp(0.05, 0.95)
        sample_w = torch.where(y > 0.5, 0.5 / pos, 0.5 / (1 - pos))     # balance the two classes
        for _ in range(epochs):
            w = torch.softmax(logits, -1)[ctx]
            ai = 100.0 * (X * w).sum(-1)
            z = (ai - theta[ctx]) / log_tau.exp()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * sample_w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                theta.clamp_(0.0, 100.0)
                log_tau.clamp_(np.log(0.5), np.log(50.0))
        w = torch.softmax(logits, -1).detach().numpy()
        seen = set(contexts)
        for i, c in enumerate(CONTEXTS):
            if c in seen:
                self.weights[c] = w[i]
                self.thresholds[c] = float(theta[i].detach())
        self.tau = float(log_tau.exp().detach())
        return float(loss.detach())

    def to_dict(self):
        return {"weights": {c: [round(float(v), 4) for v in self.weights[c]] for c in CONTEXTS},
                "thresholds": {c: round(self.thresholds[c], 2) for c in CONTEXTS},
                "tau": round(self.tau, 3),
                "feature_order": ["speed^2", "accel", "prox^2", "wave"]}

    def save(self, path, extra=None):
        with open(path, "w") as f:
            json.dump({**self.to_dict(), **(extra or {})}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["weights"], d["thresholds"], d.get("tau", 5.0))
