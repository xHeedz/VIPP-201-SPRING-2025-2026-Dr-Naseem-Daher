"""
DynamicWeightAgent: the Aggressiveness Index with its weights learned per
environment.

Last semester (experiments/weight_agent/train_strict.py) the agent learned
its weights by matching ground-truth scores in generated telemetry
(supervised regression, mean squared error). The class was defined inside
that script; it now lives here so the same agent can be reused.

This semester it is extended with one learned threshold per environment,
the score above which a driver counts as aggressive, so it can also learn
from real drivers labeled normal or aggressive (UAH-DriveSet), through
fit_labels (supervised classification, logistic loss).

  weights      softmax(raw_weights[environment])       4 shares that add up to 1
  score        100 * sum(weights * features)           0 to 100
  P(aggressive) = sigmoid((score - threshold[environment]) / tau)

Features are the four index features in model/aggressiveness_model.py
(index_features): speed squared, acceleration, proximity squared, waviness.
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENVIRONMENTS = ("highway", "urban", "weather")


class DynamicWeightAgent(nn.Module):
    def __init__(self):
        super(DynamicWeightAgent, self).__init__()
        torch.manual_seed(42)
        self.raw_weights_highway = nn.Parameter(torch.zeros(4))
        self.raw_weights_urban = nn.Parameter(torch.zeros(4))
        self.raw_weights_weather = nn.Parameter(torch.zeros(4))
        # added this semester: aggressive threshold per environment and its sharpness
        self.thresholds = nn.Parameter(torch.full((len(ENVIRONMENTS),), 70.0))
        self.log_tau = nn.Parameter(torch.tensor(math.log(5.0)))

    def get_weights(self, env_type):
        if env_type == 'highway': return F.softmax(self.raw_weights_highway, dim=0)
        elif env_type == 'urban': return F.softmax(self.raw_weights_urban, dim=0)
        elif env_type == 'weather': return F.softmax(self.raw_weights_weather, dim=0)
        raise ValueError(f"unknown environment {env_type!r}")

    def forward(self, env_type, trial_metrics):
        weights = self.get_weights(env_type)
        return torch.sum(weights * trial_metrics, dim=-1)

    # ---------------------------------------------------------------- this semester
    def score(self, env_type, metrics):
        """Aggressiveness score from 0 to 100."""
        return 100.0 * self.forward(env_type, metrics)

    def prob_aggressive(self, env_type, metrics):
        i = ENVIRONMENTS.index(env_type)
        return torch.sigmoid((self.score(env_type, metrics) - self.thresholds[i]) / self.log_tau.exp())

    def fit_labels(self, metrics, labels, env_types, epochs=600, lr=0.05):
        """Learn weights and thresholds from labeled windows.
        metrics (N, 4), labels (N,) with 1 = aggressive, env_types (N,) environment names."""
        X = torch.tensor(np.asarray(metrics), dtype=torch.float32)
        y = torch.tensor(np.asarray(labels), dtype=torch.float32)
        envs = np.asarray(env_types)
        present = [e for e in ENVIRONMENTS if (envs == e).any()]
        masks = {e: torch.tensor(envs == e) for e in present}
        # start each threshold in the middle of its scores so every weight gets a gradient
        with torch.no_grad():
            spread = []
            for e in present:
                s = self.score(e, X[masks[e]])
                self.thresholds[ENVIRONMENTS.index(e)] = s.median()
                spread.append(float(s.std()))
            self.log_tau.fill_(math.log(max(float(np.mean(spread)) / 2, 1.0)))
        pos = y.mean().clamp(0.05, 0.95)
        balance = torch.where(y > 0.5, 0.5 / pos, 0.5 / (1 - pos))
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            loss = 0.0
            for e in present:
                m = masks[e]
                z = (self.score(e, X[m]) - self.thresholds[ENVIRONMENTS.index(e)]) / self.log_tau.exp()
                loss = loss + (F.binary_cross_entropy_with_logits(z, y[m], reduction="none") * balance[m]).sum()
            loss = loss / len(y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                self.thresholds.clamp_(0.0, 100.0)
                self.log_tau.clamp_(math.log(0.5), math.log(50.0))
        return float(loss)

    # ---------------------------------------------------------------- numpy helpers for scripts
    def weights_np(self, env_type):
        with torch.no_grad():
            return self.get_weights(env_type).numpy().copy()

    def threshold(self, env_type):
        return float(self.thresholds[ENVIRONMENTS.index(env_type)].detach())

    def ai(self, features, env_type):
        """Scores for a (N, 4) or (4,) numpy array of index features."""
        return 100.0 * np.asarray(features) @ self.weights_np(env_type)

    def to_dict(self):
        return {"weights": {e: [round(float(v), 4) for v in self.weights_np(e)] for e in ENVIRONMENTS},
                "thresholds": {e: round(self.threshold(e), 2) for e in ENVIRONMENTS},
                "tau": round(float(self.log_tau.exp().detach()), 3),
                "feature_order": ["speed^2", "accel", "prox^2", "wave"]}

    def save(self, path, extra=None):
        with open(path, "w") as f:
            json.dump({**self.to_dict(), **(extra or {})}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        agent = cls()
        with torch.no_grad():
            for e in ENVIRONMENTS:
                w = torch.tensor(d["weights"][e], dtype=torch.float32).clamp_min(1e-8)
                getattr(agent, f"raw_weights_{e}").copy_(w.log())
                agent.thresholds[ENVIRONMENTS.index(e)] = d["thresholds"][e]
            agent.log_tau.fill_(math.log(d.get("tau", 5.0)))
        return agent
