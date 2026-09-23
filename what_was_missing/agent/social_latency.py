"""
Social Latency reward.

RECREATED BY CLAUDE (Anthropic), September 2026, from the final report
(Sections 3.2 and 4) and the hyperparameter addendum. This is not the original
project code and was not used to produce any result in those reports.

Final report form:
    R_total = (1 - AI_self) - lambda * max(0, dAI_neighbors)
with lambda = 2. AI values here are on the 0 to 100 scale and divided by 100
inside the reward, so both terms live in [0, 1] per unit of lambda.

dAI_neighbors is not defined precisely in the reports. It is implemented as
the mean AI of vehicles within `radius_m` of the agent now, minus an
exponential moving average of that same quantity (the recent baseline). The
max(0, .) clip means the agent is never rewarded for calming neighbors it did
not agitate, and never blamed for a drop.

LagrangianSocialLatency follows the addendum: treat
    max E[1 - AI_self]  subject to  E[harm to neighbors] <= tau
with tau = 2 AI points, and let lambda be the Lagrange multiplier, updated by
dual ascent once per episode: lambda <- max(0, lambda + eta * (mean_harm - tau)).
"""
import math


class SocialLatencyReward:
    def __init__(self, lam: float = 2.0, radius_m: float = 50.0, baseline_tau_steps: float = 15.0):
        self.lam = lam
        self.radius_m = radius_m
        self.ema_rate = 1.0 - math.exp(-1.0 / baseline_tau_steps)
        self.baseline = None
        self.episode_harm = []

    def reset(self):
        self.baseline = None
        self.episode_harm = []

    def neighbor_mean_ai(self, ego_xy, others):
        """others: iterable of (x, y, ai). Returns mean AI within radius, or None."""
        ex, ey = ego_xy
        vals = [ai for (x, y, ai) in others if math.hypot(x - ex, y - ey) <= self.radius_m]
        return sum(vals) / len(vals) if vals else None

    def __call__(self, ai_self, ego_xy, others):
        """Returns (reward, harm) where harm = max(0, dAI_neighbors) in AI points."""
        nb = self.neighbor_mean_ai(ego_xy, others)
        harm = 0.0
        if nb is not None:
            if self.baseline is None:
                self.baseline = nb
            harm = max(0.0, nb - self.baseline)
            self.baseline += self.ema_rate * (nb - self.baseline)
        self.episode_harm.append(harm)
        reward = (1.0 - ai_self / 100.0) - self.lam * harm / 100.0
        return reward, harm


class LagrangianSocialLatency(SocialLatencyReward):
    def __init__(self, tau: float = 2.0, eta: float = 0.5, lam_init: float = 0.0, **kw):
        super().__init__(lam=lam_init, **kw)
        self.tau = tau
        self.eta = eta

    def end_episode(self):
        """Dual ascent step on lambda. Returns (lambda, mean harm this episode)."""
        mean_harm = sum(self.episode_harm) / len(self.episode_harm) if self.episode_harm else 0.0
        self.lam = max(0.0, self.lam + self.eta * (mean_harm - self.tau))
        self.episode_harm = []
        return self.lam, mean_harm
