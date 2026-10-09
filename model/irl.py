"""
Per driver reward weights from car following trajectories (option b of docs/rl_pivot.md): maximum entropy
inverse RL with a short planning horizon, written as a conditional logit over acceleration choices.

Each second the driver picks an acceleration a from A_GRID and (in the model) holds it for H seconds; the leader
keeps its speed. Every candidate gets four features of the predicted H seconds:
    progress    speed reached / V_REF
    progress_sq (speed reached / V_REF)^2: with it the reward peaks at a desired speed
                v* = -theta_progress / (2 theta_progress_sq) x V_REF, learned per driver
    risk        (1 - smallest time headway / THW_MAX)^2 when 0 < headway <= THW_MAX, else 0 (no leader: 0)
    discomfort  a^2 / 9
P(a | state) = exp(theta . features(a)) / sum over the grid. theta = the driver's reward weights, fitted by
maximum likelihood on the observed choices (the mean acceleration over the next H seconds, nearest grid value),
with a pull toward a population theta0: loss = NLL / n + lam * |theta - theta0|^2.
An aggressive driver should have a higher desired speed and weigh risk and discomfort less.

Input: per second table of one vehicle (datasets/uah.py layout: t, speed_kmh, gap_m). Leader speed from the gap
change, v + dgap/dt, the same way for every source; seconds where it jumps (leader change) are dropped.
"""
import numpy as np
import torch

A_GRID = np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
H_S = 2.0
V_REF_MS = 130 / 3.6
THW_MAX = 3.0
NAMES = ("progress", "progress_sq", "risk", "discomfort")


def candidate_features(v, gap, v_lead, has_lead, a_grid=A_GRID, h=H_S, steps=4):
    """(n,) states -> (n, K, 4) features of every acceleration candidate."""
    v, gap, v_lead = (np.asarray(x, float)[:, None] for x in (v, gap, v_lead))
    has_lead = np.asarray(has_lead, bool)[:, None]
    a = a_grid[None, :]
    min_thw = np.full((v.shape[0], len(a_grid)), np.inf)
    for k in range(1, steps + 1):
        tau = h * k / steps
        vt = np.maximum(v + a * tau, 0.0)
        # distance travelled with the speed floored at 0 (a car does not reverse)
        t_stop = np.where(a < 0, np.minimum(-v / np.minimum(a, -1e-9), tau), tau)
        x_ego = v * t_stop + 0.5 * a * t_stop ** 2
        g = gap + v_lead * tau - x_ego
        thw = np.where(vt > 0.5, np.maximum(g, 0.0) / np.maximum(vt, 0.5), np.inf)
        min_thw = np.minimum(min_thw, thw)
    v_end = np.maximum(v + a * h, 0.0)
    risk = np.where(has_lead & (min_thw <= THW_MAX), (1 - np.minimum(min_thw, THW_MAX) / THW_MAX) ** 2, 0.0)
    risk = np.where(has_lead & (min_thw <= 0), 1.0, risk)
    p = v_end / V_REF_MS
    return np.stack([p, p ** 2, risk, np.broadcast_to(a ** 2 / 9.0, v_end.shape)], axis=-1)


def desired_speed_kmh(theta):
    """speed where theta_progress * p + theta_progress_sq * p^2 peaks (nan when it has no maximum)"""
    t1, t2 = theta[..., 0], theta[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t2 < 0, -t1 / (2 * t2) * V_REF_MS * 3.6, np.nan)


def decisions(ps, h=H_S, min_speed_kmh=5.0):
    """per second table of one vehicle -> (features (n, K, 3), chosen index (n,)). Uses seconds t with t + h seen."""
    t = ps["t"].to_numpy(float)
    v = ps["speed_kmh"].to_numpy(float) / 3.6
    gap = ps["gap_m"].to_numpy(float)
    hh = int(round(h))
    if len(t) <= hh + 1:
        return np.zeros((0, len(A_GRID), 4)), np.zeros(0, int)
    i = np.arange(len(t) - hh)
    ok = np.isclose(t[i + hh] - t[i], h) & (v[i] * 3.6 >= min_speed_kmh)
    has_lead = gap[i] > 0
    nxt = np.minimum(i + 1, len(t) - 1)
    v_lead = np.where(has_lead & (gap[nxt] > 0), v[i] + (gap[nxt] - gap[i]) / np.maximum(t[nxt] - t[i], 1e-9), v[i])
    ok &= np.abs(v_lead - v[i]) < 10.0           # gap jump: the leader changed
    a_obs = (v[i + hh] - v[i]) / h
    idx = np.abs(A_GRID[None, :] - a_obs[:, None]).argmin(axis=1)
    f = candidate_features(v[i], gap[i], v_lead, has_lead)
    return f[ok], idx[ok]


def fit(features, choice, theta0=None, lam=0.0, iters=200):
    """theta (4,) maximising the conditional logit likelihood of the choices, pulled toward theta0."""
    X = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(choice, dtype=torch.long)
    th0 = torch.zeros(features.shape[-1], dtype=torch.float64) if theta0 is None else torch.tensor(theta0, dtype=torch.float64)
    th = th0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([th], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = X @ th
        loss = torch.nn.functional.cross_entropy(logits, y) + lam * ((th - th0) ** 2).sum()
        loss.backward()
        return loss

    if len(y):
        opt.step(closure)
    return th.detach().numpy()


def log_likelihood(features, choice, theta):
    logits = features @ np.asarray(theta)
    logits = logits - logits.max(axis=1, keepdims=True)
    lp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return float(lp[np.arange(len(choice)), choice].sum())
