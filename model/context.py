"""
Traffic context for the Aggressiveness Index.

The index looks only at a vehicle's own motion. In dense stop-and-go traffic every
vehicle brakes, accelerates and follows closely because the traffic forces it to,
so every vehicle scores high. The context score asks instead: how would this vehicle
score if the traffic around it behaved like a normal driver?

    context score = normal_mean + (own score - mean score of the other vehicles within radius_m)

normal_mean is the mean score the agent gives normal UAH-DriveSet drivers
(saved by scripts/train_uah.py in data/dynamic_weight_agent.json). A vehicle that
behaves like the traffic around it gets normal_mean; one that brakes harder or
follows closer than its neighbours keeps the difference. A vehicle with no other
vehicle within radius_m keeps its own score.

Example: neighbours average 30, the vehicle scores 33, normal UAH drivers average 9:
context score = 9 + (33 - 30) = 12.

scorer_for() gives every script the same scoring setup: the trained agent when
data/dynamic_weight_agent.json exists, otherwise the original hand-set index.
"""
import json
import os

import numpy as np
import pandas as pd

RADIUS_M = 100.0
ORIGINAL_NORMAL_MEAN = 52.5      # middle of the original normal band (35 to 70), if no UAH summary exists


def context_adjusted(df, score, normal_mean, radius_m=RADIUS_M):
    """Context score for every row of df (columns t and pos, one row per vehicle and time).
    score is the own score of each row, in the same order."""
    t = df["t"].to_numpy()
    pos = df["pos"].to_numpy(dtype=float)
    s = np.asarray(score, dtype=float)
    out = s.copy()
    order = np.lexsort((pos, t))                     # sort by time, then position
    t_s, pos_s, s_s = t[order], pos[order], s[order]
    starts = np.flatnonzero(np.r_[True, t_s[1:] != t_s[:-1]])
    ends = np.r_[starts[1:], len(t_s)]
    res = np.empty(len(s_s))
    for a, b in zip(starts, ends):                   # one time step at a time
        p, v = pos_s[a:b], s_s[a:b]
        cs = np.r_[0.0, np.cumsum(v)]
        lo = np.searchsorted(p, p - radius_m, side="left")
        hi = np.searchsorted(p, p + radius_m, side="right")
        n_other = hi - lo - 1                        # vehicles within radius, not counting itself
        others_mean = np.where(n_other > 0, (cs[hi] - cs[lo] - v) / np.maximum(n_other, 1), np.nan)
        res[a:b] = np.where(n_other > 0, normal_mean + v - others_mean, v)
    out[order] = res
    return out


def scorer_for(environment, data_dir):
    """(score function of (N, 4) index features, aggressive threshold, normal mean, name)."""
    agent_path = os.path.join(data_dir, "dynamic_weight_agent.json")
    if os.path.exists(agent_path):
        from model.dynamic_weight_agent import DynamicWeightAgent   # needs torch, so only loaded here
        agent = DynamicWeightAgent.load(agent_path)
        with open(agent_path) as f:
            saved = json.load(f)
        normal_mean = saved.get("normal_mean_score", {}).get(environment)
        if normal_mean is None:
            raise SystemExit("data/dynamic_weight_agent.json has no normal_mean_score: "
                             "run scripts/train_uah.py again (python main.py --only train)")
        return (lambda f: agent.ai(f, environment)), agent.threshold(environment), float(normal_mean), \
            f"trained dynamic weight agent ({environment})"
    from model.aggressiveness_model import original_score
    normal_mean = ORIGINAL_NORMAL_MEAN
    summary = os.path.join(data_dir, "uah_summary.json")
    if os.path.exists(summary):
        with open(summary) as f:
            normal_mean = json.load(f)["mean_score_by_behavior"]["score_original"].get("normal", normal_mean)
    return original_score, 70.0, float(normal_mean), "original hand-set index"


def rank_categories(df, col, top=0.1):
    """Display only: at every time step the highest `top` share of col is 'aggressive',
    the lowest `top` share 'conservative', the rest 'normal'. Says nothing about how
    aggressive anyone is in absolute terms."""
    pct = df.groupby("t")[col].rank(pct=True)
    return pd.Series(np.where(pct > 1 - top, "aggressive", np.where(pct <= top, "conservative", "normal")),
                     index=df.index)
