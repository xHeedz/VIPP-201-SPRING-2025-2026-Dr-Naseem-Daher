"""
Sequential labeling on cleaned data with the reference formula (Oct 2026 revision): configurable state, reward
and action space, so each can be compared on its own. env/labeling_env.py (the earlier environment) is unchanged.

Sequences come from data/clean/*.csv.gz (sequences_from_clean): per second reference terms (NaN = unknown),
car ahead flag (1 / 0 / NaN), label, environment, group (split unit), scenario.

State: any subset of STATE_FEATURES (all between about -1 and 1; unknown terms count 0 points):
  pts_<term>_run   weight x mean term since the start (points / 100 of the score), term = speed, accel2, prox2, wave
  pts_<term>_rec   same over the last 10 s
  score_run        reference score of the means since the start / 100 (unknown terms left out)
  score_rec        same, last 10 s
  prox_follow      weight x mean proximity term over the seconds with a car ahead only (the following clue)
  leader_share     share of seconds with a car seen ahead
  time             seconds watched / horizon
  env_highway, env_urban, env_weather   environment one hot
Actions: wait (1 s), optionally wait 5 s ("wait5"), then one label per class.
Reward: +1 correct; wrong = -cost[true][predicted] (default 1, "ordinal": 1 for neighbouring classes, 2 for
conservative vs aggressive); each waited second costs wait_cost + wait_growth x seconds already watched;
no label at the horizon = -1.
Risk reward (risk=True, sequences built with extended=True): each waited second costs x (1 + 5 danger of that
second); missing an aggressive driver costs 1 + 2 x the highest danger seen (danger: TTC under 4 s or headway under 1 s).
Extended state (EXT_FEATURES): IDM residual term so far and last 10 s, speed vs limit and vs traffic so far,
highest danger in the last 10 s.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from model.aggressiveness_model import REF_WEIGHTS, reference_terms

ENVIRONMENTS = ("highway", "urban", "weather")
TERMS = ("speed", "accel2", "prox2", "wave")
RECENT_S = 10
STATE_FEATURES = ([f"pts_{t}_run" for t in TERMS] + [f"pts_{t}_rec" for t in TERMS]
                  + ["score_run", "score_rec", "prox_follow", "leader_share", "time"]
                  + [f"env_{e}" for e in ENVIRONMENTS])
ORDER = {"conservative": 0, "normal": 1, "aggressive": 2}


EXT_FEATURES = ["idm_run", "idm_rec", "rel_limit_run", "rel_traffic_run", "danger_rec"]
COUNTRY_FEATURES = ["country_de"]          # 1 for German roads (autobahn scenario, highD), 0 otherwise


def danger_per_second(d):
    """0 to 1: the larger of the TTC danger ((1 - TTC / 4 s)^2) and very close following ((1 - headway / 1 s)^2 below
    1 s); unknown counts 0 (no evidence of danger)"""
    v = d["speed_kmh"].to_numpy(float) / 3.6
    gap = d["gap_m"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        thw = np.where(gap > 0, gap / np.maximum(v, 0.5), np.inf)
    close = np.where(thw < 1.0, (1.0 - thw) ** 2, 0.0)
    return np.maximum(np.nan_to_num(d["t_ttc"].to_numpy(float), nan=0.0), close)


def sequences_from_clean(d, min_speed_kmh=5.0, min_seconds=10, extended=False):
    """cleaned per second table -> list of sequences (dict: terms (T, 4), leader (T,), label, environment, ...);
    extended: also ext (T, 4) = IDM residual term, speed vs limit, speed vs traffic (signed), and danger (T,)"""
    d = d[d["speed_kmh"] >= min_speed_kmh].sort_values(["vid", "t"])
    if extended:
        from datasets.signals import add_signals
        d = add_signals(d)
        ext = np.c_[d["t_idm"].to_numpy(float), d["rel_limit"].clip(-0.5, 0.5).to_numpy(float),
                    d["rel_traffic"].clip(-0.5, 0.5).to_numpy(float), danger_per_second(d)]
    T = reference_terms(d["speed_kmh"], d["accel"], d["gap_m"], d["wave_m"], d["environment"].to_numpy())
    lead = d["leader"].to_numpy(float)
    out = []
    starts = np.flatnonzero(np.r_[True, d["vid"].to_numpy()[1:] != d["vid"].to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(d)]
    meta = d[["vid", "label", "environment", "group", "scenario"]].to_numpy()
    raw = d[["speed_kmh", "gap_m"]].to_numpy(float)
    for a, b in zip(starts, ends):
        if b - a < min_seconds:
            continue
        vid, label, env, group, scen = meta[a]
        out.append({"terms": T[a:b], "leader": lead[a:b], "label": label, "environment": env, "group": group,
                    "scenario": scen, "trip": str(vid), "speed_kmh": raw[a:b, 0], "gap_m": raw[a:b, 1],
                    "country": "de" if str(scen).startswith("autobahn") or str(vid).startswith("autobahn") else "other"})
        if extended:
            out[-1]["ext"] = ext[a:b]
    return out


def _wmean(x):
    """mean over known values per column, NaN when none known"""
    k = ~np.isnan(x)
    n = k.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(k, x, 0.0).sum(axis=0) / np.maximum(n, 1), np.nan)


def state_vector(terms, leader, n, horizon, environment, features, weights=REF_WEIGHTS, ext=None, country=None):
    seen, lead = terms[:n], leader[:n]
    w = np.asarray(weights, float)
    run, rec = _wmean(seen), _wmean(seen[-RECENT_S:])
    vals = {}
    for i, t in enumerate(TERMS):
        vals[f"pts_{t}_run"] = 0.0 if np.isnan(run[i]) else w[i] * run[i]
        vals[f"pts_{t}_rec"] = 0.0 if np.isnan(rec[i]) else w[i] * rec[i]
    for name, m in (("score_run", run), ("score_rec", rec)):
        k = ~np.isnan(m)
        vals[name] = float((w[k] * m[k]).sum() / w[k].sum()) if k.any() else 0.0
    fol = seen[lead == 1, 2]
    vals["prox_follow"] = float(w[2] * fol.mean()) if len(fol) and not np.all(np.isnan(fol)) else 0.0
    vals["leader_share"] = float(np.nansum(lead == 1) / n)
    vals["time"] = n / horizon
    for e in ENVIRONMENTS:
        vals[f"env_{e}"] = 1.0 if environment == e else 0.0
    vals["country_de"] = 1.0 if country == "de" else 0.0
    if ext is not None:
        x = ext[:n]
        er, ec = _wmean(x), _wmean(x[-RECENT_S:])
        vals["idm_run"] = 0.0 if np.isnan(er[0]) else float(er[0])
        vals["idm_rec"] = 0.0 if np.isnan(ec[0]) else float(ec[0])
        vals["rel_limit_run"] = 0.0 if np.isnan(er[1]) else float(er[1])
        vals["rel_traffic_run"] = 0.0 if np.isnan(er[2]) else float(er[2])
        vals["danger_rec"] = float(np.nanmax(x[-RECENT_S:, 3])) if len(x) else 0.0
    return np.array([vals[f] for f in features], dtype=np.float32)


class LabelingRefEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sequences, classes=("conservative", "normal", "aggressive"), features=STATE_FEATURES,
                 horizon=60, wait_cost=0.002, wait_growth=0.0, cost="flat", wait5=False, balance=True, risk=False):
        self.sequences = list(sequences)
        self.classes = tuple(classes)
        self.features = list(features)
        self.horizon, self.wait_cost, self.wait_growth = int(horizon), float(wait_cost), float(wait_growth)
        self.cost, self.wait5, self.balance, self.risk = cost, bool(wait5), balance, bool(risk)
        self.waits = (1, 5) if self.wait5 else (1,)
        self.by_class = {c: [i for i, s in enumerate(self.sequences) if s["label"] == c] for c in self.classes}
        self.action_space = spaces.Discrete(len(self.waits) + len(self.classes))
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(len(self.features),), dtype=np.float32)

    def wrong_cost(self, true, pred):
        if self.cost == "ordinal":
            return 2.0 if abs(ORDER[true] - ORDER[pred]) == 2 else 1.0
        return 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "seq" in options:
            self.i, self.t0 = int(options["seq"]), int(options.get("t0", 0))
        else:
            if self.balance:
                cls = [c for c in self.classes if self.by_class[c]]
                pool = self.by_class[cls[self.np_random.integers(len(cls))]]
                self.i = pool[self.np_random.integers(len(pool))]
            else:
                self.i = int(self.np_random.integers(len(self.sequences)))
            T = len(self.sequences[self.i]["terms"])
            self.t0 = int(self.np_random.integers(max(T - self.horizon, 0) + 1))
        s = self.sequences[self.i]
        self.n = 1
        self.max_n = min(self.horizon, len(s["terms"]) - self.t0)
        return self._obs(), {}

    def _obs(self):
        s = self.sequences[self.i]
        return state_vector(s["terms"][self.t0:], s["leader"][self.t0:], self.n, self.horizon, s["environment"],
                            self.features, ext=s["ext"][self.t0:] if "ext" in s else None, country=s.get("country"))

    def _danger(self, j):
        s = self.sequences[self.i]
        return float(s["ext"][self.t0 + j, 3]) if self.risk and "ext" in s else 0.0

    def step(self, action):
        s = self.sequences[self.i]
        info = {"seconds": self.n, "true": s["label"]}
        if action < len(self.waits):
            k = self.waits[action]
            if self.n >= self.max_n:
                info["predicted"] = None
                return self._obs(), -1.0, True, False, info
            steps = min(k, self.max_n - self.n)
            r = -sum((self.wait_cost + self.wait_growth * (self.n + j)) * (1.0 + 5.0 * self._danger(self.n + j))
                     for j in range(steps))
            self.n += steps
            return self._obs(), r, False, False, info
        pred = self.classes[action - len(self.waits)]
        info["predicted"] = pred
        r = 1.0 if pred == s["label"] else -self.wrong_cost(s["label"], pred)
        if self.risk and s["label"] == "aggressive" and pred != "aggressive" and "ext" in s:
            seen_danger = float(np.max(s["ext"][self.t0:self.t0 + self.n, 3]))
            r = -(1.0 + 2.0 * seen_danger)                # missing a dangerous driver costs up to 3
        return self._obs(), r, True, False, info


def run_policy(model, env, n):
    preds, secs = [], []
    for i in range(n):
        obs, _ = env.reset(options={"seq": i, "t0": 0})
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(a))
        preds.append(info["predicted"])
        secs.append(info["seconds"])
    return preds, secs


def evaluate(seqs, preds, secs):
    d = pd.DataFrame({"true": [s["label"] for s in seqs], "pred": preds, "seconds": secs,
                      "scenario": [s["scenario"] for s in seqs], "group": [s["group"] for s in seqs]})
    acc = d.assign(ok=d["true"] == d["pred"]).groupby("true")["ok"].mean()
    return d, {"balanced_acc": float(acc.mean()), **{f"acc_{c}": float(v) for c, v in acc.items()},
               "seconds": float(d["seconds"].mean()), "decided_at_1s": float((d["seconds"] == 1).mean())}
