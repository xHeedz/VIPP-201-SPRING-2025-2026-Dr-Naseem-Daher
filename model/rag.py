"""
Retrieval for context: score a window relative to normal windows recorded in the same situation.

    window + Context -> WindowIndex.query -> k nearest NORMAL reference windows in a matching situation
                     -> percentile of the window's index score among them -> context score (0 to 100)
    knowledge/<region>.md -> explain() only, never the score

Why the neighbours are chosen by context and not by the driving features: a window's neighbours in
feature space drive like it does, so its percentile among them would sit near 50 for every driver.
The question is "how unusual is this driving for this situation", so the situation picks the
reference set and the score is compared inside it.

Matching, most specific first, falling back when fewer than k_min reference windows match:
    road_type + traffic_state + manoeuvre + weather -> road_type + traffic_state -> road_type -> all
inside a match, the k nearest by speed limit (and density when both have it).
FAISS is not needed at this size (thousands to about 100k windows): numpy distances are exact.
"""
import os
import re
from dataclasses import asdict, dataclass

import numpy as np

KNOWLEDGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge"))
LEVELS = [("road_type", "traffic_state", "manoeuvre", "weather"), ("road_type", "traffic_state"), ("road_type",), ()]


@dataclass
class Context:
    road_type: str                 # motorway, secondary, freeway, arterial, downtown, intersection, roundabout, merge
    speed_limit_kmh: float
    traffic_state: str = "unknown"   # free, dense, jam, unknown
    manoeuvre: str = "cruise"        # cruise, lane_change
    weather: str = "clear"           # clear, rain
    region: str = "unknown"          # spain, united_states, greece, lebanon, germany, simulation
    vehicle_class: str = "car"
    density: float = float("nan")    # vehicles per km per lane when known

    def as_dict(self):
        return asdict(self)


class WindowIndex:
    """Reference windows (index score + context) to retrieve from."""

    def __init__(self, scores, contexts, labels=None, k=200, k_min=50):
        self.scores = np.asarray(scores, dtype=float)
        self.ctx = list(contexts)
        self.labels = np.asarray(labels) if labels is not None else np.array(["normal"] * len(self.scores))
        self.k, self.k_min = k, k_min
        self.cols = {f: np.array([getattr(c, f) for c in self.ctx]) for f in ("road_type", "traffic_state", "manoeuvre", "weather")}
        self.limit = np.array([c.speed_limit_kmh for c in self.ctx], dtype=float)
        self.density = np.array([c.density for c in self.ctx], dtype=float)
        self.normal = self.labels == "normal"
        self._cache = {}

    def _pool(self, ctx):
        """indices of normal reference windows at the most specific level with >= k_min, and that level"""
        for lvl in LEVELS:
            key = tuple(getattr(ctx, f) for f in lvl)
            if key not in self._cache:
                m = self.normal.copy()
                for f in lvl:
                    m &= self.cols[f] == getattr(ctx, f)
                self._cache[key] = np.flatnonzero(m)
            ix = self._cache[key]
            if len(ix) >= self.k_min:
                return ix, lvl
        return np.flatnonzero(self.normal), ()

    def query(self, ctx):
        ix, lvl = self._pool(ctx)
        d = np.abs(self.limit[ix] - ctx.speed_limit_kmh) / 30.0
        if np.isfinite(ctx.density):
            dd = np.abs(self.density[ix] - ctx.density) / 20.0
            d = d + np.where(np.isfinite(dd), dd, 0.0)
        if len(ix) > self.k:
            near = np.argpartition(d, self.k)[:self.k]
            ix = ix[near]
        return ix, lvl

    def score(self, raw_score, ctx):
        """(context score = percentile of raw_score among the neighbours, neighbours' 95th percentile, level)"""
        ix, lvl = self.query(ctx)
        ref = self.scores[ix]
        pct = 100.0 * (np.sum(ref < raw_score) + 0.5 * np.sum(ref == raw_score)) / max(len(ref), 1)
        return pct, float(np.percentile(ref, 95)) if len(ref) else float("nan"), lvl, len(ref)

    def score_many(self, raw_scores, contexts):
        out = np.empty(len(raw_scores))
        for i, (s, c) in enumerate(zip(raw_scores, contexts)):
            out[i] = self.score(s, c)[0]
        return out


# ── explanation ─────────────────────────────────────────────────────────────
def load_knowledge(region, folder=KNOWLEDGE_DIR):
    path = os.path.join(folder, f"{region}.md")
    if not os.path.exists(path):
        return None, None
    kv = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^([a-z_]+):\s*(.+)$", line.strip())
            if m:
                kv[m.group(1)] = m.group(2)
            elif line.startswith("#"):
                break
    return kv, os.path.relpath(path, os.path.dirname(folder))


def explain(features, ctx, pct, threshold, speed_kmh, gap_m=None, lvl=(), n_ref=0):
    """One paragraph: what the score is relative to, which term drives it, and the local rule it is
    compared with (citing the knowledge file). features = (n_s^2, n_a, n_p^2, n_w) of the window."""
    from model.aggressiveness_model import WEIGHTS
    pts = 100 * np.asarray(features) * np.array(WEIGHTS)
    names = ["speed", "acceleration", "proximity", "lane offset"]
    top = int(np.argmax(pts))
    match = " + ".join(lvl) if lvl else "no matching situation (all normal windows)"
    text = [f"context score {pct:.0f} (percentile among {n_ref} normal windows matched on {match}; "
            f"{ctx.road_type}, limit {ctx.speed_limit_kmh:.0f} km/h, traffic {ctx.traffic_state}, {ctx.manoeuvre})."]
    text.append(f"largest term: {names[top]} ({pts[top]:.1f} of {pts.sum():.1f} points).")
    kv, path = load_knowledge(ctx.region)
    if top == 2 and gap_m and speed_kmh > 5:
        thw = gap_m / (speed_kmh / 3.6)
        if kv and "following_s" in kv:
            text.append(f"gap {gap_m:.1f} m at {speed_kmh:.0f} km/h = {thw:.1f} s time headway; local guidance "
                        f"{kv['following_s']} s ({path}: {kv.get('following_rule', '')}).")
        else:
            text.append(f"gap {gap_m:.1f} m at {speed_kmh:.0f} km/h = {thw:.1f} s time headway.")
    elif top == 0 and kv:
        key = {"motorway": "limit_motorway_kmh", "freeway": "limit_motorway_kmh", "secondary": "limit_rural_kmh"}.get(
            ctx.road_type, "limit_urban_kmh")
        text.append(f"mean speed {speed_kmh:.0f} km/h against the posted limit {ctx.speed_limit_kmh:.0f} km/h "
                    f"(default for this road type {kv.get(key, '?')} km/h, {path}).")
    if pct >= 95:
        text.append(f"flagged: above the 95th percentile of normal drivers in this situation (raw score {threshold:.1f}).")
    return " ".join(text)
