"""
ROS-independent logic for the nodes, kept separate so it can be tested without ROS.

RECREATED BY CLAUDE (Anthropic), September 2026, from the final report
(Section 2.6) and weekly reports 12 to 14. Not the original project code.

The index definitions below mirror what_was_missing/agent/aggressiveness.py.
They are copied rather than imported because an installed colcon package cannot
reach files outside the workspace; a test checks the two copies agree.
"""
import json
import statistics
from collections import defaultdict, deque

V_REF_MS = 150.0 / 3.6
A_REF = 5.0
GAP_REF = 50.0
WAVE_REF = 1.5
DENSITY_REF = 50.0
AGGRESSIVE_FROM = 70.0


def normalize_features(speed_ms, accel_ms2, gap_m, wave_m):
    n_speed = min(max(speed_ms / V_REF_MS, 0.0), 1.0)
    n_accel = min(abs(accel_ms2) / A_REF, 1.0)
    n_prox = max(0.0, 1.0 - gap_m / GAP_REF) if gap_m < GAP_REF else 0.0
    n_wave = min(abs(wave_m) / WAVE_REF, 1.0)
    return [n_speed, n_accel, n_prox, n_wave]


def context_vector(road_type, density_veh_km_lane, friction, slip):
    return [
        1.0 if road_type == "highway" else 0.0,
        1.0 if road_type == "urban" else 0.0,
        min(max(density_veh_km_lane / DENSITY_REF, 0.0), 1.0),
        min(max(friction, 0.0), 1.0),
        min(max(slip, 0.0), 1.0),
    ]


def ai_from_lists(features, weights):
    s = sum(f * w for f, w in zip(features, weights))
    return 100.0 * min(max(s, 0.0), 1.0)


class AgentRuntime:
    """Loads the TorchScript gate head once and scores one telemetry sample at a time."""

    def __init__(self, model_path, road_type="highway", friction=1.0, road_length_km=1.0,
                 lanes=3, activity_window_s=2.0):
        import torch
        self.torch = torch
        extra = {"provenance.json": ""}
        self.model = torch.jit.load(model_path, _extra_files=extra)
        self.model.eval()
        self.provenance = json.loads(extra["provenance.json"] or "{}")
        with torch.no_grad():
            self.regime_weights = self.model.regime_weights().tolist()
        self.road_type = road_type
        self.friction = friction
        self.lane_km = max(road_length_km * lanes, 1e-6)
        self.window = activity_window_s
        self.last_seen = {}

    def density(self, now):
        """Vehicles seen in the last window, per km per lane."""
        for vid in [v for v, t in self.last_seen.items() if now - t > self.window]:
            del self.last_seen[vid]
        return len(self.last_seen) / self.lane_km

    def process(self, vehicle_id, t, speed, accel, prox, wave, slip):
        self.last_seen[vehicle_id] = t
        feats = normalize_features(speed, accel, prox, wave)
        ctx = context_vector(self.road_type, self.density(t), self.friction, slip)
        with self.torch.no_grad():
            w, mix = self.model(self.torch.tensor([feats]), self.torch.tensor([ctx]))
        weights = w[0].tolist()
        return {
            "ai_score": ai_from_lists(feats, weights),
            "ai_weather": ai_from_lists(feats, self.regime_weights[2]),
            "mixture_weights": mix[0].tolist(),
            "weights": weights,
        }


class SummaryAggregator:
    """Per-vehicle AI history over a sliding window, reduced to population aggregates."""

    def __init__(self, window_s=2.0, aggressive_from=AGGRESSIVE_FROM):
        self.window = window_s
        self.aggressive_from = aggressive_from
        self.hist = defaultdict(deque)

    def add(self, vehicle_id, t, ai):
        self.hist[vehicle_id].append((t, ai))

    def tick(self, now):
        for vid in list(self.hist):
            q = self.hist[vid]
            while q and now - q[0][0] > self.window:
                q.popleft()
            if not q:
                del self.hist[vid]
        if not self.hist:
            return {"mean_ai": 0.0, "max_ai": 0.0, "n_vehicles": 0, "n_affected_neighbors": 0,
                    "aggressive_fraction": 0.0}
        latest = [q[-1][1] for q in self.hist.values()]
        means = [sum(a for _, a in q) / len(q) for q in self.hist.values()]
        ref = sum(means) / len(means)
        sd = statistics.pstdev(means) if len(means) > 1 else 0.0
        return {
            "mean_ai": sum(latest) / len(latest),
            "max_ai": max(latest),
            "n_vehicles": len(latest),
            "n_affected_neighbors": sum(1 for m in means if sd > 0 and m > ref + sd),
            "aggressive_fraction": sum(1 for a in latest if a >= self.aggressive_from) / len(latest),
        }


def load_telemetry_csv(path):
    """Rows grouped by time step: list of (t, [row dicts])."""
    import csv
    steps = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = float(r["t"])
            if not steps or steps[-1][0] != t:
                steps.append((t, []))
            steps[-1][1].append(r)
    return steps


def latency_stats(latencies_ms):
    if not latencies_ms:
        return {}
    s = sorted(latencies_ms)
    return {"n": len(s), "median_ms": s[len(s) // 2], "p99_ms": s[min(len(s) - 1, int(0.99 * len(s)))]}
