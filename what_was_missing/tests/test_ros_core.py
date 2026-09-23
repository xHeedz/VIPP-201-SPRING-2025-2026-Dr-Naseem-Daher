# Written by Claude (Anthropic), September 2026, as part of what_was_missing/. See ../README.md.
import os
import random

from agent import aggressiveness as ref
from vipp_aggressiveness import core

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.path.join(ROOT, "model", "gate_v2.pt")
CSV = os.path.join(ROOT, "results", "telemetry_highway.csv")


def test_core_copy_matches_reference_definitions():
    rng = random.Random(0)
    for _ in range(500):
        args = (rng.uniform(0, 50), rng.uniform(-6, 6), rng.uniform(0, 120), rng.uniform(0, 3))
        assert core.normalize_features(*args) == ref.normalize_features(*args)
        ctx = (rng.choice(["highway", "urban"]), rng.uniform(0, 80), rng.uniform(0, 1), rng.uniform(0, 2))
        assert core.context_vector(*ctx) == ref.context_vector(*ctx)


def _run(slice_rows):
    rt = core.AgentRuntime(MODEL)
    return [rt.process(r["vehicle_id"], float(r["t"]), float(r["speed"]), float(r["accel"]),
                       float(r["prox"]), float(r["wave"]), float(r["slip"]))["ai_score"] for r in slice_rows]


def test_replay_is_bit_identical():
    """Same 100-message slice through two fresh runtimes gives identical scores."""
    rows = [r for _, step in core.load_telemetry_csv(CSV) for r in step][:100]
    assert _run(rows) == _run(rows)


def test_summary_eviction_and_aggregates():
    agg = core.SummaryAggregator(window_s=2.0)
    for i in range(10):
        agg.add(f"v{i}", 0.0, 20.0)
    agg.add("hot", 0.0, 95.0)
    s = agg.tick(1.0)
    assert s["n_vehicles"] == 11 and s["max_ai"] == 95.0
    assert s["n_affected_neighbors"] == 1 and abs(s["aggressive_fraction"] - 1 / 11) < 1e-9
    assert agg.tick(3.5)["n_vehicles"] == 0          # everything older than 2 s is gone
