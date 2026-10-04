"""Retrieval of normal windows by context, percentile score and explanation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.rag import Context, WindowIndex, explain, load_knowledge  # noqa: E402


def _index():
    # normal windows: motorway scores 30..39, jam scores 60..69; plus aggressive windows that must be ignored
    s = list(range(30, 40)) * 6 + list(range(60, 70)) * 6 + [99] * 60
    c = [Context("motorway", 120, "free")] * 60 + [Context("motorway", 120, "jam")] * 60 + [Context("motorway", 120, "free")] * 60
    lab = ["normal"] * 120 + ["aggressive"] * 60
    return WindowIndex(s, c, lab, k=200, k_min=50)


def test_percentile_is_relative_to_the_situation():
    idx = _index()
    pct_free, _, lvl, n = idx.score(65.0, Context("motorway", 120, "free"))
    pct_jam, _, _, _ = idx.score(65.0, Context("motorway", 120, "jam"))
    assert pct_free == 100.0 and n == 60          # 65 is above every normal free flow window
    assert 45 <= pct_jam <= 55                      # and ordinary in the jam
    assert lvl == ("road_type", "traffic_state", "manoeuvre", "weather")   # most specific level had enough windows


def test_fallback_when_situation_unseen():
    idx = _index()
    _, _, lvl, n = idx.score(50.0, Context("roundabout", 50, "dense"))
    assert lvl == () and n == 120                   # no roundabout reference: all normal windows


def test_aggressive_reference_windows_never_used():
    idx = _index()
    _, thr, _, _ = idx.score(50.0, Context("motorway", 120, "free"))
    assert thr < 40                                 # the 99s of the aggressive windows are not in the reference


def test_explain_cites_knowledge_file():
    kv, path = load_knowledge("spain")
    assert kv["following_s"] == "2" and path.endswith("spain.md")
    txt = explain(np.array([0.69, 0.03, 0.57, 0.13]), Context("motorway", 120, region="spain"), 97.0, 80.0, 125.0, 12.0,
                  ("road_type",), 200)
    assert "0.3 s time headway" in txt and "knowledge/spain.md" in txt and "flagged" in txt
