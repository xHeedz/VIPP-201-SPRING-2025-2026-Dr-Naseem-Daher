import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.context import context_adjusted, rank_categories  # noqa: E402


def test_context_score_matches_the_worked_example():
    # neighbours within 100 m average 30, the vehicle scores 33, normal drivers average 9
    df = pd.DataFrame({"t": [0.0] * 4, "pos": [0.0, 20.0, 40.0, 60.0]})
    out = context_adjusted(df, [33.0, 28.0, 30.0, 32.0], normal_mean=9.0)
    assert np.isclose(out[0], 9.0 + 33.0 - 30.0)


def test_vehicle_like_its_traffic_scores_the_normal_mean_and_isolated_keeps_its_own():
    df = pd.DataFrame({"t": [0.0, 0.0, 0.0, 0.0, 1.0], "pos": [0.0, 10.0, 20.0, 500.0, 0.0]})
    out = context_adjusted(df, [40.0, 40.0, 40.0, 25.0, 17.0], normal_mean=9.0)
    assert np.allclose(out[:3], 9.0)
    assert out[3] == 25.0 and out[4] == 17.0        # nobody within 100 m, or alone at t = 1


def test_neighbours_are_only_taken_from_the_same_moment():
    df = pd.DataFrame({"t": [0.0, 1.0], "pos": [0.0, 5.0]})
    assert np.allclose(context_adjusted(df, [50.0, 10.0], normal_mean=9.0), [50.0, 10.0])


def test_rank_colours_top_and_bottom_tenth_at_each_moment():
    df = pd.DataFrame({"t": [0.0] * 20, "x": np.arange(20.0)})
    cats = rank_categories(df, "x")
    assert (cats == "aggressive").sum() == 2 and (cats == "conservative").sum() == 2
    assert cats.iloc[-1] == "aggressive" and cats.iloc[0] == "conservative"
