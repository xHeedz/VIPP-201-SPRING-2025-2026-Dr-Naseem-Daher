import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from fixtures import make_ngsim, make_uah  # noqa: E402

from datasets.ngsim import read_raw, trajectories  # noqa: E402
from datasets.uah import find_trips, load_trip, load_windows  # noqa: E402
from model.aggressiveness_model import AggressivenessModel  # noqa: E402
from model.calibrated_index import CalibratedIndex, original_ai, phi  # noqa: E402
from model.noise import KINDS, make_noise, noise_suite  # noqa: E402
from model.shockwave import shockwave_factor  # noqa: E402


def test_phi_matches_the_original_model():
    m = AggressivenessModel()
    rng = np.random.default_rng(1)
    for _ in range(300):
        a = (rng.uniform(0, 200), rng.uniform(-8, 8), rng.choice([0.0, rng.uniform(0.1, 120)]), rng.uniform(-3, 3))
        assert abs(m.get_ai_score(*a)[0] - original_ai(phi(*a))) < 1e-9


def test_uah_loader_and_calibration(tmp_path):
    root = make_uah(str(tmp_path))
    trips = find_trips(root)
    assert len(trips) == 6 * 7
    assert {t["behavior"] for t in trips} == {"normal", "aggressive", "drowsy"}
    ps = load_trip(trips[0])
    assert {"speed_kmh", "accel", "gap_m", "wave_m"} <= set(ps.columns) and len(ps) > 100
    win, report = load_windows(root)
    b = win[win["behavior"].isin(["normal", "aggressive"])]
    feats = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]
    m = CalibratedIndex()
    m.fit(b[feats].to_numpy(), (b["behavior"] == "aggressive").astype(int), b["road"], epochs=200)
    for c in ("motorway", "secondary"):
        assert abs(m.weights[c].sum() - 1) < 1e-5 and (m.weights[c] >= 0).all()
    p = np.array([m.prob_aggressive(x, c) for x, c in zip(b[feats].to_numpy(), b["road"])])
    y = (b["behavior"] == "aggressive").to_numpy()
    assert p[y].mean() > p[~y].mean()


@pytest.mark.parametrize("header", [True, False])
def test_ngsim_both_formats(tmp_path, header):
    path = make_ngsim(str(tmp_path / ("n.csv" if header else "n.txt")), header=header)
    raw = read_raw(path, "us-101" if header else None, minutes=1)
    tr = trajectories(raw)
    assert tr["t"].max() <= 60.0 + 1e-9
    assert (tr["speed"] >= 0).all() and tr["speed"].between(0, 40).all()
    assert (tr["gap"] >= 0).all() and (tr["gap"] > 0).mean() > 0.5
    lead_gap = tr[(tr["lane"] == 2)]["gap"]
    assert lead_gap[lead_gap > 0].median() == pytest.approx(60 * 0.3048 - 15 * 0.3048, rel=0.5)


def test_shockwave_blames_the_slow_free_leader():
    rows = []
    for t in range(5):
        rows += [(t, "S", 0, 300, 10, 0), (t, "F1", 0, 280, 10, -2), (t, "F2", 0, 260, 10, -1),
                 (t, "F3", 0, 240, 10, -1), (t, "A", 1, 500, 30, 0), (t, "B", 1, 470, 30, 0)]
    sf = shockwave_factor(pd.DataFrame(rows, columns=["t", "vehicle_id", "lane", "pos", "speed", "accel"]), v_ref=30)
    assert sf.loc["S", "shockwave_factor"] > 0.3
    assert sf.loc["A", "shockwave_factor"] == 0.0
    assert sf.loc["F2", "shockwave_factor"] < sf.loc["S", "shockwave_factor"]


@pytest.mark.parametrize("kind", KINDS)
def test_every_noise_kind_runs_and_zero_level_is_clean(kind):
    n = make_noise(kind, "speed", level=1.0, seed=0)
    out = [n(20.0, "v1") for _ in range(50)]
    assert all(np.isfinite(out))
    clean = make_noise(kind, "speed", level=0.0)
    assert clean(20.0, "v1") == 20.0


def test_delay_and_dropout_semantics():
    d = make_noise("delay", "gap", level=2)
    assert [d(x, "k") for x in [1, 2, 3, 4]] == [1, 1, 1, 2]
    drop = make_noise("dropout", "gap", level=9, seed=0)          # 90 % dropout
    vals = [drop(float(x), "k") for x in range(100)]
    assert vals[0] == 0.0 and len(set(vals)) < 30                 # mostly held values
    suite = noise_suite(["gaussian", "bias"], 1.0)
    assert set(suite) == {"speed", "accel", "gap", "lateral"}
