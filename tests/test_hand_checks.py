"""
Hand checked numbers (docs/hand_checks/hand_check_3_windows.md, scripts/hand_check.py) hard coded,
so any change to the index or to the feature extraction that moves a real window shows up here.

The first group needs no data: per second inputs of each window go through the reference index.
The second group re-reads the raw UAH and NGSIM files and is skipped when they are not next to the repo.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.aggressiveness_model import breakdown, index_features, label, original_score  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "28:9:2026"))

# (speed_kmh, accel_ms2, gap_m, wave_m) per GPS second of the window
UAH_NORMAL = [(106.966667, 0.062209, 30.07, 0.040), (107.433333, 0.091870, 30.39, 0.030),
              (107.633333, 0.029278, 29.97, 0.000), (107.633333, -0.048913, 30.03, 0.109),
              (107.333333, -0.112714, 28.75, 0.020), (106.800000, -0.138954, 28.47, 0.040),
              (106.333333, -0.094530, 28.87, 0.099), (106.166667, -0.068873, 27.65, 0.050),
              (105.833333, -0.106614, 27.08, 0.200), (105.366667, -0.156936, 27.04, 0.255)]
UAH_AGGRESSIVE = [(127.666667, -0.319629, 13.52, 0.061), (125.700000, -0.343670, 13.15, 0.101),
                  (125.400000, -0.105750, 12.41, 0.030), (125.000000, -0.087127, 12.27, 0.040),
                  (124.866667, -0.034790, 12.02, 0.010), (124.733333, 0.072356, 11.94, 0.199),
                  (125.266667, 0.156614, 11.71, 0.535), (125.833333, -0.036076, 11.58, 0.679),
                  (125.166667, -0.224746, 11.43, 0.181), (124.200000, -0.301814, 11.49, 0.127),
                  (123.066667, -0.189928, 11.69, 0.270)]
# hand results: mean features (n_s^2, n_a, n_p^2, n_w), score, label
# labels with the fitted cut offs 29 / 42: the median window of a normal trip (107 km/h, 1.0 s headway) is at 42.32
UAH_NORMAL_HAND = ([0.506495, 0.018218, 0.179814, 0.056200], 42.32, "Aggressive")
UAH_AGGRESSIVE_HAND = ([0.696416, 0.034045, 0.574431, 0.135333], 86.87, "Aggressive")
# NGSIM US-101 car 983, t = 259.0 s: inputs and hand score
NGSIM_983 = (33.884555, 0.761317, 15.797784, 0.355183)
NGSIM_983_HAND = (52.50, "Aggressive")


@pytest.mark.parametrize("seconds, hand", [(UAH_NORMAL, UAH_NORMAL_HAND), (UAH_AGGRESSIVE, UAH_AGGRESSIVE_HAND)])
def test_uah_window_score_matches_hand(seconds, hand):
    phi_hand, score_hand, label_hand = hand
    phi = index_features(*np.array(seconds).T).mean(axis=0)
    assert np.allclose(phi, phi_hand, atol=1e-5)
    s = float(original_score(phi))
    assert round(s, 2) == score_hand
    assert label(s) == label_hand


def test_ngsim_row_matches_hand():
    b = breakdown(*NGSIM_983)
    assert round(b["n_speed"] ** 2, 4) == 0.0510
    assert round(b["n_accel"], 4) == 0.1523
    assert round(b["n_prox"] ** 2, 4) == 0.4679
    assert round(b["n_wave"], 4) == 0.2368
    assert (round(b["score"], 2), b["label"]) == NGSIM_983_HAND


def test_single_second_by_hand():
    # first second of the normal window, written out: 106.966667/150 = 0.713111, squared 0.508528;
    # 0.062209/5 = 0.012442; 1 - 30.07/50 = 0.3986, squared 0.158882; 0.040/1.5 = 0.026667;
    # 0.5*0.508528 + 0.2*0.012442 + 0.8*0.158882 + 0.4*0.026667 = 0.394465 -> 39.45
    b = breakdown(*UAH_NORMAL[0])
    assert round(b["score"], 2) == 39.45


def test_breakdown_terms_add_up():
    b = breakdown(*UAH_AGGRESSIVE[0])
    assert np.isclose(b["c_speed"] + b["c_accel"] + b["c_prox"] + b["c_wave"], b["raw"])
    assert np.isclose(min(100 * b["raw"], 100), b["score"])


# ── raw data (skipped without the data folder) ──────────────────────────────
uah_present = os.path.isdir(os.path.join(DATA, "UAH-DRIVESET-v1"))
ngsim_present = os.path.isfile(os.path.join(DATA, "ngsim_us-101_5min.csv"))


@pytest.mark.skipif(not uah_present, reason="UAH-DriveSet not next to the repo")
@pytest.mark.parametrize("folder, t0, hand", [
    ("20151111123124-25km-D1-NORMAL-MOTORWAY", 306.88, UAH_NORMAL_HAND),
    ("20151111125233-24km-D1-AGGRESSIVE-MOTORWAY", 688.94, UAH_AGGRESSIVE_HAND)])
def test_uah_loader_reproduces_hand_window(folder, t0, hand):
    from datasets.uah import find_trips, load_trip, windows
    trip = next(t for t in find_trips(os.path.join(DATA, "UAH-DRIVESET-v1")) if t["trip"] == folder)
    w = windows(load_trip(trip), trip)
    row = w.loc[(w["t0"] - t0).abs() < 1e-6].iloc[0]
    phi = [row["phi_speed"], row["phi_accel"], row["phi_prox"], row["phi_wave"]]
    assert np.allclose(phi, hand[0], atol=1e-5)
    assert round(float(original_score(phi)), 2) == hand[1]


@pytest.mark.skipif(not ngsim_present, reason="NGSIM US-101 extract not next to the repo")
def test_ngsim_loader_reproduces_hand_row():
    from datasets.ngsim import read_raw, trajectories
    tr = trajectories(read_raw(os.path.join(DATA, "ngsim_us-101_5min.csv"), "us-101"))
    car = tr[tr["vehicle_id"] == 983]
    row = car.iloc[int((car["t"] - 259.0).abs().argmin())]
    got = (row["speed"] * 3.6, row["accel_1hz"], row["gap"], row["wave"])
    assert np.allclose(got, NGSIM_983, atol=1e-5)


# ── arterial and pNEUMA (docs/hand_checks/hand_check_arterial_pneuma.md) ────
LANK_100065 = (13.354879, -1.202448, 4.608576, 0.224180, 77.12)     # speed, accel, gap, wave, score at t = 29 s
PNEUMA_19 = (13.671933, -0.912782, 9.256161, 0.0, 57.19)            # second 37, leader 46 stopped ahead


def test_arterial_and_pneuma_hand_scores():
    for s, a, g, w, score in (LANK_100065, PNEUMA_19):
        assert round(float(original_score(index_features(s, a, g, w))), 2) == score


@pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "ngsim_lankershim.csv")), reason="Lankershim extract not present")
def test_lankershim_loader_reproduces_hand_row():
    from datasets.ngsim import per_second, read_raw, trajectories
    ps = per_second(trajectories(read_raw(os.path.join(DATA, "ngsim_lankershim.csv"), "lankershim"), planar=True))
    row = ps[(ps["vehicle_id"] == 100065) & (ps["t"] == 29)].iloc[0]
    assert np.allclose((row["speed_kmh"], row["accel"], row["gap_m"], row["wave_m"]), LANK_100065[:4], atol=1e-5)


@pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "pneuma", "20181024_d1_0830_0900.csv")), reason="pNEUMA slice not present")
def test_pneuma_loader_reproduces_hand_second():
    from datasets.pneuma import per_second, read_raw
    ps = per_second(read_raw(os.path.join(DATA, "pneuma", "20181024_d1_0830_0900.csv")))
    row = ps[(ps["vehicle_id"] == 19) & (ps["t"] == 37)].iloc[0]
    assert np.allclose((row["speed_kmh"], row["accel"], row["gap_m"]), PNEUMA_19[:3], atol=1e-5)
