"""
Two-track aggressiveness assessment pipeline.

GroundTruthAssessor  — uses simulation internals (oracle / "hand calculation" baseline).
AgentAssessor        — uses only the Kinematics observation vector + sensor noise.

The two are intentionally asymmetric so they do NOT match 100 %.
Expected category agreement: 90–95 % (boundary cases get misclassified due to noise).

Observation format expected by AgentAssessor:
    absolute=True, normalize=False  →  [presence, x_m, y_m, vx_ms, vy_ms]
"""

import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_W_SPEED = 0.5
_W_ACCEL = 0.2
_W_PROX  = 0.8
_W_WAVE  = 0.3
_MAX_RAW = _W_SPEED + _W_ACCEL + _W_PROX + _W_WAVE   # 1.8
_LANE_W  = 4.0   # highway_env lane width in metres


def _label(score: float) -> str:
    if score < 30:
        return "Conservative"
    if score < 65:
        return "Normal"
    return "Aggressive"


def _formula(speed_kmh: float, accel_ms2: float, prox_m: float, wave_m: float):
    """Core aggressiveness formula shared by both assessors."""
    ns  = np.clip(speed_kmh / 150.0, 0, 1)
    na  = np.clip(abs(accel_ms2) / 5.0, 0, 1)
    np_ = max(0.0, 1.0 - prox_m / 50.0) if 0 < prox_m <= 50 else 0.0
    nw  = np.clip(wave_m / 2.0, 0, 1)
    raw = ns**2 * _W_SPEED + na * _W_ACCEL + np_ * _W_PROX + nw * _W_WAVE
    score = (raw / _MAX_RAW) * 100.0
    return round(score, 2), _label(score)


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH ASSESSOR  (has full simulation access)
# ─────────────────────────────────────────────────────────────────────────────
class GroundTruthAssessor:
    """
    Computes the 'hand calculation' baseline using direct access to simulation
    vehicle objects.  No noise, no estimation — exact physics.
    """

    # policy_frequency = 5 Hz → dt = 0.2 s between successive calls
    _DT = 0.2

    def __init__(self):
        # vehicle id → deque of past speeds (m/s) for finite-difference accel
        self._speed_hist: dict[int, deque] = {}

    def reset(self):
        self._speed_hist.clear()

    def compute(self, vehicle, ego_vehicle):
        """
        Returns (score_0_to_100, category_label) using true simulation values.
        Never called from inside the agent decision path.
        """
        vid = id(vehicle)

        # True speed (m/s → km/h)
        speed_kmh = vehicle.speed * 3.6

        # True acceleration via finite differences on speed history
        hist = self._speed_hist.setdefault(vid, deque(maxlen=5))
        accel = (vehicle.speed - hist[-1]) / self._DT if hist else 0.0
        hist.append(vehicle.speed)

        # True longitudinal proximity to ego vehicle
        prox_m = abs(vehicle.position[0] - ego_vehicle.position[0])

        # True lateral deviation from nearest lane centre
        lane_centre = round(vehicle.position[1] / _LANE_W) * _LANE_W
        wave_m = abs(vehicle.position[1] - lane_centre)

        return _formula(speed_kmh, accel, prox_m, wave_m)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT ASSESSOR  (observation vector + sensor noise only)
# ─────────────────────────────────────────────────────────────────────────────
class AgentAssessor:
    """
    Estimates aggressiveness using ONLY what the ego agent can observe:
        obs[i] = [presence, x_m, y_m, vx_ms, vy_ms]  (absolute, non-normalised)

    Sensor noise models camera + radar measurement uncertainty.
    The agent has NO access to vehicle.speed, vehicle.action, or vehicle.position.

    noise_scale > 1.0 for adverse weather (poor visibility / reduced sensor quality).
    """

    # Camera + radar noise (physically motivated, σ in metres / m/s)
    _SIGMA_POS  = 0.8   # position noise [m]          (≈ camera projection error)
    _SIGMA_VEL  = 0.5   # velocity noise [m/s]         (≈ radar Doppler error)
    _SIGMA_ACC  = 0.6   # extra noise on accel estimate (hard to measure indirectly)

    # Obs timestep (policy_frequency = 5 Hz)
    _DT = 0.2

    def __init__(self, history_len: int = 5, noise_scale: float = 1.0):
        self.noise_scale = noise_scale
        self._spd_hist: dict[int, deque] = {}   # slot → deque of speed estimates
        self._y_hist:   dict[int, deque] = {}   # slot → deque of y positions

    def reset(self):
        self._spd_hist.clear()
        self._y_hist.clear()

    def assess_all(self, obs) -> dict:
        """
        Assess every visible non-ego vehicle.
        Returns {slot_idx: (score, category)}.
        Slots with presence < 0.5 are skipped.
        """
        ego_x = float(obs[0][1])
        results = {}
        for i in range(1, len(obs)):
            if obs[i][0] < 0.5:
                continue
            results[i] = self._assess_slot(i, obs[i], ego_x)
        return results

    def _assess_slot(self, slot: int, row, ego_x: float):
        s = self.noise_scale

        # Unpack observation row
        _, x, y, vx, vy = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])

        # ── Add sensor noise ──────────────────────────────────────────────────
        x  += np.random.normal(0, self._SIGMA_POS * s)
        y  += np.random.normal(0, self._SIGMA_POS * s)
        vx += np.random.normal(0, self._SIGMA_VEL * s)
        vy += np.random.normal(0, self._SIGMA_VEL * s)

        # Feature 1 — speed estimate from velocity magnitude
        spd_ms  = np.sqrt(vx**2 + vy**2)
        spd_kmh = spd_ms * 3.6

        # Feature 2 — acceleration estimate from noisy velocity history
        hist = self._spd_hist.setdefault(slot, deque(maxlen=5))
        if hist:
            # Linear-regression slope over history reduces single-step noise
            speeds = list(hist) + [spd_ms]
            n = len(speeds)
            t = np.arange(n) * self._DT
            if n >= 3:
                slope = np.polyfit(t, speeds, 1)[0]   # m/s per second
                accel_est = abs(slope)
            else:
                accel_est = abs(spd_ms - hist[-1]) / self._DT
            # Add residual noise that represents real estimation uncertainty
            accel_est += abs(np.random.normal(0, self._SIGMA_ACC * s))
        else:
            accel_est = 0.0
        hist.append(spd_ms)

        # Feature 3 — proximity from longitudinal offset to ego
        prox_est = abs(x - ego_x)

        # Feature 4 — lane waviness estimated from lateral position history
        yh = self._y_hist.setdefault(slot, deque(maxlen=5))
        yh.append(y)
        if len(yh) >= 3:
            y_arr = np.array(yh)
            lc_est = round(float(np.mean(y_arr)) / _LANE_W) * _LANE_W
            wave_est = float(np.std(y_arr - lc_est))
        else:
            wave_est = 0.0

        return _formula(spd_kmh, accel_est, prox_est, wave_est)


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING & COMPARISON UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def match_obs_to_vehicles(obs, road_vehicles, ego_vehicle, tol: float = 5.0) -> dict:
    """
    Maps each observation slot index to a road vehicle object using position
    proximity (works because obs uses absolute coordinates).
    Returns {slot_idx: vehicle_object}.
    """
    mapping = {}
    for i in range(1, len(obs)):
        if obs[i][0] < 0.5:
            continue
        ox, oy = float(obs[i][1]), float(obs[i][2])
        best_v, best_d = None, float("inf")
        for v in road_vehicles:
            if v is ego_vehicle:
                continue
            d = float(np.hypot(v.position[0] - ox, v.position[1] - oy))
            if d < best_d:
                best_d, best_v = d, v
        if best_v is not None and best_d < tol:
            mapping[i] = best_v
    return mapping


def compare_and_report(gt_results: dict, agent_results: dict):
    """
    Compare ground-truth vs agent category labels.
    Both dicts keyed by slot index: {idx: (score, category)}.
    Returns (accuracy_pct, details_list).
    """
    common = sorted(set(gt_results) & set(agent_results))
    if not common:
        return 0.0, []

    hits, details = 0, []
    for k in common:
        gs, gl = gt_results[k]
        ag_s, ag_l = agent_results[k]
        match = gl == ag_l
        hits += match
        details.append({
            "slot":     k,
            "gt_score": gs,  "gt_label":    gl,
            "ag_score": ag_s, "ag_label":   ag_l,
            "match":    match,
        })

    return (hits / len(common)) * 100.0, details
