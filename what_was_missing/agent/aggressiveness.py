"""
Torch-free pieces shared by the agent, SumoEnv and the tests: regime targets,
feature normalization and the index itself.

RECREATED BY CLAUDE (Anthropic), September 2026. See ../README.md.
"""

REGIMES = ("highway", "urban", "weather")
FEATURES = ("speed", "accel", "prox", "wave")
CONTEXT = ("is_highway", "is_urban", "density", "friction", "slip")

# Final report Section 3.1 (weather target after the week-11 retune)
REGIME_TARGETS = {
    "highway": (0.30, 0.10, 0.45, 0.15),
    "urban":   (0.20, 0.15, 0.35, 0.30),
    "weather": (0.10, 0.40, 0.15, 0.35),
}

V_REF_MS = 150.0 / 3.6     # 150 km/h
A_REF = 5.0                # m/s^2
GAP_REF = 50.0             # m; proximity counts only inside this range
WAVE_REF = 1.5             # m of lateral drift
DENSITY_REF = 50.0         # vehicles per km per lane that maps to density = 1
CONSERVATIVE_BELOW = 35.0
AGGRESSIVE_FROM = 70.0


def normalize_features(speed_ms, accel_ms2, gap_m, wave_m):
    """Raw kinematics -> [n_speed, n_accel, n_prox, n_wave], each in [0, 1]."""
    n_speed = min(max(speed_ms / V_REF_MS, 0.0), 1.0)
    n_accel = min(abs(accel_ms2) / A_REF, 1.0)
    n_prox = max(0.0, 1.0 - gap_m / GAP_REF) if gap_m < GAP_REF else 0.0
    n_wave = min(abs(wave_m) / WAVE_REF, 1.0)
    return [n_speed, n_accel, n_prox, n_wave]


def context_vector(road_type, density_veh_km_lane, friction, slip):
    """road_type in {'highway', 'urban'} -> [is_highway, is_urban, density, friction, slip]."""
    return [
        1.0 if road_type == "highway" else 0.0,
        1.0 if road_type == "urban" else 0.0,
        min(max(density_veh_km_lane / DENSITY_REF, 0.0), 1.0),
        min(max(friction, 0.0), 1.0),
        min(max(slip, 0.0), 1.0),
    ]


def ai_from_lists(features, weights):
    """Aggressiveness Index in [0, 100] for one vehicle."""
    s = sum(f * w for f, w in zip(features, weights))
    return 100.0 * min(max(s, 0.0), 1.0)


def category(ai):
    if ai < CONSERVATIVE_BELOW:
        return "conservative"
    if ai < AGGRESSIVE_FROM:
        return "normal"
    return "aggressive"
