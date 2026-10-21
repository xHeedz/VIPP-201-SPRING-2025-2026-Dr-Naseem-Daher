"""
The one reference definition of the Aggressiveness Index.

    AI = min(100 * (w_s * n_s^2 + w_a * n_a + w_p * n_p^2 + w_w * n_w), 100)

    n_s = min(speed_kmh / 150, 1)
    n_a = min(|accel| / 5, 1)
    n_p = 1 - gap / 50 if 0 < gap <= 50 else 0      (gap to the car ahead, 0 = no car ahead)
    n_w = min(|wave| / 1.5, 1)                       (offset from the lane centre)

    weights (0.5, 0.2, 0.8, 0.4), labels: < 29 Conservative, < 42 Normal, else Aggressive.

Every script imports the constants and functions below; nothing else re-implements them.
"""
import numpy as np

SPEED_MAX_KMH = 150.0
ACCEL_MAX_MS2 = 5.0
PROX_MAX_M = 50.0
WAVE_MAX_M = 1.5
WEIGHTS = (0.5, 0.2, 0.8, 0.4)          # speed, accel, prox, wave
# cut offs fitted on labelled data (scripts/fit_cutoffs.py, data/cutoffs_fitted.json): Youden J on UAH (aggressive
# vs normal, 41.2) and on the calibrated planted SUMO drivers (conservative 29.9, aggressive 42.9); were 35 / 70
THRESHOLDS = (29.0, 42.0)               # Conservative | Normal | Aggressive


def label(score):
    if score < THRESHOLDS[0]:
        return "Conservative"
    if score < THRESHOLDS[1]:
        return "Normal"
    return "Aggressive"


def index_features(speed_kmh, accel_ms2, prox_m, wave_m):
    """(n_speed^2, n_accel, n_prox^2, n_wave) for arrays; prox_m <= 0 means no car ahead."""
    speed_kmh, accel_ms2, prox_m, wave_m = (np.asarray(x, dtype=float) for x in (speed_kmh, accel_ms2, prox_m, wave_m))
    ns = np.minimum(speed_kmh / SPEED_MAX_KMH, 1.0)
    na = np.minimum(np.abs(accel_ms2) / ACCEL_MAX_MS2, 1.0)
    npx = np.where((prox_m > 0) & (prox_m <= PROX_MAX_M), 1.0 - prox_m / PROX_MAX_M, 0.0)
    nw = np.minimum(np.abs(wave_m) / WAVE_MAX_M, 1.0)
    return np.stack([ns ** 2, na, npx ** 2, nw], axis=-1)


THW_MAX_S = 3.0      # time headway variant: 50 m at 60 km/h is 3.0 s


def headway_features(speed_kmh, accel_ms2, prox_m, wave_m, thw_max=THW_MAX_S):
    """index_features with proximity as time headway (gap / speed) instead of metres:
    n_p = 1 - thw / thw_max if 0 < thw <= thw_max, else 0. A stopped car (speed < 0.5 m/s)
    has no headway term. Candidate replacement for bug 4, not the reference yet."""
    f = index_features(speed_kmh, accel_ms2, prox_m, wave_m)
    v = np.asarray(speed_kmh, dtype=float) / 3.6
    gap = np.asarray(prox_m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        thw = np.where((gap > 0) & (v >= 0.5), gap / np.maximum(v, 0.5), 0.0)
    npx = np.where((thw > 0) & (thw <= thw_max), 1.0 - thw / thw_max, 0.0)
    f[..., 2] = npx ** 2
    return f


# share of the metres term per environment, fitted by scripts/fit_proximity_mix.py (data/proximity_mix.json):
# max mean AUC aggressive vs normal on planted SUMO drivers and UAH. proposal, the reference still uses metres
PROX_MIX = {"highway": 0.5, "urban": 0.5, "weather": 0.2}


def mixed_features(speed_kmh, accel_ms2, prox_m, wave_m, alpha, thw_max=THW_MAX_S):
    """index_features with a proximity term mixing both measures (Dr. Daher, Oct 2026: metres or time headway
    depending on the environment): n_p^2 = alpha * metres term + (1 - alpha) * headway term.
    alpha = 1 is index_features, alpha = 0 is headway_features. Per environment alphas: PROX_MIX."""
    f = index_features(speed_kmh, accel_ms2, prox_m, wave_m)
    h = headway_features(speed_kmh, accel_ms2, prox_m, wave_m, thw_max)
    f[..., 2] = alpha * f[..., 2] + (1.0 - alpha) * h[..., 2]
    return f


def original_score(features):
    """The score from index_features (the hand-set reference index)."""
    # explicit weighted sum: numpy 2.0 matmul on macOS raises spurious divide-by-zero warnings
    return np.minimum(100.0 * (np.asarray(features, dtype=float) * np.array(WEIGHTS)).sum(axis=-1), 100.0)


def breakdown(speed_kmh, accel_ms2, prox_m, wave_m):
    """Every intermediate number of one score, for hand checks and printed traces."""
    f = index_features(speed_kmh, accel_ms2, prox_m, wave_m)
    n = (np.sqrt(f[0]), f[1], np.sqrt(f[2]), f[3])
    terms = tuple(float(w * x) for w, x in zip(WEIGHTS, f))
    score = float(original_score(f))
    return {
        "n_speed": float(n[0]), "n_accel": float(n[1]), "n_prox": float(n[2]), "n_wave": float(n[3]),
        "c_speed": terms[0], "c_accel": terms[1], "c_prox": terms[2], "c_wave": terms[3],
        "raw": sum(terms), "score": score, "label": label(score),
    }


class AggressivenessModel:
    """Scalar interface to the reference index (kept for the SUMO scripts and older callers)."""

    def __init__(self):
        self.w_speed, self.w_accel, self.w_prox, self.w_wave = WEIGHTS

    def normalize(self, speed_kmh, accel_ms2, prox_m, wave_m):
        b = breakdown(speed_kmh, accel_ms2, prox_m, wave_m)
        return b["n_speed"], b["n_accel"], b["n_prox"], b["n_wave"]

    def get_ai_score(self, speed, accel, prox, wave):
        b = breakdown(speed, accel, prox, wave)
        return b["score"], b["label"]


# ── reference formula, Oct 2026 revision (Dr. Daher) ────────────────────────
# speed linear, acceleration squared, proximity mixed per environment (PROX_MIX), unknown terms (NaN, from
# datasets/cleaning.py) left out of the score with the remaining weights rescaled. Weights and cut offs are fitted
# by scripts/fit_reference_weights.py (data/reference_weights.json); the old functions above stay unchanged.
REF_TERMS = ("speed", "accel2", "prox2", "wave")
_REF_FILE = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "data", "reference_weights.json")


def _load_reference():
    import json
    import os
    if os.path.exists(_REF_FILE):
        with open(_REF_FILE) as f:
            d = json.load(f)
        return tuple(d["weights"]), tuple(d["cutoffs"])
    w = np.array(WEIGHTS) / sum(WEIGHTS)
    return tuple(float(x) for x in w), THRESHOLDS


REF_WEIGHTS, REF_CUTOFFS = _load_reference()


def reference_terms(speed_kmh, accel_ms2, gap_m, wave_m, environment, thw_max=THW_MAX_S):
    """(..., 4) terms (speed, accel^2, prox^2 mixed, wave), each 0 to 1, NaN where the input is unknown.
    gap: NaN unknown, 0 no car ahead (term 0), else metres. environment: a name or an array of names."""
    speed_kmh, accel_ms2, gap_m, wave_m = (np.asarray(x, dtype=float) for x in (speed_kmh, accel_ms2, gap_m, wave_m))
    ns = np.minimum(speed_kmh / SPEED_MAX_KMH, 1.0)
    na2 = np.minimum(np.abs(accel_ms2) / ACCEL_MAX_MS2, 1.0) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        metres = np.where((gap_m > 0) & (gap_m <= PROX_MAX_M), 1.0 - gap_m / PROX_MAX_M, 0.0) ** 2
        v = speed_kmh / 3.6
        thw = np.where((gap_m > 0) & (v >= 0.5), gap_m / np.maximum(v, 0.5), 0.0)
        headway = np.where((thw > 0) & (thw <= thw_max), 1.0 - thw / thw_max, 0.0) ** 2
    if isinstance(environment, str):
        alpha = PROX_MIX.get(environment, 0.5)
    else:
        alpha = np.array([PROX_MIX.get(e, 0.5) for e in np.asarray(environment)], dtype=float)
    prox = np.where(np.isnan(gap_m), np.nan, alpha * metres + (1.0 - alpha) * headway)
    nw = np.where(np.isnan(wave_m), np.nan, np.minimum(np.abs(wave_m) / WAVE_MAX_M, 1.0))
    return np.stack([ns, na2, prox, nw], axis=-1)


def reference_score(terms, weights=None):
    """0 to 100: weighted mean of the known terms (unknown terms left out, weights rescaled to sum to 1)."""
    t = np.asarray(terms, dtype=float)
    w = np.asarray(REF_WEIGHTS if weights is None else weights, dtype=float)
    known = ~np.isnan(t)
    num = np.where(known, t, 0.0) * w
    den = (known * w).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * num.sum(axis=-1) / np.where(den > 0, den, np.nan)


def reference_label(score, cutoffs=None):
    lo, hi = REF_CUTOFFS if cutoffs is None else cutoffs
    s = np.asarray(score, dtype=float)
    return np.where(s < lo, "conservative", np.where(s < hi, "normal", "aggressive"))
