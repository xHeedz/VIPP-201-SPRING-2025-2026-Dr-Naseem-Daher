"""
Sensor noise models for the agent-side assessment.

Each model corrupts one signal (speed, acceleration, gap, lateral position).
Models keep per-vehicle state where the physics needs it (drift, delay,
correlated noise, held values after a dropout), keyed by vehicle id.
Pipelines chain several models on the same signal.

Two ways to use them:
  streaming   model(value, key)            one reading at a time, as in SUMO
  batch       model.apply_series(x, key)   a whole time series, as in datasets

Noise kinds (make_noise builds any of them from a single severity level):
  gaussian        zero-mean white noise                          x + N(0, s)
  bias            constant offset, fixed per vehicle             x + b
  drift           slowly wandering offset (random walk)          x + sum N(0, s)
  colored         noise correlated in time, AR(1)                x + e_t, e_t = r e_(t-1) + N
  multiplicative  error proportional to the reading              x (1 + N(0, s))
  spikes          rare large outliers                            x + big error, with prob p
  quantization    rounding to the sensor resolution              round(x / q) q
  dropout         missing readings; the last valid value is held
  delay           readings arrive k steps late
"""
import math
import random
from collections import defaultdict, deque

KINDS = ("gaussian", "bias", "drift", "colored", "multiplicative",
         "spikes", "quantization", "dropout", "delay")

# Base scale per signal at severity 1.0. Gaussian values match the original
# SumoAgentAssessor sigmas (speed 0.5 m/s, accel 0.3 m/s^2, gap 0.8 m).
BASE_SCALE = {"speed": 0.5, "accel": 0.3, "gap": 0.8, "lateral": 0.2}


class NoiseModel:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.state = {}

    def reset(self):
        self.state = {}

    def __call__(self, x, key=None):
        raise NotImplementedError

    def apply_series(self, values, key="series"):
        return [self(v, key) for v in values]


class Gaussian(NoiseModel):
    def __init__(self, sigma, seed=None):
        super().__init__(seed)
        self.sigma = sigma

    def __call__(self, x, key=None):
        return x + self.rng.gauss(0.0, self.sigma)


class Bias(NoiseModel):
    """Constant offset per vehicle, drawn once, sign random."""
    def __init__(self, magnitude, seed=None):
        super().__init__(seed)
        self.magnitude = magnitude

    def __call__(self, x, key=None):
        if key not in self.state:
            self.state[key] = self.magnitude * self.rng.choice((-1.0, 1.0)) * self.rng.uniform(0.5, 1.0)
        return x + self.state[key]


class Drift(NoiseModel):
    """Offset that performs a bounded random walk (calibration drift)."""
    def __init__(self, step_sigma, limit, seed=None):
        super().__init__(seed)
        self.step_sigma, self.limit = step_sigma, limit

    def __call__(self, x, key=None):
        d = self.state.get(key, 0.0) + self.rng.gauss(0.0, self.step_sigma)
        d = max(-self.limit, min(self.limit, d))
        self.state[key] = d
        return x + d


class Colored(NoiseModel):
    """AR(1) noise with stationary standard deviation sigma and correlation rho."""
    def __init__(self, sigma, rho=0.9, seed=None):
        super().__init__(seed)
        self.sigma, self.rho = sigma, rho

    def __call__(self, x, key=None):
        e = self.state.get(key, 0.0)
        e = self.rho * e + self.rng.gauss(0.0, self.sigma * math.sqrt(1.0 - self.rho ** 2))
        self.state[key] = e
        return x + e


class Multiplicative(NoiseModel):
    def __init__(self, rel_sigma, seed=None):
        super().__init__(seed)
        self.rel_sigma = rel_sigma

    def __call__(self, x, key=None):
        return x * (1.0 + self.rng.gauss(0.0, self.rel_sigma))


class Spikes(NoiseModel):
    """With probability p, add a large error of typical size `size` (heavy-tailed outlier)."""
    def __init__(self, p, size, seed=None):
        super().__init__(seed)
        self.p, self.size = p, size

    def __call__(self, x, key=None):
        if self.rng.random() < self.p:
            return x + self.rng.choice((-1.0, 1.0)) * self.size * (1.0 + abs(self.rng.gauss(0.0, 1.0)))
        return x


class Quantization(NoiseModel):
    def __init__(self, step, seed=None):
        super().__init__(seed)
        self.step = step

    def __call__(self, x, key=None):
        return round(x / self.step) * self.step if self.step > 0 else x


class Dropout(NoiseModel):
    """With probability p the reading is lost and the last delivered value is repeated."""
    def __init__(self, p, seed=None):
        super().__init__(seed)
        self.p = p

    def __call__(self, x, key=None):
        if key in self.state and self.rng.random() < self.p:
            return self.state[key]
        self.state[key] = x
        return x


class Delay(NoiseModel):
    """The reading delivered now is the one measured `steps` calls ago."""
    def __init__(self, steps, seed=None):
        super().__init__(seed)
        self.steps = int(steps)

    def __call__(self, x, key=None):
        q = self.state.setdefault(key, deque(maxlen=self.steps + 1))
        q.append(x)
        return q[0]


class Pipeline(NoiseModel):
    def __init__(self, models):
        super().__init__()
        self.models = list(models)

    def reset(self):
        for m in self.models:
            m.reset()

    def __call__(self, x, key=None):
        for m in self.models:
            x = m(x, key)
        return x


def make_noise(kind, signal, level=1.0, seed=None):
    """One noise model for `signal` at severity `level` (0 means no noise)."""
    s = BASE_SCALE[signal] * level
    if level <= 0:
        return Pipeline([])
    if kind == "gaussian":
        return Gaussian(s, seed)
    if kind == "bias":
        return Bias(2.0 * s, seed)
    if kind == "drift":
        return Drift(0.2 * s, 3.0 * s, seed)
    if kind == "colored":
        return Colored(s, 0.9, seed)
    if kind == "multiplicative":
        return Multiplicative(0.05 * level, seed)
    if kind == "spikes":
        return Spikes(min(0.02 * level, 0.5), 6.0 * BASE_SCALE[signal], seed)
    if kind == "quantization":
        return Quantization(2.0 * s, seed)
    if kind == "dropout":
        return Dropout(min(0.1 * level, 0.9), seed)
    if kind == "delay":
        return Delay(max(1, round(level)), seed)
    raise ValueError(f"unknown noise kind {kind!r}; choose from {KINDS}")


def noise_suite(kinds, level=1.0, seed=0):
    """{signal: Pipeline} with the given kinds stacked on every signal."""
    if isinstance(kinds, str):
        kinds = [kinds]
    return {sig: Pipeline([make_noise(k, sig, level, seed=seed + 31 * i + j)
                           for j, k in enumerate(kinds)])
            for i, sig in enumerate(BASE_SCALE)}


def reset_suite(suite):
    for p in suite.values():
        p.reset()
