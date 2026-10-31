"""
Additive Kolmogorov-Arnold Network (Liu et al. 2024, "KAN: Kolmogorov-Arnold Networks", arXiv 2404.19756), one layer:

    score(x) = bias + sum_i  m_i * phi_i(x_i),     phi_i(x) = w_i * silu(x) + sum_j c_ij * B_j(x)

Every input x_i gets its own learned curve phi_i: a cubic (degree k) B-spline on a grid of G intervals plus a SiLU
base term, as in the KAN paper. Unknown inputs (NaN) are masked out (m_i = 0) instead of filled with 0, the same rule
as the reference formula. With one layer the model is fully readable: plot phi_i to see the learned shape of each
term, or fit it with a simple formula (symbolic_fit) to get an equation.
"""
import numpy as np
import torch
import torch.nn as nn


def bspline_basis(x, grid, k):
    """x (N, D), grid (D, G + 2k + 1) knots -> (N, D, G + k) B-spline basis values (Cox-de Boor recursion)"""
    x = x.unsqueeze(-1)
    b = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for p in range(1, k + 1):
        left = (x - grid[:, :-(p + 1)]) / (grid[:, p:-1] - grid[:, :-(p + 1)]) * b[:, :, :-1]
        right = (grid[:, p + 1:] - x) / (grid[:, p + 1:] - grid[:, 1:-p]) * b[:, :, 1:]
        b = left + right
    return b


class AdditiveKAN(nn.Module):
    def __init__(self, n_in, grid_size=5, degree=3, lo=-1.0, hi=1.0):
        super().__init__()
        self.k, self.G = degree, grid_size
        h = (hi - lo) / grid_size
        knots = torch.arange(-degree, grid_size + degree + 1, dtype=torch.float32) * h + lo
        self.register_buffer("grid", knots.repeat(n_in, 1))
        self.coef = nn.Parameter(torch.zeros(n_in, grid_size + degree))
        self.base = nn.Parameter(torch.zeros(n_in))
        self.bias = nn.Parameter(torch.zeros(1))
        self.lo, self.hi = lo, hi

    def curves(self, x):
        """(N, D) inputs, NaN allowed -> (N, D) phi_i(x_i), 0 where x_i is unknown"""
        known = ~torch.isnan(x)
        xc = torch.nan_to_num(x, nan=0.0).clamp(self.lo, self.hi - 1e-6)
        spl = (bspline_basis(xc, self.grid, self.k) * self.coef).sum(-1)
        return (self.base * nn.functional.silu(xc) + spl) * known

    def forward(self, x):
        return self.bias + self.curves(x).sum(-1)

    def curve_points(self, i, n=101):
        x = torch.linspace(self.lo, self.hi - 1e-6, n)
        X = torch.full((n, self.grid.shape[0]), float("nan"))
        X[:, i] = x
        with torch.no_grad():
            return x.numpy(), self.curves(X)[:, i].numpy()


def symbolic_fit(x, y):
    """best of a few simple shapes for y = a f(x) + c on the sampled curve; returns (name, a, c, R^2), preferring the
    simpler shape when R^2 is within 0.01"""
    cands = [("x", x), ("x^2", x ** 2), ("x^3", x ** 3), ("sqrt(x)", np.sqrt(np.clip(x, 0, None))),
             ("exp(x)", np.exp(x)), ("|x|", np.abs(x)), ("max(x, 0)^2", np.clip(x, 0, None) ** 2)]
    best, sst = None, ((y - y.mean()) ** 2).sum() + 1e-12
    for k, (name, f) in enumerate(cands):
        A = np.c_[f, np.ones_like(f)]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((A @ coef - y) ** 2).sum() / sst
        if best is None or r2 > best[3] + 0.01:
            best = (name, float(coef[0]), float(coef[1]), float(r2))
    return best
