"""KAN building blocks: B-spline basis sums to 1 (partition of unity), masking of unknown inputs, symbolic fit."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.kan import AdditiveKAN, bspline_basis, symbolic_fit  # noqa: E402


def test_partition_of_unity_and_mask():
    m = AdditiveKAN(2, grid_size=5, degree=3)
    x = torch.tensor([[-0.9, 0.3], [0.5, float("nan")]])
    b = bspline_basis(x.nan_to_num(0.0), m.grid, 3)
    assert torch.allclose(b.sum(-1), torch.ones(2, 2), atol=1e-5)      # cubic B-splines sum to 1 inside the grid
    with torch.no_grad():
        m.coef.fill_(1.0)
    assert m.curves(x)[1, 1] == 0.0                                    # unknown input contributes nothing


def test_symbolic_fit_finds_square():
    x = np.linspace(0, 1, 50)
    name, a, c, r2 = symbolic_fit(x, 3 * x ** 2 + 1)
    assert name == "x^2" and np.isclose(a, 3, atol=1e-6) and np.isclose(c, 1, atol=1e-6) and r2 > 0.999
