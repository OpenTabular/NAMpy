import numpy as np
import pandas as pd

from nampy.basemodels.gam import GAM
from nampy.gam.smoothing_selection.criteria import _penalty_derivative_matrices


def _build_te_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    y = np.sin(x0) * np.cos(0.4 * x1) + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _build_gamma_data(n=120, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 1.5, size=n)
    eta = 0.5 + 0.8 * x
    mu = np.exp(eta)
    shape = 3.0
    y = rng.gamma(shape=shape, scale=mu / shape)
    return pd.DataFrame({"y": y, "x": x})


def test_penalty_derivative_matrices_match_block_sums():
    data = _build_te_data()
    model = GAM(
        formula='y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])',
        family="gaussian",
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    model.fit(data=data)

    sp = np.asarray(model.smoothing_params, dtype=np.float64)
    derivatives = _penalty_derivative_matrices(model, sp)

    n_full = int(model.n_coef_ + (1 if model.fit_intercept else 0))
    offset = 1 if model.fit_intercept else 0
    expected = [
        np.zeros((n_full, n_full), dtype=np.float64)
        for _ in range(int(model.n_smoothing_params_ or 0))
    ]

    for pb in model.penalty_blocks_:
        sl = slice(offset + pb.coef_slice.start, offset + pb.coef_slice.stop)
        expected[pb.smoothing_index][sl, sl] += np.asarray(pb.matrix, dtype=np.float64)

    assert len(derivatives) == len(expected)
    for got, want in zip(derivatives, expected):
        np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10)


def test_gamma_pirls_gradient_and_hessian_use_exact_derivatives():
    data = _build_gamma_data()
    model = GAM(
        formula='y ~ s(x, bs="cr", k=8)',
        family="gamma",
        optimize_smoothing=False,
        smoothing_method="reml",
    )
    model.fit(data=data)

    fixed_mask = (
        np.zeros(model.n_smoothing_params_, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~fixed_mask
    log_free = np.log(np.asarray(model.smoothing_params[free_mask], dtype=np.float64))

    grad = model._criterion_gradient(model.y_, log_free, method="reml")
    hess = model._criterion_hessian(model.y_, log_free, method="reml")

    assert grad.shape == (int(np.sum(free_mask)),)
    assert hess.shape == (int(np.sum(free_mask)), int(np.sum(free_mask)))
    assert np.isfinite(grad).all()
    assert np.isfinite(hess).all()
    np.testing.assert_allclose(hess, hess.T, atol=1e-8, rtol=1e-8)
