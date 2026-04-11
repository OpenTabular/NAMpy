from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from nampy.gam import GAM
from nampy.gam.families.gamlss import gevlss, shashlss
from nampy.gam.fit.solvers.general_fit5 import (
    criterion_gradient_ml_reml_general_fit5,
    criterion_hessian_ml_reml_general_fit5,
)


def _gammals_data(n=100, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    mu = np.exp(0.4 + 0.3 * x)
    phi = np.exp(-0.5)
    y = rng.gamma(shape=1.0 / phi, scale=mu * phi)
    return pd.DataFrame({"y": y, "x": x})


def _ziplss_data(n=120, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    gamma = 0.2 + 0.4 * x
    eta = np.full(n, -0.3)
    lam = np.exp(gamma)
    p = 1.0 - np.exp(-np.exp(eta))
    y = np.zeros(n)
    ind = rng.uniform(size=n) < p
    u = rng.uniform(size=ind.sum())
    u = poisson.cdf(0, lam[ind]) + u * (1.0 - poisson.cdf(0, lam[ind]))
    y[ind] = poisson.ppf(np.minimum(u, 1.0 - 1e-12), lam[ind])
    return pd.DataFrame({"y": y, "x": x})


def _gevlss_data(n=90, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    mu = 0.2 + 0.5 * x
    rho = np.full(n, -0.4)
    xi = np.full(n, 0.1)
    u = rng.uniform(size=n)
    y = mu + ((-np.log(u)) ** (-xi) - 1.0) * np.exp(rho) / xi
    return pd.DataFrame({"y": y, "x": x})


def _shashlss_data(n=120, seed=4):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    mu = 0.5 + 0.8 * x
    sigma = np.full(n, 0.7)
    eps = np.full(n, 0.2)
    delta = np.full(n, 1.1)
    z = rng.standard_normal(n)
    y = mu + (delta * sigma) * np.sinh((1.0 / delta) * np.arcsinh(z) + eps / delta)
    return pd.DataFrame({"y": y, "x": x})


@pytest.mark.parametrize(
    ("family", "formula", "data_factory", "response_shape"),
    [
        ("gammals", ["y ~ x", "~ 1"], _gammals_data, (100, 2)),
        ("ziplss", ["y ~ x", "~ 1"], _ziplss_data, (120, 1)),
        ("gevlss", ["y ~ x", "~ 1", "~ 1"], _gevlss_data, (90, 3)),
        ("shashlss", ["y ~ x", "~ 1", "~ 1", "~ 1"], _shashlss_data, (120, 4)),
    ],
)
def test_general_family_response_prediction_shapes(
    family, formula, data_factory, response_shape
):
    data = data_factory()
    gam = GAM(family=family, formula=formula, optimize_smoothing=False)
    gam.fit(data=data)

    link = np.asarray(gam.predict(data, type="link"), dtype=np.float64)
    response = np.asarray(gam.predict(data, type="response"), dtype=np.float64)

    assert link.shape[0] == len(data)
    assert response.shape == response_shape
    assert np.all(np.isfinite(response))


def test_gaulss_reml_outer_smoothing_smoke():
    rng = np.random.default_rng(0)
    n = 80
    x = np.linspace(-1.0, 1.0, n)
    mu = 0.5 + np.sin(np.pi * x)
    y = rng.normal(mu, 0.6, size=n)
    data = pd.DataFrame({"y": y, "x": x})

    gam = GAM(
        family="gaulss",
        formula=['y ~ s(x, bs="cr", k=6)', "~ 1"],
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    response = np.asarray(gam.predict(data, type="response"), dtype=np.float64)
    assert response.shape == (n, 2)
    assert np.isfinite(float(gam.smoothing_score_))
    assert gam._optim_result is not None


@pytest.mark.parametrize("factory", [gevlss, shashlss])
def test_general_families_expose_outer_derivative_modes(factory, monkeypatch):
    family = factory()
    assert not family.supports_analytic_outer_derivatives
    assert family.supports_analytic_outer_gradient
    assert family.supports_analytic_outer_hessian

    class _Model:
        def __init__(self, family):
            self.family = family
            self.prior_weights_ = np.ones(3, dtype=np.float64)

        @staticmethod
        def _expand_smoothing_params_from_log(log_sp):
            return np.exp(np.asarray(log_sp, dtype=np.float64))

    model = _Model(family)
    y = np.ones(3, dtype=np.float64)
    log_sp = np.array([0.0, 0.5], dtype=np.float64)

    monkeypatch.setattr(
        "nampy.gam.fit.solvers.general_fit5._run_general_fit5",
        lambda *_args, **_kwargs: {
            "fit": {
                "score1": np.array([1.25, -0.5], dtype=np.float64),
                "score2": np.array([[2.0, 0.3], [0.3, 4.0]], dtype=np.float64),
            }
        },
    )
    monkeypatch.setattr(
        "nampy.gam.fit.solvers.general_fit5._finite_difference_general_fit5_hessian_from_gradient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("finite-difference Hessian fallback should not run")
        ),
    )

    grad = criterion_gradient_ml_reml_general_fit5(model, y, log_sp, "REML")
    hess = criterion_hessian_ml_reml_general_fit5(model, y, log_sp, "REML")

    np.testing.assert_allclose(grad, np.array([1.25, -0.5], dtype=np.float64))
    np.testing.assert_allclose(
        hess, np.array([[2.0, 0.3], [0.3, 4.0]], dtype=np.float64)
    )
    assert model._general_fit5_outer_derivative_info == {
        "gradient_source": "analytic",
        "hessian_source": "analytic",
        "supports_analytic_outer_derivatives": False,
    }
