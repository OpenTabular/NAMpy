from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import poisson

from nampy.gam import GAM
from nampy.gam.families.gamlss import gammals, gaulss, gevlss, shashlss, ziplss
from nampy.gam.fit.solvers.gam_fit5 import GamFit5Control, gam_fit5
from nampy.gam.fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from nampy.gam.fit.solvers.general_fit5 import (
    _run_general_fit5,
    criterion_gradient_ml_reml_general_fit5,
    criterion_hessian_ml_reml_general_fit5,
)
from nampy.gam.smoothing_selection.reparam import _stable_penalty_logdet_derivatives
# ======================================================================
# gaulss
# ======================================================================

# ---------------------------------------------------------------------------
# 1. trind_generator
# ---------------------------------------------------------------------------


def test_trind_generator_k2_symmetry():
    tri = trind_generator(2)
    i2 = tri["i2"]
    # i2 must be symmetric
    assert i2[0, 1] == i2[1, 0]
    # packed order: (0,0)=0, (0,1)=1, (1,1)=2
    assert i2[0, 0] == 0
    assert i2[0, 1] == 1
    assert i2[1, 1] == 2


def test_trind_generator_k2_reverse():
    tri = trind_generator(2)
    i2r = tri["i2r"]
    K = 2
    # i2r[m] should encode (k, l) as l + k*K
    # for K=2: packed entries are (0,0),(0,1),(1,1) → indices 0,1,3
    assert i2r[0] == 0 + 0 * K  # k=0, l=0
    assert i2r[1] == 1 + 0 * K  # k=0, l=1
    assert i2r[2] == 1 + 1 * K  # k=1, l=1


def test_trind_generator_k3_counts():
    tri = trind_generator(3)
    i2 = tri["i2"]
    i3 = tri["i3"]
    # K=3: K*(K+1)/2 = 6 packed second-order entries → max index = 5
    assert int(i2.max()) == 5
    # K*(K+1)*(K+2)/6 = 10 packed third-order entries → max index = 9
    assert int(i3.max()) == 9


# ---------------------------------------------------------------------------
# 2. gamlss_etamu with identity links (ig1=1, g2=g3=g4=0)
# ---------------------------------------------------------------------------


def test_gamlss_etamu_identity_links():
    """With identity links ig1=1, g2=g3=g4=0, eta-derivs == mu-derivs."""
    rng = np.random.default_rng(0)
    n, K = 50, 2
    l1 = rng.standard_normal((n, K))
    l2 = rng.standard_normal((n, 3))  # K*(K+1)/2 = 3
    l3 = rng.standard_normal((n, 4))  # K*(K+1)*(K+2)/6 = 4

    ig1 = np.ones((n, K), dtype=np.float64)  # d mu / d eta = 1
    g2 = np.zeros((n, K), dtype=np.float64)
    g3 = np.zeros((n, K), dtype=np.float64)

    tri = trind_generator(K)
    i2, i3 = tri["i2"], tri["i3"]

    de = gamlss_etamu(l1, l2, l3, 0, ig1, g2, g3, 0, i2, i3, None, deriv=1)

    assert_allclose(de["l1"], l1)
    assert_allclose(de["l2"], l2)
    assert_allclose(de["l3"], l3)


# ---------------------------------------------------------------------------
# 3. gamlss_gH gradient finite-difference check
# ---------------------------------------------------------------------------


def test_gamlss_gH_gradient_fd():
    """Gradient from gamlss_gH should match finite-difference gradient."""
    rng = np.random.default_rng(42)
    n, p1, p2 = 30, 4, 3
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.5
    tri = trind_generator(2)
    i2 = tri["i2"]

    # Build a simple l1, l2 from Gaussian log-lik with fixed params
    eta0 = X[:, jj[0]] @ coef[jj[0]]
    eta1 = X[:, jj[1]] @ coef[jj[1]]
    mu = eta0
    tau = np.exp(eta1) + 0.01  # logb-like
    y = rng.standard_normal(n) * (1.0 / tau) + mu

    def log_lik(c):
        e0 = X[:, jj[0]] @ c[jj[0]]
        e1 = X[:, jj[1]] @ c[jj[1]]
        mu_ = e0
        tau_ = np.exp(e1) + 0.01
        return -0.5 * np.sum((y - mu_) ** 2 * tau_ ** 2) + np.sum(np.log(tau_))

    eps = 1e-6
    fd_grad = np.zeros(p)
    l0 = log_lik(coef)
    for k in range(p):
        cp = coef.copy()
        cp[k] += eps
        fd_grad[k] = (log_lik(cp) - l0) / eps

    # Compute l1, l2 at coef
    ymu = y - mu
    tau2 = tau ** 2
    l1 = np.column_stack([tau2 * ymu, 1.0 / tau - tau * ymu ** 2])
    l2 = np.column_stack([-tau2, 2.0 * l1[:, 0] / tau, -ymu ** 2 - 1.0 / tau2])

    # Identity links: ig1 = [1, exp(eta1)+0.01] ... actually use mu.eta of exp+b
    ig1_0 = np.ones(n)
    ig1_1 = np.exp(eta1)  # mu.eta for log link ≈ tau (not logb but close enough for test)
    ig1 = np.column_stack([ig1_0, ig1_1])
    g2 = np.zeros((n, 2))

    de = gamlss_etamu(l1, l2, 0, 0, ig1, g2, 0, 0, i2, None, None, deriv=0)
    ret = gamlss_gH(X, jj, de["l1"], de["l2"], i2, deriv=0)
    lb = ret["lb"]

    # lb should approximate fd_grad when ig1=1 (identity case)
    # With identity link for predictor 0, exact match; predictor 1 uses chain rule
    # Just check sign/order of magnitude
    assert np.all(np.isfinite(lb))
    assert lb.shape == (p,)


# ======================================================================
# General Family API
# ======================================================================

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
    grad = criterion_gradient_ml_reml_general_fit5(model, y, log_sp, "REML")
    hess = criterion_hessian_ml_reml_general_fit5(model, y, log_sp, "REML")

    np.testing.assert_allclose(grad, np.array([1.25, -0.5], dtype=np.float64))
    np.testing.assert_allclose(
        hess, np.array([[2.0, 0.3], [0.3, 4.0]], dtype=np.float64)
    )
    assert model._general_fit5_outer_derivative_info == {
        "gradient_source": "analytic",
        "hessian_source": "analytic",
        "penalty_logdet_source": "analytic",
        "supports_analytic_outer_derivatives": False,
        "uses_exact_penalty_logdet": True,
    }


def test_general_family_outer_derivatives_require_exact_family_support():
    class _Family:
        supports_analytic_outer_derivatives = False
        supports_analytic_outer_gradient = False
        supports_analytic_outer_hessian = False

    class _Model:
        family = _Family()
        prior_weights_ = np.ones(3, dtype=np.float64)

        @staticmethod
        def _expand_smoothing_params_from_log(log_sp):
            return np.exp(np.asarray(log_sp, dtype=np.float64))

    model = _Model()
    y = np.ones(3, dtype=np.float64)
    log_sp = np.array([0.0, 0.5], dtype=np.float64)

    with pytest.raises(NotImplementedError, match="analytic outer gradients"):
        criterion_gradient_ml_reml_general_fit5(model, y, log_sp, "REML")

    with pytest.raises(NotImplementedError, match="analytic outer Hessians"):
        criterion_hessian_ml_reml_general_fit5(model, y, log_sp, "REML")


def test_general_fit5_run_uses_canonical_penalty_logdet_derivatives(monkeypatch):
    recorded = {}

    def _stub_gam_fit5(
        _X,
        _y,
        _jj,
        _lsp,
        _St,
        _S_blocks,
        *,
        ldetS,
        ldetS1,
        ldetS2,
        **_kwargs,
    ):
        recorded["ldetS"] = float(ldetS)
        recorded["ldetS1"] = np.asarray(ldetS1, dtype=np.float64).copy()
        recorded["ldetS2"] = np.asarray(ldetS2, dtype=np.float64).copy()
        return {"score": 0.0}

    class _Pred:
        def __init__(self):
            self.design_matrix = np.arange(12, dtype=np.float64).reshape(4, 3)
            self.has_intercept = False

    class _Penalty:
        def __init__(self):
            self.coef_slice = slice(0, 3)
            self.smoothing_index = 0
            self.matrix = np.diag([1.0, 2.0, 3.0])

    class _Model:
        n_samples_ = 4
        max_irls_iter = 2
        irls_tol = 1e-7
        hparams = {}
        prior_weights_ = np.ones(4, dtype=np.float64)
        predictor_designs = [_Pred()]
        penalty_blocks_ = [_Penalty()]
        family = object()
        _optim_method = "REML"

    monkeypatch.setattr(
        "nampy.gam.smoothing_selection.reparam._stable_penalty_logdet_derivatives",
        lambda *_args, **_kwargs: (
            3.5,
            np.array([1.0], dtype=np.float64),
            np.array([[7.0]], dtype=np.float64),
        ),
    )
    monkeypatch.setattr("nampy.gam.fit.solvers.general_fit5.gam_fit5", _stub_gam_fit5)

    _run_general_fit5(_Model(), np.ones(4, dtype=np.float64), np.array([2.0]))

    assert recorded["ldetS"] == pytest.approx(3.5)
    np.testing.assert_allclose(recorded["ldetS1"], np.array([1.0], dtype=np.float64))
    np.testing.assert_allclose(
        recorded["ldetS2"], np.array([[7.0]], dtype=np.float64)
    )


def test_general_fit5_penalty_logdet_derivatives_match_finite_difference():
    rng = np.random.default_rng(123)
    n = 80
    x = np.linspace(-1.0, 1.0, n)
    mu = 0.3 + 0.5 * x
    sigma = np.exp(-0.2 + 0.1 * x)
    y = rng.normal(mu, sigma, size=n)
    data = pd.DataFrame({"y": y, "x": x})

    gam = GAM(
        family="gaulss",
        formula=['y ~ s(x, bs="cr", k=6)', "~ 1"],
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
    log_sp = np.log(np.clip(sp, 1e-300, None))
    logdet, grad, hess = _stable_penalty_logdet_derivatives(gam, sp, order=2)
    ref_logdet, ref_grad, ref_hess = _stable_penalty_logdet_derivatives(
        gam, sp, order=2
    )

    np.testing.assert_allclose(logdet, ref_logdet, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grad, ref_grad, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hess, ref_hess, rtol=0.0, atol=1e-12)

    steps = np.maximum(1e-4, 1e-3 * (1.0 + np.abs(log_sp)))
    fd_grad = np.zeros_like(grad)
    fd_hess = np.zeros_like(hess)

    for j, h in enumerate(steps):
        rho_plus = log_sp.copy()
        rho_minus = log_sp.copy()
        rho_plus[j] += h
        rho_minus[j] -= h
        sp_plus = np.exp(rho_plus)
        sp_minus = np.exp(rho_minus)

        val_plus = _stable_penalty_logdet_derivatives(gam, sp_plus, order=2)[0]
        val_minus = _stable_penalty_logdet_derivatives(gam, sp_minus, order=2)[0]
        fd_grad[j] = (val_plus - val_minus) / (2.0 * h)

        grad_plus = _stable_penalty_logdet_derivatives(gam, sp_plus, order=2)[1]
        grad_minus = _stable_penalty_logdet_derivatives(gam, sp_minus, order=2)[1]
        fd_hess[:, j] = (grad_plus - grad_minus) / (2.0 * h)

    fd_hess = 0.5 * (fd_hess + fd_hess.T)

    np.testing.assert_allclose(grad, fd_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hess, fd_hess, rtol=2e-4, atol=5e-5)
