from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.special import gammaln

from nampy.gam import GAM
from nampy.gam.families.gamlss import gammals
from nampy.gam.families.gamlss.gammals import _SoftplusBLinkInfo
from nampy.gam.fit.solvers.general_family.newton import (
    GeneralNewtonControl,
    solve_general_newton_fit,
)

# ======================================================================
# gammals
# ======================================================================

# ---------------------------------------------------------------------------
# 1. SoftplusBLinkInfo roundtrip
# ---------------------------------------------------------------------------


def test_softplusb_roundtrip():
    """Verify that softplusb roundtrip."""
    link = _SoftplusBLinkInfo(b=-7.0)
    rng = np.random.default_rng(0)
    # log(sigma) values well above b=-7
    mu = rng.uniform(-5.0, 2.0, 50)
    eta = link.linkfun(mu)
    mu_back = link.linkinv(eta)
    assert_allclose(mu_back, mu, rtol=1e-10, atol=1e-12)


def test_softplusb_linkinv_lower_bound():
    """Verify that softplusb linkinv lower bound."""
    link = _SoftplusBLinkInfo(b=-7.0)
    # linkinv must always be >= b
    eta = np.linspace(-20.0, 5.0, 200)
    mu = link.linkinv(eta)
    assert np.all(mu >= -7.0 - 1e-12)


# ---------------------------------------------------------------------------
# 2. mu_eta finite-difference check
# ---------------------------------------------------------------------------


def test_softplusb_mu_eta_fd():
    """Verify that softplusb mu eta finite differences."""
    link = _SoftplusBLinkInfo(b=-7.0)
    rng = np.random.default_rng(1)
    eta = rng.uniform(-5.0, 3.0, 40)
    eps = 1e-7
    fd = (link.linkinv(eta + eps) - link.linkinv(eta - eps)) / (2 * eps)
    analytic = link.mu_eta(eta)
    assert_allclose(analytic, fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. d2link finite-difference check
# ---------------------------------------------------------------------------


def test_softplusb_d2link_fd():
    """Verify that softplusb d2link finite differences."""
    link = _SoftplusBLinkInfo(b=-7.0)
    rng = np.random.default_rng(2)
    # mu values close to b to get non-negligible d2link values
    mu = rng.uniform(-6.5, -4.0, 40)  # mu - b in [0.5, 3.0]
    eps = 1e-5  # larger eps to reduce cancellation in double FD

    def eta_from_mu(m):
        return link.linkfun(m)

    def d1link_numeric(m):
        return (eta_from_mu(m + eps) - eta_from_mu(m - eps)) / (2 * eps)

    fd_d2 = (d1link_numeric(mu + eps) - d1link_numeric(mu - eps)) / (2 * eps)
    analytic = link.d2link(mu)
    # double FD has O(eps^2) error; values ~0.05-0.5, so atol=1e-4 is appropriate
    assert_allclose(analytic, fd_d2, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. gammals ll: log-lik value against direct formula
# ---------------------------------------------------------------------------


def test_gammals_ll_loglik():
    """gammals ll gives log-lik matching direct Gamma formula."""
    rng = np.random.default_rng(7)
    n, p1, p2 = 80, 4, 3
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.1
    # strictly positive y
    y = rng.gamma(shape=2.0, scale=0.5, size=n)
    weights = np.ones(n)

    fam = gammals()
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=0)

    # Direct formula
    eta0 = X[:, jj[0]] @ coef[jj[0]]
    eta1 = X[:, jj[1]] @ coef[jj[1]]
    mu_ref = eta0  # identity link on log-mean
    th_ref = fam.linfo[1].linkinv(eta1)  # log-sigma
    eth_ref = np.exp(-th_ref)  # shape = 1/sigma
    ethmuy = np.exp(-th_ref - mu_ref) * y
    etlymt = eth_ref * (np.log(y) - mu_ref - th_ref)
    l_ref = float(np.sum(etlymt - np.log(y) - ethmuy - gammaln(eth_ref)))
    assert_allclose(result["l"], l_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. gammals ll gradient finite-difference check
# ---------------------------------------------------------------------------


def test_gammals_ll_gradient_fd():
    """gammals gradient matches finite-difference gradient."""
    rng = np.random.default_rng(11)
    n, p1, p2 = 50, 4, 3
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.1
    y = rng.gamma(shape=2.0, scale=0.5, size=n)
    weights = np.ones(n)

    fam = gammals()
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lb = result["lb"]
    assert lb.shape == (p,)

    eps = 1e-6
    fd = np.zeros(p)
    l0 = fam.ll(y, X, jj, coef, weights, offset=None, deriv=0)["l"]
    for k in range(p):
        cp = coef.copy()
        cp[k] += eps
        l1 = fam.ll(y, X, jj, cp, weights, offset=None, deriv=0)["l"]
        fd[k] = (l1 - l0) / eps

    assert_allclose(lb, fd, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. gammals ll Hessian shape and negative semi-definiteness
# ---------------------------------------------------------------------------


def test_gammals_ll_hessian_shape():
    """gammals Hessian has correct shape and is negative semi-definite."""
    rng = np.random.default_rng(13)
    n, p1, p2 = 40, 4, 3
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.1
    y = rng.gamma(shape=2.0, scale=0.5, size=n)
    weights = np.ones(n)

    fam = gammals()
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lbb = result["lbb"]
    assert lbb.shape == (p, p)
    ev = np.linalg.eigvalsh(lbb)
    assert np.all(ev <= 1e-8), f"Hessian has positive eigenvalue: {ev.max():.4g}"


# ---------------------------------------------------------------------------
# 8. solve_general_newton_fit end-to-end on simulated gamma data
# ---------------------------------------------------------------------------


def test_gam_fit5_gammals_convergence():
    """
    solve_general_newton_fit with gammals recovers mean slope and log-sigma intercept.

    True model: log(mean) = 1.0 + 2.0*x, log(sigma) = -2.0 (constant).
    """
    rng = np.random.default_rng(42)
    n = 300
    x_true = rng.standard_normal(n)
    log_mean_true = 1.0 + 2.0 * x_true
    log_sigma_true = -2.0  # sigma = exp(-2) ≈ 0.135
    sigma_true = np.exp(log_sigma_true)
    mean_true = np.exp(log_mean_true)

    # Gamma(shape=1/sigma, scale=mean*sigma)
    shape_true = 1.0 / sigma_true
    y = rng.gamma(shape=shape_true, scale=mean_true * sigma_true, size=n)

    p1, p2 = 2, 1
    p = p1 + p2
    X = np.zeros((n, p), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = x_true
    X[:, 2] = 1.0  # intercept for sigma predictor
    jj = [np.arange(p1), np.arange(p1, p)]

    fam = gammals()

    St = np.zeros((p, p), dtype=np.float64)
    lsp = np.array([], dtype=np.float64)
    S_blocks: list = []

    ctl = GeneralNewtonControl(maxit=100, epsilon=1e-8, trace=False)
    fit = solve_general_newton_fit(
        X,
        y,
        jj,
        lsp,
        St,
        S_blocks,
        ldetS=0.0,
        ldetS1=None,
        ldetS2=None,
        family=fam,
        weights=None,
        offset=None,
        deriv=0,
        control=ctl,
    )

    assert fit["iter"] > 0
    coef = fit["coef"]

    # log-mean predictor: coef[0] ≈ 1.0, coef[1] ≈ 2.0
    assert_allclose(coef[0], 1.0, atol=0.25), f"intercept = {coef[0]:.3f}"
    assert_allclose(coef[1], 2.0, atol=0.25), f"slope = {coef[1]:.3f}"

    # log-sigma predictor: linkinv(coef[2]) ≈ log_sigma_true = -2.0
    log_sigma_est = fam.linfo[1].linkinv(coef[2])
    assert_allclose(log_sigma_est, log_sigma_true, atol=0.3), (
        f"log_sigma_est = {log_sigma_est:.3f}, expected {log_sigma_true:.3f}"
    )
    assert not fit["warn"], f"Warnings: {fit['warn']}"


def test_gam_public_api_gammals_formula_list_fit():
    """Public GAM API fits a two-predictor gammals model via formula list."""
    rng = np.random.default_rng(1234)
    n = 220
    x = rng.standard_normal(n)
    log_mean_true = 0.8 + 0.6 * x
    log_sigma_true = -1.2
    sigma_true = np.exp(log_sigma_true)
    mean_true = np.exp(log_mean_true)
    y = rng.gamma(shape=1.0 / sigma_true, scale=mean_true * sigma_true, size=n)
    data = pd.DataFrame({"y": y, "x": x})

    gam = GAM(
        family="gammals",
        formula=["y ~ x", "~ 1"],
        optimize_smoothing=False,
    )
    gam.fit(data=data)
    fit = gam.fit_result(include_covariances=True)
    coef_full = np.asarray(fit.coef_full, dtype=np.float64)
    vp = np.asarray(gam.vcov(), dtype=np.float64)
    vc = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)

    assert coef_full.shape == (3,)
    assert vp.shape == (3, 3)
    assert vc.shape == (3, 3)

    eta = np.asarray(gam.predict(data, type="link"), dtype=np.float64)
    fitted = np.asarray(gam.predict(data, type="response"), dtype=np.float64)
    assert eta.shape == (n, 2)
    assert fitted.shape == (n, 2)

    assert_allclose(coef_full[0], 0.8, atol=0.3)
    assert_allclose(coef_full[1], 0.6, atol=0.3)
    log_sigma_est = gam.family.linfo[1].linkinv(np.array([coef_full[2]]))[0]
    assert_allclose(log_sigma_est, log_sigma_true, atol=0.35)
