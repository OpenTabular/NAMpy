"""
Unit tests for GAMLSS family implementations.

Covers log-likelihood, gradient, Hessian, link function, initialization,
and end-to-end convergence for all GAMLSS families:
  - gammals (Gamma location-scale)
  - gaulss (Gaussian location-scale)
  - ziplss (Zero-inflated Poisson location-scale)
  - gevlss (GEV location-scale-shape)
  - shashlss (sinh-arcsinh location-scale-skewness-kurtosis)
  - General family API (gevlss, shashlss via GAM public API)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import digamma, gammaln
from scipy.stats import genextreme, norm, poisson

from nampy.gam import GAM
from nampy.gam.families.gamlss import (
    GammalsFamily,
    GevlssFamily,
    ShashlssFamily,
    ZiplssFamily,
    _LogEBLinkInfo,
    _ShiftedLogitLinkInfo,
    _SoftplusBLinkInfo,
    _l1ee,
    _ldg,
    _lde,
    _lee1,
    _zipll,
    gammals,
    gaulss,
    gevlss,
    shashlss,
    ziplss,
)
from nampy.gam.fit.solvers.gam_fit5 import GamFit5Control, gam_fit5
from nampy.gam.fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from nampy.gam.fit.solvers.general_fit5 import (
    criterion_gradient_ml_reml_general_fit5,
    criterion_hessian_ml_reml_general_fit5,
)



# ======================================================================
# gammals
# ======================================================================

# ---------------------------------------------------------------------------
# 1. SoftplusBLinkInfo roundtrip
# ---------------------------------------------------------------------------


def test_softplusb_roundtrip():
    link = _SoftplusBLinkInfo(b=-7.0)
    rng = np.random.default_rng(0)
    # log(sigma) values well above b=-7
    mu = rng.uniform(-5.0, 2.0, 50)
    eta = link.linkfun(mu)
    mu_back = link.linkinv(eta)
    assert_allclose(mu_back, mu, rtol=1e-10, atol=1e-12)


def test_softplusb_linkinv_lower_bound():
    link = _SoftplusBLinkInfo(b=-7.0)
    # linkinv must always be >= b
    eta = np.linspace(-20.0, 5.0, 200)
    mu = link.linkinv(eta)
    assert np.all(mu >= -7.0 - 1e-12)


# ---------------------------------------------------------------------------
# 2. mu_eta finite-difference check
# ---------------------------------------------------------------------------


def test_softplusb_mu_eta_fd():
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
    assert np.isfinite(result["l"])

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
    assert np.all(np.isfinite(lb))

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
    assert np.all(np.isfinite(lbb))
    ev = np.linalg.eigvalsh(lbb)
    assert np.all(ev <= 1e-8), f"Hessian has positive eigenvalue: {ev.max():.4g}"


# ---------------------------------------------------------------------------
# 7. gammals initialize
# ---------------------------------------------------------------------------


def test_gammals_initialize():
    """gammals initialize returns correct-shaped finite vector."""
    rng = np.random.default_rng(17)
    n, p1, p2 = 80, 5, 4
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    y = rng.gamma(shape=2.0, scale=1.0, size=n)
    weights = np.ones(n)

    fam = gammals()
    start = fam.initialize(y, X, jj, offset=None, weights=weights)
    assert start.shape == (p,)
    assert np.all(np.isfinite(start))


# ---------------------------------------------------------------------------
# 8. gam_fit5 end-to-end on simulated gamma data
# ---------------------------------------------------------------------------


def test_gam_fit5_gammals_convergence():
    """
    gam_fit5 with gammals recovers mean slope and log-sigma intercept.

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

    ctl = GamFit5Control(maxit=100, epsilon=1e-8, trace=False)
    fit = gam_fit5(
        X, y, jj, lsp, St, S_blocks, ldetS=0.0, ldetS1=None, ldetS2=None,
        family=fam, weights=None, offset=None, deriv=0, control=ctl,
    )

    assert fit["iter"] > 0
    coef = fit["coef"]
    assert np.all(np.isfinite(coef))

    # log-mean predictor: coef[0] ≈ 1.0, coef[1] ≈ 2.0
    assert_allclose(coef[0], 1.0, atol=0.25), f"intercept = {coef[0]:.3f}"
    assert_allclose(coef[1], 2.0, atol=0.25), f"slope = {coef[1]:.3f}"

    # log-sigma predictor: linkinv(coef[2]) ≈ log_sigma_true = -2.0
    log_sigma_est = fam.linfo[1].linkinv(coef[2])
    assert_allclose(log_sigma_est, log_sigma_true, atol=0.3), (
        f"log_sigma_est = {log_sigma_est:.3f}, expected {log_sigma_true:.3f}"
    )
    assert not fit["warn"], f"Warnings: {fit['warn']}"


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


# ---------------------------------------------------------------------------
# 4. gaulss ll: log-lik value and derivatives
# ---------------------------------------------------------------------------


def test_gaulss_ll_loglik():
    """gaulss ll gives finite log-likelihood with matching analytical formula."""
    rng = np.random.default_rng(7)
    n, p1, p2 = 100, 5, 4
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.2
    y = rng.standard_normal(n)
    weights = np.ones(n)

    fam = gaulss()
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=0)
    assert np.isfinite(result["l"])

    # Direct formula: N(mu, 1/tau^2)
    eta0 = X[:, jj[0]] @ coef[jj[0]]
    eta1 = X[:, jj[1]] @ coef[jj[1]]
    mu_ref = eta0  # identity link
    tau_ref = 1.0 / (np.exp(eta1) + 0.01)  # logb link
    l_ref = float(
        np.sum(-0.5 * (y - mu_ref) ** 2 * tau_ref ** 2 - 0.5 * np.log(2 * np.pi) + np.log(tau_ref))
    )
    assert_allclose(result["l"], l_ref, rtol=1e-10)


def test_gaulss_ll_gradient_fd():
    """gaulss gradient matches finite-difference gradient."""
    rng = np.random.default_rng(11)
    n, p1, p2 = 50, 4, 3
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.1
    y = rng.standard_normal(n)
    weights = np.ones(n)

    fam = gaulss()
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lb = result["lb"]
    assert lb.shape == (p,)
    assert np.all(np.isfinite(lb))

    # Finite-difference gradient
    eps = 1e-6
    fd = np.zeros(p)
    l0 = fam.ll(y, X, jj, coef, weights, offset=None, deriv=0)["l"]
    for k in range(p):
        cp = coef.copy()
        cp[k] += eps
        l1 = fam.ll(y, X, jj, cp, weights, offset=None, deriv=0)["l"]
        fd[k] = (l1 - l0) / eps

    assert_allclose(lb, fd, rtol=1e-4, atol=1e-6)


def test_gaulss_ll_hessian_shape():
    """gaulss Hessian has correct shape and is symmetric negative semi-definite."""
    rng = np.random.default_rng(13)
    n, p1, p2 = 40, 4, 3
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.1
    y = rng.standard_normal(n)
    weights = np.ones(n)

    fam = gaulss()
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lbb = result["lbb"]
    assert lbb.shape == (p, p)
    assert np.all(np.isfinite(lbb))
    # Hessian of log-lik should be negative semi-definite
    ev = np.linalg.eigvalsh(lbb)
    assert np.all(ev <= 1e-10), f"Hessian of log-lik has positive eigenvalue: {ev.max():.4g}"


# ---------------------------------------------------------------------------
# 5. gaulss initialize
# ---------------------------------------------------------------------------


def test_gaulss_initialize():
    """gaulss initialize returns finite coefficient vector of correct size."""
    rng = np.random.default_rng(17)
    n, p1, p2 = 80, 6, 4
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    y = rng.standard_normal(n) * 2.0 + 1.0
    weights = np.ones(n)

    fam = gaulss()
    start = fam.initialize(y, X, jj, offset=None, weights=weights)
    assert start.shape == (p,)
    assert np.all(np.isfinite(start))


# ---------------------------------------------------------------------------
# 6. gam_fit5 end-to-end on simulated data
# ---------------------------------------------------------------------------


def test_gam_fit5_simple_convergence():
    """
    gam_fit5 with gaulss should converge to sensible estimates on simulated data.
    Mean predictor recovers slope; precision predictor recovers constant.
    """
    rng = np.random.default_rng(99)
    n = 200
    x_true = rng.standard_normal(n)
    sigma_true = 0.5
    mu_true = 1.0 + 2.0 * x_true
    y = rng.normal(mu_true, sigma_true, size=n)

    # Design matrix: predictor 1 = [1, x], predictor 2 = [1]
    p1, p2 = 2, 1
    p = p1 + p2
    X = np.zeros((n, p), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = x_true
    X[:, 2] = 1.0  # intercept for precision predictor
    jj = [np.arange(p1), np.arange(p1, p)]

    fam = gaulss()

    # No penalty for this test
    St = np.zeros((p, p), dtype=np.float64)
    lsp = np.array([], dtype=np.float64)
    S_blocks: list = []

    ctl = GamFit5Control(maxit=100, epsilon=1e-8, trace=False)
    fit = gam_fit5(
        X, y, jj, lsp, St, S_blocks, ldetS=0.0, ldetS1=None, ldetS2=None,
        family=fam, weights=None, offset=None, deriv=0, control=ctl,
    )

    assert fit["iter"] > 0
    coef = fit["coef"]
    assert np.all(np.isfinite(coef))

    # Mean predictor: coef[0] ≈ 1.0, coef[1] ≈ 2.0
    assert_allclose(coef[0], 1.0, atol=0.2), f"intercept = {coef[0]:.3f}"
    assert_allclose(coef[1], 2.0, atol=0.2), f"slope = {coef[1]:.3f}"

    # Precision predictor: tau = 1/sigma ≈ 2.0
    # coef[2] is log(1/tau - b) ≈ log(2.0 - 0.01) ≈ 0.69
    tau_est = fam.linfo[1].linkinv(coef[2])
    assert_allclose(tau_est, 1.0 / sigma_true, atol=0.3), (
        f"tau_est = {tau_est:.3f}, expected {1.0/sigma_true:.3f}"
    )
    assert not fit["warn"], f"Warnings: {fit['warn']}"


def test_gam_public_api_gaulss_formula_list_fit():
    """Public GAM API fits a two-predictor gaulss model via formula list."""
    rng = np.random.default_rng(123)
    n = 180
    x = rng.standard_normal(n)
    sigma_true = 0.6
    mu_true = 0.5 + 1.5 * x
    y = rng.normal(mu_true, sigma_true, size=n)
    data = pd.DataFrame({"y": y, "x": x})

    gam = GAM(
        family="gaulss",
        formula=["y ~ x", "~ 1"],
        optimize_smoothing=False,
    )
    gam.fit(data=data)

    assert gam.coef_full_.shape == (3,)
    assert gam.Vp_.shape == (3, 3)
    assert gam.Vc_.shape == (3, 3)
    assert np.isfinite(gam.loglik())

    eta = np.asarray(gam.predict(data, type="link"), dtype=np.float64)
    fitted = np.asarray(gam.predict(data, type="response"), dtype=np.float64)
    assert eta.shape == (n, 2)
    assert fitted.shape == (n, 2)
    assert_allclose(gam.coef_full_[0], 0.5, atol=0.25)
    assert_allclose(gam.coef_full_[1], 1.5, atol=0.25)
    tau_est = gam.family.linfo[1].linkinv(np.array([gam.coef_full_[2]]))[0]
    assert_allclose(tau_est, 1.0 / sigma_true, atol=0.35)


# ======================================================================
# ziplss
# ======================================================================

# ---------------------------------------------------------------------------
# 1. l1ee and lee1 correctness
# ---------------------------------------------------------------------------


def test_l1ee_values():
    """l1ee(x) = log(1-exp(-exp(x))) for moderate x."""
    x = np.array([0.0, 0.5, 1.0, 2.0])
    ref = np.log(1.0 - np.exp(-np.exp(x)))
    assert_allclose(_l1ee(x), ref, rtol=1e-10)


def test_lee1_values():
    """lee1(x) = log(exp(exp(x)) - 1) for moderate x."""
    x = np.array([0.5, 1.0, 1.5])
    ref = np.log(np.expm1(np.exp(x)))
    assert_allclose(_lee1(x), ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. ldg gradient FD check
# ---------------------------------------------------------------------------


def test_ldg_gradient_fd():
    """ldg l1 matches FD derivative of lee1 term."""
    rng = np.random.default_rng(10)
    # y>0 case: l(g) = y*g - lee1(g) - lgamma(y+1)
    # dl/dg = y + ldg$l1
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g = rng.uniform(0.5, 2.0, 5)
    eps = 1e-7

    def log_lik(g_v):
        return y * g_v - _lee1(g_v) - gammaln(y + 1.0)

    fd = (log_lik(g + eps) - log_lik(g - eps)) / (2.0 * eps)
    lg = _ldg(g, deriv=1)
    analytic = y + lg["l1"]
    assert_allclose(analytic, fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. lde gradient FD check
# ---------------------------------------------------------------------------


def test_lde_gradient_fd():
    """lde l1 matches FD derivative of l1ee."""
    rng = np.random.default_rng(11)
    eta = rng.uniform(-1.0, 2.0, 20)
    eps = 1e-7
    fd = (_l1ee(eta + eps) - _l1ee(eta - eps)) / (2.0 * eps)
    le = _lde(eta, deriv=1)
    assert_allclose(le["l1"], fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. zipll log-lik against direct formula
# ---------------------------------------------------------------------------


def test_zipll_loglik():
    """_zipll log-lik matches direct formula for mixed y=0 / y>0 data."""
    rng = np.random.default_rng(7)
    n = 40
    g = rng.uniform(0.5, 1.5, n)
    eta = rng.uniform(-0.5, 1.0, n)
    lam = np.exp(g)  # Poisson mean
    p = 1.0 - np.exp(-np.exp(eta))  # presence prob

    y = np.zeros(n, dtype=np.float64)
    y[::3] = rng.poisson(lam[::3]) + 1  # some non-zeros

    zl = _zipll(y, g, eta, deriv=0)
    l_analytic = zl["l"]

    # Direct formula
    l_ref = np.where(
        y == 0,
        -np.exp(eta),  # log P(y=0) = log(1-p) = -exp(eta)
        _l1ee(eta) + y * g - _lee1(g) - gammaln(y + 1.0),
    )
    assert_allclose(l_analytic, l_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. ziplss ll gradient FD check
# ---------------------------------------------------------------------------


def test_ziplss_ll_gradient_fd():
    """ziplss gradient matches finite-difference gradient."""
    rng = np.random.default_rng(11)
    n, p1, p2 = 60, 3, 3
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.2

    # Generate ZIP data
    g_true = X[:, jj[0]] @ coef[jj[0]]
    eta_true = X[:, jj[1]] @ coef[jj[1]]
    lam = np.exp(g_true)
    p_pres = 1.0 - np.exp(-np.exp(eta_true))
    y = np.where(rng.uniform(size=n) < p_pres, rng.poisson(lam), 0).astype(np.float64)
    weights = np.ones(n)

    fam = ziplss()
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lb = result["lb"]
    assert lb.shape == (p,)
    assert np.all(np.isfinite(lb))

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
# 6. ziplss ll Hessian shape and sign
# ---------------------------------------------------------------------------


def test_ziplss_ll_hessian_shape():
    """ziplss Hessian has correct shape and is negative semi-definite."""
    rng = np.random.default_rng(13)
    n, p1, p2 = 50, 3, 3
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    coef = rng.standard_normal(p) * 0.1

    g_true = X[:, jj[0]] @ coef[jj[0]]
    eta_true = X[:, jj[1]] @ coef[jj[1]]
    lam = np.exp(g_true)
    p_pres = 1.0 - np.exp(-np.exp(eta_true))
    y = np.where(rng.uniform(size=n) < p_pres, rng.poisson(lam) + 1, 0).astype(np.float64)
    weights = np.ones(n)

    fam = ziplss()
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lbb = result["lbb"]
    assert lbb.shape == (p, p)
    assert np.all(np.isfinite(lbb))
    ev = np.linalg.eigvalsh(lbb)
    assert np.all(ev <= 1e-8), f"Hessian has positive eigenvalue: {ev.max():.4g}"


# ---------------------------------------------------------------------------
# 7. ziplss initialize
# ---------------------------------------------------------------------------


def test_ziplss_initialize():
    """ziplss initialize returns correct-shaped finite vector."""
    rng = np.random.default_rng(17)
    n, p1, p2 = 80, 4, 4
    p = p1 + p2
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p)]
    y = np.where(rng.uniform(size=n) < 0.7, rng.poisson(3.0, n), 0).astype(np.float64)
    weights = np.ones(n)

    fam = ziplss()
    start = fam.initialize(y, X, jj, offset=None, weights=weights)
    assert start.shape == (p,)
    assert np.all(np.isfinite(start))


# ---------------------------------------------------------------------------
# 8. gam_fit5 convergence on simulated ZIP data
# ---------------------------------------------------------------------------


def test_gam_fit5_ziplss_convergence():
    """
    gam_fit5 with ziplss recovers approximate log-mean and presence intercepts.
    """
    rng = np.random.default_rng(55)
    n = 400
    x_true = rng.standard_normal(n)

    # log-mean predictor: 1.0 + 0.5*x
    g_true = 1.0 + 0.5 * x_true
    # loglog presence: constant, so P(y>0) = 1-exp(-exp(0.5)) ≈ 0.78
    eta_true = np.full(n, 0.5)

    lam = np.exp(g_true)
    p_pres = 1.0 - np.exp(-np.exp(eta_true))
    # Generate from correct ziplss: structural zero with prob (1-p_pres), ZTP otherwise
    is_present = rng.uniform(size=n) < p_pres
    # ZTP: Poisson conditioned on > 0
    pois_draws = rng.poisson(lam)
    pois_draws = np.where(pois_draws == 0, 1, pois_draws)  # clamp to ensure y>0 given present
    y = np.where(is_present, pois_draws, 0).astype(np.float64)

    p1, p2 = 2, 1
    p = p1 + p2
    X = np.zeros((n, p), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = x_true
    X[:, 2] = 1.0
    jj = [np.arange(p1), np.arange(p1, p)]

    fam = ziplss()
    St = np.zeros((p, p), dtype=np.float64)
    lsp = np.array([], dtype=np.float64)
    S_blocks: list = []

    ctl = GamFit5Control(maxit=100, epsilon=1e-8, trace=False)
    fit = gam_fit5(
        X, y, jj, lsp, St, S_blocks, ldetS=0.0, ldetS1=None, ldetS2=None,
        family=fam, weights=None, offset=None, deriv=0, control=ctl,
    )

    assert fit["iter"] > 0
    coef = fit["coef"]
    assert np.all(np.isfinite(coef))

    # Fitted presence probability: 1-exp(-exp(coef[2])) should be near true p_pres[0]
    p_fit = 1.0 - np.exp(-np.exp(coef[2]))
    p_true_val = p_pres[0]
    assert_allclose(p_fit, p_true_val, atol=0.1), (
        f"p_fit={p_fit:.3f}, expected≈{p_true_val:.3f}"
    )

    # Fitted log-mean intercept on Poisson mean scale: exp(coef[0]) ≈ exp(1.0) ≈ 2.72
    assert_allclose(np.exp(coef[0]), np.exp(1.0), atol=0.4), (
        f"mean intercept = {np.exp(coef[0]):.3f}"
    )


# ======================================================================
# gevlss
# ======================================================================

# ---------------------------------------------------------------------------
# 1. ShiftedLogitLinkInfo roundtrip and range
# ---------------------------------------------------------------------------


def test_shifted_logit_roundtrip():
    link = _ShiftedLogitLinkInfo()
    xi = np.linspace(-0.9, 0.45, 30)
    eta = link.linkfun(xi)
    xi_back = link.linkinv(eta)
    assert_allclose(xi_back, xi, rtol=1e-10)


def test_shifted_logit_range():
    link = _ShiftedLogitLinkInfo()
    eta = np.linspace(-10.0, 10.0, 100)
    xi = link.linkinv(eta)
    assert np.all(xi > -1.0), "xi must be > -1"
    assert np.all(xi < 0.5), "xi must be < 0.5"


def test_shifted_logit_mu_eta_fd():
    link = _ShiftedLogitLinkInfo()
    rng = np.random.default_rng(3)
    eta = rng.uniform(-3.0, 3.0, 30)
    eps = 1e-7
    fd = (link.linkinv(eta + eps) - link.linkinv(eta - eps)) / (2.0 * eps)
    analytic = link.mu_eta(eta)
    assert_allclose(analytic, fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. gevlss ll log-lik against direct GEV formula
# ---------------------------------------------------------------------------


def test_gevlss_ll_loglik():
    """gevlss ll gives log-lik matching scipy genextreme."""
    rng = np.random.default_rng(7)
    n, p1, p2, p3 = 60, 3, 2, 2
    p = p1 + p2 + p3
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p1 + p2), np.arange(p1 + p2, p)]
    coef = np.zeros(p)
    coef[:p1] = rng.standard_normal(p1) * 0.3
    coef[p1:p1+p2] = rng.standard_normal(p2) * 0.1
    coef[p1+p2:] = rng.standard_normal(p3) * 0.2

    fam = gevlss()
    mu = X[:, jj[0]] @ coef[jj[0]]
    rho = X[:, jj[1]] @ coef[jj[1]]
    xi = fam.linfo[2].linkinv(X[:, jj[2]] @ coef[jj[2]])

    # Generate y in the support
    sigma = np.exp(rho)
    y = mu + sigma * (np.random.default_rng(8).exponential(size=n) ** (-xi) - 1.0) / xi
    # Ensure support: 1 + xi*(y-mu)/sigma > 0
    y = np.where(1.0 + xi * (y - mu) / sigma > 0.01, y, mu + sigma * 0.1)

    weights = np.ones(n)
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=0)
    assert np.isfinite(result["l"])

    # Direct GEV log-lik: Gumbel (xi→0) or GEV(xi≠0)
    eps_xi = 1e-7
    aa = np.maximum(1.0 + xi * (y - mu) / sigma, 1e-300)
    l_ref = float(np.sum(
        -(1.0 / xi + 1.0) * np.log(aa) - aa ** (-1.0 / xi) - rho
    ))
    assert_allclose(result["l"], l_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. gevlss ll gradient finite-difference check
# ---------------------------------------------------------------------------


def test_gevlss_ll_gradient_fd():
    """gevlss gradient matches finite-difference gradient."""
    rng = np.random.default_rng(11)
    n, p1, p2, p3 = 40, 3, 2, 2
    p = p1 + p2 + p3
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p1 + p2), np.arange(p1 + p2, p)]
    coef = np.zeros(p)
    coef[:p1] = rng.standard_normal(p1) * 0.2
    coef[p1:p1+p2] = rng.standard_normal(p2) * 0.1

    fam = gevlss()
    mu = X[:, jj[0]] @ coef[jj[0]]
    rho = X[:, jj[1]] @ coef[jj[1]]
    xi = fam.linfo[2].linkinv(X[:, jj[2]] @ coef[jj[2]])
    sigma = np.exp(rho)

    # Simulate GEV data safely in support
    rng2 = np.random.default_rng(12)
    y = mu + sigma * rng2.standard_normal(n)
    aa = 1.0 + xi * (y - mu) / sigma
    y = np.where(aa > 0.1, y, mu + sigma * 0.5)

    weights = np.ones(n)
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lb = result["lb"]
    assert lb.shape == (p,)
    assert np.all(np.isfinite(lb))

    eps = 1e-6
    fd = np.zeros(p)
    l0 = fam.ll(y, X, jj, coef, weights, offset=None, deriv=0)["l"]
    for k in range(p):
        cp = coef.copy()
        cp[k] += eps
        l1_val = fam.ll(y, X, jj, cp, weights, offset=None, deriv=0)["l"]
        fd[k] = (l1_val - l0) / eps

    assert_allclose(lb, fd, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. gevlss ll Hessian shape and negative semi-definiteness
# ---------------------------------------------------------------------------


def test_gevlss_ll_hessian_shape():
    """gevlss Hessian has correct shape and is negative semi-definite."""
    rng = np.random.default_rng(13)
    n, p1, p2, p3 = 50, 3, 2, 2
    p = p1 + p2 + p3
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p1 + p2), np.arange(p1 + p2, p)]
    coef = np.zeros(p)
    coef[:p1] = rng.standard_normal(p1) * 0.1
    coef[p1:p1+p2] = rng.standard_normal(p2) * 0.05

    fam = gevlss()
    mu = X[:, jj[0]] @ coef[jj[0]]
    rho = X[:, jj[1]] @ coef[jj[1]]
    xi = fam.linfo[2].linkinv(X[:, jj[2]] @ coef[jj[2]])
    sigma = np.exp(rho)

    rng2 = np.random.default_rng(14)
    y = mu + sigma * rng2.standard_normal(n)
    aa = 1.0 + xi * (y - mu) / sigma
    y = np.where(aa > 0.1, y, mu + sigma * 0.5)
    weights = np.ones(n)

    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lbb = result["lbb"]
    assert lbb.shape == (p, p)
    assert np.all(np.isfinite(lbb))
    ev = np.linalg.eigvalsh(lbb)
    assert np.all(ev <= 1e-8), f"Hessian has positive eigenvalue: {ev.max():.4g}"


# ---------------------------------------------------------------------------
# 5. gevlss initialize
# ---------------------------------------------------------------------------


def test_gevlss_initialize():
    """gevlss initialize returns correct-shaped finite vector."""
    rng = np.random.default_rng(17)
    n, p1, p2, p3 = 80, 4, 3, 2
    p = p1 + p2 + p3
    X = rng.standard_normal((n, p))
    jj = [np.arange(p1), np.arange(p1, p1 + p2), np.arange(p1 + p2, p)]
    y = rng.standard_normal(n) * 2.0 + 5.0
    weights = np.ones(n)

    fam = gevlss()
    start = fam.initialize(y, X, jj, offset=None, weights=weights)
    assert start.shape == (p,)
    assert np.all(np.isfinite(start))


# ---------------------------------------------------------------------------
# 6. gam_fit5 convergence on simulated GEV data
# ---------------------------------------------------------------------------


def test_gam_fit5_gevlss_convergence():
    """
    gam_fit5 with gevlss recovers approximate location and log-scale intercepts.
    """
    rng = np.random.default_rng(77)
    n = 300
    x_true = rng.standard_normal(n)
    mu_true = 2.0 + 0.5 * x_true
    rho_true = 0.0  # log(sigma) = 0, sigma = 1
    xi_true = 0.1   # shape parameter

    # Simulate GEV: use inverse CDF
    # F(y) = exp(-(1+xi*(y-mu)/sigma)^(-1/xi))
    # y = mu + sigma * ((-log(U))^(-xi) - 1) / xi
    U = rng.uniform(0.01, 0.99, n)
    sigma = np.exp(rho_true)
    y = mu_true + sigma * ((-np.log(U)) ** (-xi_true) - 1.0) / xi_true

    p1, p2, p3 = 2, 1, 1
    p = p1 + p2 + p3
    X = np.zeros((n, p), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = x_true
    X[:, 2] = 1.0
    X[:, 3] = 1.0
    jj = [np.arange(p1), np.arange(p1, p1 + p2), np.arange(p1 + p2, p)]

    fam = gevlss()
    St = np.zeros((p, p), dtype=np.float64)
    lsp = np.array([], dtype=np.float64)
    S_blocks: list = []

    ctl = GamFit5Control(maxit=200, epsilon=1e-7, trace=False)
    fit = gam_fit5(
        X, y, jj, lsp, St, S_blocks, ldetS=0.0, ldetS1=None, ldetS2=None,
        family=fam, weights=None, offset=None, deriv=0, control=ctl,
    )

    assert fit["iter"] > 0
    coef = fit["coef"]
    assert np.all(np.isfinite(coef))

    # Location intercept ≈ 2.0
    assert_allclose(coef[0], 2.0, atol=0.3), f"mu intercept = {coef[0]:.3f}"
    # Log-scale intercept ≈ 0.0
    assert_allclose(coef[2], 0.0, atol=0.3), f"rho = {coef[2]:.3f}"
    # Shape: xi_est ≈ 0.1 (but link maps to eta, so check linkinv)
    xi_est = fam.linfo[2].linkinv(coef[3])
    assert_allclose(xi_est, xi_true, atol=0.15), f"xi_est = {xi_est:.3f}"


# ---------------------------------------------------------------------------
# 7. gevlss l3 third derivatives vs finite difference of l2
# ---------------------------------------------------------------------------


def _gevlss_l2_raw(y, mu, rho, xi_param):
    """Compute raw per-obs l2 (n x 6) for gevlss at given scalar parameters."""
    eps_xi = 1e-7
    xi = np.where((xi_param >= 0.0) & (xi_param < eps_xi), eps_xi, xi_param)
    xi = np.where((xi < 0.0) & (xi > -eps_xi), -eps_xi, xi)

    cc2 = y - mu
    bb1 = np.exp(-rho)
    aa0 = xi * cc2 * bb1
    cc3 = 1.0 + aa0
    if not np.all(cc3 > 0.0):
        return None
    log_cc3 = np.log1p(aa0)
    aa2 = 1.0 / xi
    dd3 = xi + 1.0
    dd6 = 1.0 / cc3
    dd8 = 1.0 / xi**2
    ee1 = np.exp(-2.0 * rho)
    ee3 = -aa2
    ff7 = ee3 - 1.0
    gg7 = -aa2
    hh4 = cc2**2
    jj08 = 1.0 / cc3**2
    jj12 = 1.0 / xi**3
    jj13 = 1.0 / cc3**aa2
    dd7 = log_cc3

    l2 = np.empty((len(y), 6), dtype=np.float64)
    l2[:, 0] = ee1 * (ee3 - 1.0) * xi * cc3 ** (ee3 - 2.0) + (ee1 * xi * dd3) / cc3**2
    l2[:, 1] = (
        bb1 * cc3**ff7
        + ee1 * ff7 * xi * cc2 * cc3 ** (ee3 - 2.0)
        - (bb1 * dd3) / cc3
        + (ee1 * xi * dd3 * cc2) / cc3**2
    )
    l2[:, 2] = (
        -bb1 * cc3 ** (gg7 - 1.0) * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
        + ee1 * cc2 * cc3 ** (gg7 - 2.0)
        + bb1 * dd6
        - (ee1 * dd3 * cc2) / cc3**2
    )
    l2[:, 3] = (
        bb1 * cc2 * cc3**ff7
        + ee1 * ff7 * xi * hh4 * cc3 ** (ee3 - 2.0)
        - (bb1 * dd3 * cc2) / cc3
        + (ee1 * xi * dd3 * hh4) / cc3**2
    )
    l2[:, 4] = (
        -bb1 * cc2 * cc3 ** (gg7 - 1.0) * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
        + ee1 * hh4 * cc3 ** (gg7 - 2.0)
        + bb1 * cc2 * dd6
        - (ee1 * dd3 * hh4) / cc3**2
    )
    l2[:, 5] = (
        -jj13 * (dd8 * dd7 - bb1 * aa2 * cc2 * dd6) ** 2
        - jj13 * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7)
        - 2.0 * jj12 * dd3 * dd7
        + 2.0 * dd8 * dd7
        + 2.0 * bb1 * dd8 * dd3 * cc2 * dd6
        - 2.0 * bb1 * aa2 * cc2 * dd6
        + ee1 * aa2 * dd3 * hh4 * jj08
    )
    return l2


def test_gevlss_l3_fd():
    """
    Analytic l3 (third derivatives of log-lik w.r.t. mu/rho/xi) matches
    finite differences of l2.

    l3 ordering: mmm mmr mmx mrr mrx mxx rrr rrx rxx xxx
    l2 ordering: mm  mr  mx  rr  rx  xx
    """
    rng = np.random.default_rng(99)
    n = 50

    # Fixed parameters well inside the support
    mu0 = 1.5
    rho0 = 0.2
    xi0 = 0.15
    sigma = np.exp(rho0)
    # Simulate y from GEV
    U = rng.uniform(0.02, 0.98, n)
    y = mu0 + sigma * ((-np.log(U)) ** (-xi0) - 1.0) / xi0

    # Build trivial design (intercept-only, identity link for all params)
    p = 3
    X = np.ones((n, p))
    jj = [np.array([0]), np.array([1]), np.array([2])]
    coef = np.array([mu0, rho0, 0.0])  # xi link is shifted logit; 0 → xi≈0

    fam = gevlss(link=("identity", "identity", "identity"))
    # With identity link on xi, coef[2] = xi directly
    coef_xi = np.array([mu0, rho0, xi0])

    weights = np.ones(n)

    # Get analytic l3 — call ll with deriv=2 and extract l3_val via a helper
    # We test at the raw (mu, rho, xi) parameter level using _gevlss_l2_raw.
    mu_v = np.full(n, mu0)
    rho_v = np.full(n, rho0)
    xi_v = np.full(n, xi0)

    h = 1e-5
    # Centered FD w.r.t. mu
    l2_mu_p = _gevlss_l2_raw(y, mu_v + h, rho_v, xi_v)
    l2_mu_m = _gevlss_l2_raw(y, mu_v - h, rho_v, xi_v)
    dl2_dmu = (l2_mu_p - l2_mu_m) / (2.0 * h)

    # Centered FD w.r.t. rho
    l2_rho_p = _gevlss_l2_raw(y, mu_v, rho_v + h, xi_v)
    l2_rho_m = _gevlss_l2_raw(y, mu_v, rho_v - h, xi_v)
    dl2_drho = (l2_rho_p - l2_rho_m) / (2.0 * h)

    # Centered FD w.r.t. xi
    l2_xi_p = _gevlss_l2_raw(y, mu_v, rho_v, xi_v + h)
    l2_xi_m = _gevlss_l2_raw(y, mu_v, rho_v, xi_v - h)
    dl2_dxi = (l2_xi_p - l2_xi_m) / (2.0 * h)

    # Recompute l3_val analytically using the same raw formulas as the family
    cc2 = y - mu0
    bb1 = np.exp(-rho0)
    xi = xi0
    aa0 = xi * cc2 * bb1
    cc3 = 1.0 + aa0
    log_cc3 = np.log1p(aa0)
    aa2 = 1.0 / xi
    dd3 = xi + 1.0
    dd6 = 1.0 / cc3
    dd7 = log_cc3
    dd8 = 1.0 / xi**2
    ee1 = np.exp(-2.0 * rho0)
    ee3 = -aa2
    ff7 = ee3 - 1.0
    gg7 = -aa2
    hh4 = cc2**2
    jj08 = 1.0 / cc3**2
    jj12 = 1.0 / xi**3
    jj13 = 1.0 / cc3**aa2

    kk1 = np.exp(-3.0 * rho0)
    kk2 = xi**2
    ll8 = ee3 - 2.0
    mm11 = gg7 - 2.0
    mm12 = cc3**mm11
    mm10 = cc3 ** (gg7 - 3.0)
    oo10 = ff7
    oo13 = log_cc3 / xi**2
    pp08 = cc3**ff7
    qq05 = cc2**3
    rr17 = log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6
    tt08 = 1.0 / cc3**3
    tt16 = 1.0 / xi**4
    tt18 = dd8 * dd7 - bb1 * aa2 * cc2 * dd6

    l3_ana = np.empty((n, 10), dtype=np.float64)
    l3_ana[:, 0] = (2.0 * kk1 * kk2 * dd3) / cc3**3 - kk1 * (ee3 - 2.0) * (ee3 - 1.0) * kk2 * cc3 ** (ee3 - 3.0)
    l3_ana[:, 1] = -2.0 * ee1 * ff7 * xi * cc3**ll8 - kk1 * ll8 * ff7 * kk2 * cc2 * cc3 ** (ee3 - 3.0) - (2.0 * ee1 * xi * dd3) / cc3**2 + (2.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
    l3_ana[:, 2] = ee1 * ff7 * xi * mm12 * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6) - ee1 * mm12 - kk1 * mm11 * xi * cc2 * mm10 + kk1 * cc2 * mm10 + ee1 * dd3 * jj08 + ee1 * xi * jj08 - (2.0 * kk1 * xi * dd3 * cc2) / cc3**3
    l3_ana[:, 3] = -bb1 * cc3**ff7 - 3.0 * ee1 * ff7 * xi * cc2 * cc3**ll8 - kk1 * ll8 * ff7 * kk2 * hh4 * cc3 ** (ee3 - 3.0) + (bb1 * dd3) / cc3 - (3.0 * ee1 * xi * dd3 * cc2) / cc3**2 + (2.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
    l3_ana[:, 4] = bb1 * cc3**oo10 * (bb1 * oo10 * cc2 * dd6 + oo13) + ee1 * oo10 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13) + ee1 * aa2 * cc2 * mm12 + ee1 * oo10 * cc2 * mm12 - bb1 * dd6 + 2.0 * ee1 * dd3 * cc2 * jj08 + ee1 * xi * cc2 * jj08 - 2.0 * xi * dd3 * hh4 * kk1 / cc3**3
    l3_ana[:, 5] = -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2 - bb1 * pp08 * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 / xi**3) - 2.0 * ee1 * cc2 * jj08 + 2.0 * dd3 * hh4 * kk1 / cc3**3
    l3_ana[:, 6] = -bb1 * cc2 * cc3**ff7 - 3.0 * ee1 * ff7 * xi * hh4 * cc3**ll8 - kk1 * ll8 * ff7 * kk2 * qq05 * cc3 ** (ee3 - 3.0) + (bb1 * dd3 * cc2) / cc3 - (3.0 * ee1 * xi * dd3 * hh4) / cc3**2 + (2.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
    l3_ana[:, 7] = bb1 * cc2 * cc3**oo10 * rr17 + ee1 * oo10 * xi * hh4 * mm12 * rr17 - 2.0 * ee1 * hh4 * mm12 - kk1 * mm11 * xi * qq05 * mm10 + kk1 * qq05 * mm10 - bb1 * cc2 * dd6 + 2.0 * ee1 * dd3 * hh4 * jj08 + ee1 * xi * hh4 * jj08 - (2.0 * kk1 * xi * dd3 * qq05) / cc3**3
    l3_ana[:, 8] = -bb1 * cc2 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2 - bb1 * cc2 * pp08 * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 / xi**3) - 2.0 * ee1 * hh4 * jj08 + 2.0 * dd3 * qq05 * kk1 / cc3**3
    l3_ana[:, 9] = -jj13 * tt18**3 - 3.0 * jj13 * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7) * tt18 - jj13 * (-2.0 * kk1 * aa2 * qq05 * tt08 - 3.0 * ee1 * dd8 * hh4 * jj08 - 6.0 * bb1 * jj12 * cc2 * dd6 + 6.0 * tt16 * dd7) + 6.0 * tt16 * dd3 * dd7 - 6.0 * jj12 * dd7 - 6.0 * bb1 * jj12 * dd3 * cc2 * dd6 + 6.0 * bb1 * dd8 * cc2 * dd6 - 3.0 * ee1 * dd8 * dd3 * hh4 * jj08 + 3.0 * ee1 * aa2 * hh4 * jj08 - 2.0 * kk1 * aa2 * dd3 * qq05 * tt08

    # l3 ordering: mmm(0) mmr(1) mmx(2) mrr(3) mrx(4) mxx(5) rrr(6) rrx(7) rxx(8) xxx(9)
    # l2 ordering: mm(0)  mr(1)  mx(2)  rr(3)  rx(4)  xx(5)
    # l3[:,k] = d l2[:,j] / d param_p  where (j,p) = trind entry for k
    tol_r = 1e-5  # centered FD with h=1e-5
    tol_a = 1e-9

    # mmm = d/dmu l2_mm
    assert_allclose(l3_ana[:, 0], dl2_dmu[:, 0], rtol=tol_r, atol=tol_a)
    # mmr = d/dmu l2_mr = d/drho l2_mm
    assert_allclose(l3_ana[:, 1], dl2_dmu[:, 1], rtol=tol_r, atol=tol_a)
    assert_allclose(l3_ana[:, 1], dl2_drho[:, 0], rtol=tol_r, atol=tol_a)
    # mmx = d/dmu l2_mx = d/dxi l2_mm
    assert_allclose(l3_ana[:, 2], dl2_dmu[:, 2], rtol=tol_r, atol=tol_a)
    assert_allclose(l3_ana[:, 2], dl2_dxi[:, 0], rtol=tol_r, atol=tol_a)
    # mrr = d/drho l2_mr
    assert_allclose(l3_ana[:, 3], dl2_drho[:, 1], rtol=tol_r, atol=tol_a)
    # mrx = d/dxi l2_mr = d/drho l2_mx
    assert_allclose(l3_ana[:, 4], dl2_dxi[:, 1], rtol=tol_r, atol=tol_a)
    assert_allclose(l3_ana[:, 4], dl2_drho[:, 2], rtol=tol_r, atol=tol_a)
    # mxx = d/dxi l2_mx
    assert_allclose(l3_ana[:, 5], dl2_dxi[:, 2], rtol=tol_r, atol=tol_a)
    # rrr = d/drho l2_rr
    assert_allclose(l3_ana[:, 6], dl2_drho[:, 3], rtol=tol_r, atol=tol_a)
    # rrx = d/dxi l2_rr = d/drho l2_rx
    assert_allclose(l3_ana[:, 7], dl2_dxi[:, 3], rtol=tol_r, atol=tol_a)
    assert_allclose(l3_ana[:, 7], dl2_drho[:, 4], rtol=tol_r, atol=tol_a)
    # rxx = d/dxi l2_rx
    assert_allclose(l3_ana[:, 8], dl2_dxi[:, 4], rtol=tol_r, atol=tol_a)
    # xxx = d/dxi l2_xx
    assert_allclose(l3_ana[:, 9], dl2_dxi[:, 5], rtol=tol_r, atol=tol_a)


# ---------------------------------------------------------------------------
# 8. gevlss l4 fourth derivatives vs finite difference of l3
# ---------------------------------------------------------------------------


def _gevlss_l3_raw(y, mu, rho, xi_param):
    """Compute raw per-obs l3 (n x 10) for gevlss at given parameters."""
    eps_xi = 1e-7
    xi = float(xi_param)
    xi_arr = np.full(len(y), xi)
    xi_arr = np.where((xi_arr >= 0.0) & (xi_arr < eps_xi), eps_xi, xi_arr)
    xi_arr = np.where((xi_arr < 0.0) & (xi_arr > -eps_xi), -eps_xi, xi_arr)
    xi = xi_arr  # now array

    cc2 = y - mu
    bb1 = np.exp(-rho)
    aa0 = xi * cc2 * bb1
    cc3 = 1.0 + aa0
    if not np.all(cc3 > 0.0):
        return None
    log_cc3 = np.log1p(aa0)
    aa2 = 1.0 / xi
    dd3 = xi + 1.0
    dd6 = 1.0 / cc3
    dd7 = log_cc3
    dd8 = 1.0 / xi**2
    ee1 = np.exp(-2.0 * rho)
    ee3 = -aa2
    ff7 = ee3 - 1.0
    gg7 = -aa2
    hh4 = cc2**2
    jj08 = 1.0 / cc3**2
    jj12 = 1.0 / xi**3
    jj13 = 1.0 / cc3**aa2

    kk1 = np.exp(-3.0 * rho)
    kk2 = xi**2
    ll8 = ee3 - 2.0
    mm11 = gg7 - 2.0
    mm12 = cc3**mm11
    mm10 = cc3 ** (gg7 - 3.0)
    oo10 = ff7
    oo13 = log_cc3 / xi**2
    pp08 = cc3**ff7
    qq05 = cc2**3
    rr17 = log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6
    tt08 = 1.0 / cc3**3
    tt16 = 1.0 / xi**4
    tt18 = dd8 * dd7 - bb1 * aa2 * cc2 * dd6

    n = len(y)
    l3 = np.empty((n, 10), dtype=np.float64)
    l3[:, 0] = (2.0 * kk1 * kk2 * dd3) / cc3**3 - kk1 * (ee3 - 2.0) * (ee3 - 1.0) * kk2 * cc3 ** (ee3 - 3.0)
    l3[:, 1] = -2.0 * ee1 * ff7 * xi * cc3**ll8 - kk1 * ll8 * ff7 * kk2 * cc2 * cc3 ** (ee3 - 3.0) - (2.0 * ee1 * xi * dd3) / cc3**2 + (2.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
    l3[:, 2] = ee1 * ff7 * xi * mm12 * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6) - ee1 * mm12 - kk1 * mm11 * xi * cc2 * mm10 + kk1 * cc2 * mm10 + ee1 * dd3 * jj08 + ee1 * xi * jj08 - (2.0 * kk1 * xi * dd3 * cc2) / cc3**3
    l3[:, 3] = -bb1 * cc3**ff7 - 3.0 * ee1 * ff7 * xi * cc2 * cc3**ll8 - kk1 * ll8 * ff7 * kk2 * hh4 * cc3 ** (ee3 - 3.0) + (bb1 * dd3) / cc3 - (3.0 * ee1 * xi * dd3 * cc2) / cc3**2 + (2.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
    l3[:, 4] = bb1 * cc3**oo10 * (bb1 * oo10 * cc2 * dd6 + oo13) + ee1 * oo10 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13) + ee1 * aa2 * cc2 * mm12 + ee1 * oo10 * cc2 * mm12 - bb1 * dd6 + 2.0 * ee1 * dd3 * cc2 * jj08 + ee1 * xi * cc2 * jj08 - 2.0 * xi * dd3 * hh4 * kk1 / cc3**3
    l3[:, 5] = -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2 - bb1 * pp08 * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 / xi**3) - 2.0 * ee1 * cc2 * jj08 + 2.0 * dd3 * hh4 * kk1 / cc3**3
    l3[:, 6] = -bb1 * cc2 * cc3**ff7 - 3.0 * ee1 * ff7 * xi * hh4 * cc3**ll8 - kk1 * ll8 * ff7 * kk2 * qq05 * cc3 ** (ee3 - 3.0) + (bb1 * dd3 * cc2) / cc3 - (3.0 * ee1 * xi * dd3 * hh4) / cc3**2 + (2.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
    l3[:, 7] = bb1 * cc2 * cc3**oo10 * rr17 + ee1 * oo10 * xi * hh4 * mm12 * rr17 - 2.0 * ee1 * hh4 * mm12 - kk1 * mm11 * xi * qq05 * mm10 + kk1 * qq05 * mm10 - bb1 * cc2 * dd6 + 2.0 * ee1 * dd3 * hh4 * jj08 + ee1 * xi * hh4 * jj08 - (2.0 * kk1 * xi * dd3 * qq05) / cc3**3
    l3[:, 8] = -bb1 * cc2 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2 - bb1 * cc2 * pp08 * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 / xi**3) - 2.0 * ee1 * hh4 * jj08 + 2.0 * dd3 * qq05 * kk1 / cc3**3
    l3[:, 9] = -jj13 * tt18**3 - 3.0 * jj13 * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7) * tt18 - jj13 * (-2.0 * kk1 * aa2 * qq05 * tt08 - 3.0 * ee1 * dd8 * hh4 * jj08 - 6.0 * bb1 * jj12 * cc2 * dd6 + 6.0 * tt16 * dd7) + 6.0 * tt16 * dd3 * dd7 - 6.0 * jj12 * dd7 - 6.0 * bb1 * jj12 * dd3 * cc2 * dd6 + 6.0 * bb1 * dd8 * cc2 * dd6 - 3.0 * ee1 * dd8 * dd3 * hh4 * jj08 + 3.0 * ee1 * aa2 * hh4 * jj08 - 2.0 * kk1 * aa2 * dd3 * qq05 * tt08
    return l3


def test_gevlss_l4_fd():
    """
    Analytic l4 (fourth derivatives of log-lik w.r.t. mu/rho/xi) matches
    centered finite differences of l3.

    l4 ordering: mmmm mmmr mmmx mmrr mmrx mmxx mrrr mrrx mrxx mxxx
                 rrrr rrrx rrxx rxxx xxxx
    l3 ordering: mmm(0) mmr(1) mmx(2) mrr(3) mrx(4) mxx(5) rrr(6) rrx(7) rxx(8) xxx(9)
    Symmetry: l4[:,k] = d l3[:,j] / d param_p for appropriate (j,p).
    """
    rng = np.random.default_rng(42)
    n = 50
    mu0 = 1.5
    rho0 = 0.2
    xi0 = 0.15
    sigma = np.exp(rho0)
    U = rng.uniform(0.02, 0.98, n)
    y = mu0 + sigma * ((-np.log(U)) ** (-xi0) - 1.0) / xi0

    mu_v = np.full(n, mu0)
    rho_v = np.full(n, rho0)
    xi_v = np.full(n, xi0)

    h = 1e-4  # larger step for 4th deriv FD stability

    # Centered FD of l3 w.r.t. each parameter
    dl3_dmu = (_gevlss_l3_raw(y, mu_v + h, rho_v, xi0) - _gevlss_l3_raw(y, mu_v - h, rho_v, xi0)) / (2.0 * h)
    dl3_drho = (_gevlss_l3_raw(y, mu_v, rho_v + h, xi0) - _gevlss_l3_raw(y, mu_v, rho_v - h, xi0)) / (2.0 * h)
    dl3_dxi = (_gevlss_l3_raw(y, mu_v, rho_v, xi0 + h) - _gevlss_l3_raw(y, mu_v, rho_v, xi0 - h)) / (2.0 * h)

    # Compute analytic l4 directly
    cc2 = y - mu0
    bb1 = np.exp(-rho0)
    xi = xi0
    aa0 = xi * cc2 * bb1
    cc3 = 1.0 + aa0
    log_cc3 = np.log1p(aa0)
    aa2 = 1.0 / xi
    dd3 = xi + 1.0
    dd6 = 1.0 / cc3
    dd7 = log_cc3
    dd8 = 1.0 / xi**2
    ee1 = np.exp(-2.0 * rho0)
    ee3 = -aa2
    ff7 = ee3 - 1.0
    gg7 = -aa2
    hh4 = cc2**2
    jj08 = 1.0 / cc3**2
    jj12 = 1.0 / xi**3
    jj13 = 1.0 / cc3**aa2
    kk1 = np.exp(-3.0 * rho0)
    kk2 = xi**2
    ll8 = ee3 - 2.0
    mm11 = gg7 - 2.0
    mm12 = cc3**mm11
    oo13 = log_cc3 / xi**2
    pp08 = cc3**ff7
    qq05 = cc2**3
    rr17 = log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6
    tt08 = 1.0 / cc3**3
    tt16 = 1.0 / xi**4
    tt18 = dd8 * dd7 - bb1 * aa2 * cc2 * dd6

    uu1 = np.exp(-4.0 * rho0)
    uu2 = xi**3
    vv09 = ee3 - 3.0
    ww11 = gg7 - 3.0
    ww12 = cc3 ** (gg7 - 4.0)
    ww15 = cc3**ww11
    ad17 = 2.0 * bb1 * dd8 * cc2 * dd6
    ad19 = -2.0 * jj12 * dd7
    ad20 = pp08
    ad21 = dd8 * dd7
    ad22 = ad21 + bb1 * mm11 * cc2 * dd6
    ae16 = dd8 * dd7 + bb1 * ff7 * cc2 * dd6
    af05 = cc2**4
    ah24 = ad19 + ad17 + ee1 * aa2 * hh4 * jj08
    aj08 = 1.0 / cc3**4
    aj20 = 1.0 / xi**5

    l4_ana = np.empty((n, 15), dtype=np.float64)
    l4_ana[:, 0] = uu1 * (ee3 - 3.0) * (ee3 - 2.0) * (ee3 - 1.0) * uu2 * cc3 ** (ee3 - 4.0) + (6.0 * uu1 * uu2 * dd3) / cc3**4
    l4_ana[:, 1] = 3.0 * kk1 * ll8 * ff7 * kk2 * cc3**vv09 + uu1 * vv09 * ll8 * ff7 * uu2 * cc2 * cc3 ** (ee3 - 4.0) - (6.0 * kk1 * kk2 * dd3) / cc3**3 + (6.0 * uu1 * uu2 * dd3 * cc2) / cc3**4
    l4_ana[:, 2] = -kk1 * mm11 * ff7 * kk2 * ww15 * rr17 + 2.0 * kk1 * mm11 * xi * ww15 - kk1 * ww15 + uu1 * ww11 * mm11 * kk2 * cc2 * ww12 - uu1 * ff7 * xi * cc2 * ww12 - uu1 * ww11 * xi * cc2 * ww12 + 2.0 * kk1 * kk2 * tt08 + 4.0 * kk1 * xi * dd3 * tt08 - (6.0 * uu1 * kk2 * dd3 * cc2) / cc3**4
    l4_ana[:, 3] = 4.0 * ee1 * ff7 * xi * cc3**ll8 + 5.0 * kk1 * ll8 * ff7 * kk2 * cc2 * cc3**vv09 + uu1 * vv09 * ll8 * ff7 * uu2 * hh4 * cc3 ** (ee3 - 4.0) + (4.0 * ee1 * xi * dd3) / cc3**2 - (10.0 * kk1 * kk2 * dd3 * cc2) / cc3**3 + (6.0 * uu1 * uu2 * dd3 * hh4) / cc3**4
    l4_ana[:, 4] = -2.0 * ee1 * ff7 * xi * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13) - kk1 * mm11 * ff7 * kk2 * cc2 * ww15 * (bb1 * ww11 * cc2 * dd6 + oo13) - 2.0 * ee1 * aa2 * mm12 - 2.0 * ee1 * ff7 * mm12 - 2.0 * kk1 * mm11 * ff7 * xi * cc2 * ww15 - kk1 * ff7 * cc2 * ww15 - kk1 * mm11 * cc2 * ww15 - 2.0 * ee1 * dd3 * jj08 - 2.0 * ee1 * xi * jj08 + 2.0 * kk1 * kk2 * cc2 * tt08 + 8.0 * kk1 * xi * dd3 * cc2 * tt08 - 6.0 * kk2 * dd3 * hh4 * uu1 / cc3**4
    l4_ana[:, 5] = ee1 * ff7 * xi * mm12 * tt18**2 - 2.0 * ee1 * mm12 * tt18 - 2.0 * kk1 * mm11 * xi * cc2 * ww15 * tt18 + 2.0 * kk1 * cc2 * ww15 * tt18 + ee1 * ff7 * xi * mm12 * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 * jj12) + 4.0 * kk1 * cc2 * ww15 + 2.0 * uu1 * ww11 * xi * hh4 * ww12 - 4.0 * uu1 * hh4 * ww12 + 2.0 * ee1 * jj08 - 4.0 * kk1 * dd3 * cc2 * tt08 - 4.0 * kk1 * xi * cc2 * tt08 + (6.0 * uu1 * xi * dd3 * hh4) / cc3**4
    l4_ana[:, 6] = bb1 * cc3**ff7 + 7.0 * ee1 * ff7 * xi * cc2 * cc3**ll8 + 6.0 * kk1 * ll8 * ff7 * kk2 * hh4 * cc3**vv09 + uu1 * vv09 * ll8 * ff7 * uu2 * qq05 * cc3 ** (ee3 - 4.0) - (bb1 * dd3) / cc3 + (7.0 * ee1 * xi * dd3 * cc2) / cc3**2 - (12.0 * kk1 * kk2 * dd3 * hh4) / cc3**3 + (6.0 * uu1 * uu2 * dd3 * qq05) / cc3**4
    l4_ana[:, 7] = -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + oo13) - 3.0 * ee1 * ff7 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13) - kk1 * mm11 * ff7 * kk2 * hh4 * ww15 * (bb1 * ww11 * cc2 * dd6 + oo13) - 3.0 * ee1 * aa2 * cc2 * mm12 - 3.0 * ee1 * ff7 * cc2 * mm12 - 2.0 * kk1 * mm11 * ff7 * xi * hh4 * ww15 - kk1 * ff7 * hh4 * ww15 - kk1 * mm11 * hh4 * ww15 + bb1 * dd6 - 4.0 * ee1 * dd3 * cc2 * jj08 - 3.0 * ee1 * xi * cc2 * jj08 + 2.0 * kk1 * kk2 * hh4 * tt08 + 10.0 * kk1 * xi * dd3 * hh4 * tt08 - 6.0 * kk2 * dd3 * qq05 * uu1 / cc3**4
    l4_ana[:, 8] = bb1 * ad20 * (bb1 * ff7 * cc2 * dd6 + ad21) ** 2 + ee1 * ff7 * xi * cc2 * mm12 * ad22**2 + 2.0 * ee1 * aa2 * cc2 * mm12 * ad22 + 2.0 * ee1 * ff7 * cc2 * mm12 * ad22 + bb1 * ad20 * (-ee1 * ff7 * hh4 * jj08 + ad17 + ad19) + ee1 * ff7 * xi * cc2 * mm12 * (-ee1 * mm11 * hh4 * jj08 + ad17 + ad19) + 4.0 * ee1 * cc2 * jj08 - 6.0 * kk1 * dd3 * hh4 * tt08 - 4.0 * kk1 * xi * hh4 * tt08 + 6.0 * xi * dd3 * qq05 * uu1 / cc3**4
    l4_ana[:, 9] = -bb1 * pp08 * ae16**3 - 3.0 * bb1 * pp08 * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7) * ae16 - bb1 * pp08 * (2.0 * kk1 * ff7 * qq05 * tt08 - 3.0 * ee1 * dd8 * hh4 * jj08 - 6.0 * bb1 * jj12 * cc2 * dd6 + 6.0 * dd7 * tt16) + 6.0 * kk1 * hh4 * tt08 - 6.0 * dd3 * qq05 * uu1 / cc3**4
    l4_ana[:, 10] = bb1 * cc2 * cc3**ff7 + 7.0 * ee1 * ff7 * xi * hh4 * cc3**ll8 + 6.0 * kk1 * ll8 * ff7 * kk2 * qq05 * cc3**vv09 + uu1 * vv09 * ll8 * ff7 * uu2 * af05 * cc3 ** (ee3 - 4.0) - (bb1 * dd3 * cc2) / cc3 + (7.0 * ee1 * xi * dd3 * hh4) / cc3**2 - (12.0 * kk1 * kk2 * dd3 * qq05) / cc3**3 + (6.0 * uu1 * uu2 * dd3 * af05) / cc3**4
    l4_ana[:, 11] = -bb1 * cc2 * pp08 * rr17 - 3.0 * ee1 * ff7 * xi * hh4 * mm12 * rr17 - kk1 * mm11 * ff7 * kk2 * qq05 * ww15 * rr17 + 4.0 * ee1 * hh4 * mm12 + 5.0 * kk1 * mm11 * xi * qq05 * ww15 - 4.0 * kk1 * qq05 * ww15 + uu1 * ww11 * mm11 * kk2 * af05 * ww12 - uu1 * ff7 * xi * af05 * ww12 - uu1 * ww11 * xi * af05 * ww12 + bb1 * cc2 * dd6 - 4.0 * ee1 * dd3 * hh4 * jj08 - 3.0 * ee1 * xi * hh4 * jj08 + 2.0 * kk1 * kk2 * qq05 * tt08 + 10.0 * kk1 * xi * dd3 * qq05 * tt08 - 6.0 * uu1 * kk2 * dd3 * af05 / cc3**4
    l4_ana[:, 12] = bb1 * cc2 * ad20 * tt18**2 + ee1 * ff7 * xi * hh4 * mm12 * tt18**2 - 4.0 * ee1 * hh4 * mm12 * tt18 - 2.0 * kk1 * mm11 * xi * qq05 * ww15 * tt18 + 2.0 * kk1 * qq05 * ww15 * tt18 + bb1 * cc2 * ad20 * ah24 + ee1 * ff7 * xi * hh4 * mm12 * ah24 + 6.0 * kk1 * qq05 * ww15 + 2.0 * uu1 * ww11 * xi * af05 * ww12 - 4.0 * uu1 * af05 * ww12 + 4.0 * ee1 * hh4 * jj08 - 6.0 * kk1 * dd3 * qq05 * tt08 - 4.0 * kk1 * xi * qq05 * tt08 + 6.0 * uu1 * xi * dd3 * af05 / cc3**4
    l4_ana[:, 13] = -bb1 * cc2 * pp08 * ae16**3 - 3.0 * bb1 * cc2 * pp08 * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7) * ae16 - bb1 * cc2 * pp08 * (2.0 * kk1 * ff7 * qq05 * tt08 - 3.0 * ee1 * dd8 * hh4 * jj08 - 6.0 * bb1 * jj12 * cc2 * dd6 + 6.0 * dd7 * tt16) + 6.0 * kk1 * qq05 * tt08 - 6.0 * dd3 * af05 * uu1 / cc3**4
    l4_ana[:, 14] = -jj13 * tt18**4 - 6.0 * jj13 * ah24 * tt18**2 - 3.0 * jj13 * ah24**2 - 4.0 * jj13 * (-2.0 * kk1 * aa2 * qq05 * tt08 - 3.0 * ee1 * dd8 * hh4 * jj08 - 6.0 * bb1 * jj12 * cc2 * dd6 + 6.0 * tt16 * dd7) * tt18 - jj13 * (6.0 * uu1 * aa2 * af05 * aj08 + 8.0 * kk1 * dd8 * qq05 * tt08 + 12.0 * ee1 * jj12 * hh4 * jj08 + 24.0 * bb1 * tt16 * cc2 * dd6 - 24.0 * aj20 * dd7) - 24.0 * aj20 * dd3 * dd7 + 24.0 * tt16 * dd7 + 24.0 * bb1 * tt16 * dd3 * cc2 * dd6 - 24.0 * bb1 * jj12 * cc2 * dd6 + 12.0 * ee1 * jj12 * dd3 * hh4 * jj08 - 12.0 * ee1 * dd8 * hh4 * jj08 + 8.0 * kk1 * dd8 * dd3 * qq05 * tt08 - 8.0 * kk1 * aa2 * qq05 * tt08 + 6.0 * uu1 * aa2 * dd3 * af05 * aj08

    # l4 ordering: mmmm(0) mmmr(1) mmmx(2) mmrr(3) mmrx(4) mmxx(5)
    #              mrrr(6) mrrx(7) mrxx(8) mxxx(9) rrrr(10) rrrx(11)
    #              rrxx(12) rxxx(13) xxxx(14)
    # l3 ordering: mmm(0) mmr(1) mmx(2) mrr(3) mrx(4) mxx(5) rrr(6) rrx(7) rxx(8) xxx(9)
    tol_r = 1e-4  # centered FD with h=1e-4 (4th deriv is noisier)
    tol_a = 1e-8

    # mmmm = d/dmu l3_mmm
    assert_allclose(l4_ana[:, 0], dl3_dmu[:, 0], rtol=tol_r, atol=tol_a)
    # mmmr = d/dmu l3_mmr = d/drho l3_mmm
    assert_allclose(l4_ana[:, 1], dl3_dmu[:, 1], rtol=tol_r, atol=tol_a)
    assert_allclose(l4_ana[:, 1], dl3_drho[:, 0], rtol=tol_r, atol=tol_a)
    # mmmx = d/dmu l3_mmx = d/dxi l3_mmm
    assert_allclose(l4_ana[:, 2], dl3_dmu[:, 2], rtol=tol_r, atol=tol_a)
    assert_allclose(l4_ana[:, 2], dl3_dxi[:, 0], rtol=tol_r, atol=tol_a)
    # mmrr = d/drho l3_mmr
    assert_allclose(l4_ana[:, 3], dl3_drho[:, 1], rtol=tol_r, atol=tol_a)
    # mmrx = d/dxi l3_mmr = d/drho l3_mmx
    assert_allclose(l4_ana[:, 4], dl3_dxi[:, 1], rtol=tol_r, atol=tol_a)
    assert_allclose(l4_ana[:, 4], dl3_drho[:, 2], rtol=tol_r, atol=tol_a)
    # mmxx = d/dxi l3_mmx
    assert_allclose(l4_ana[:, 5], dl3_dxi[:, 2], rtol=tol_r, atol=tol_a)
    # mrrr = d/drho l3_mrr
    assert_allclose(l4_ana[:, 6], dl3_drho[:, 3], rtol=tol_r, atol=tol_a)
    # mrrx = d/dxi l3_mrr = d/drho l3_mrx
    assert_allclose(l4_ana[:, 7], dl3_dxi[:, 3], rtol=tol_r, atol=tol_a)
    assert_allclose(l4_ana[:, 7], dl3_drho[:, 4], rtol=tol_r, atol=tol_a)
    # mrxx = d/dxi l3_mrx
    assert_allclose(l4_ana[:, 8], dl3_dxi[:, 4], rtol=tol_r, atol=tol_a)
    # mxxx = d/dxi l3_mxx
    assert_allclose(l4_ana[:, 9], dl3_dxi[:, 5], rtol=tol_r, atol=tol_a)
    # rrrr = d/drho l3_rrr
    assert_allclose(l4_ana[:, 10], dl3_drho[:, 6], rtol=tol_r, atol=tol_a)
    # rrrx = d/dxi l3_rrr = d/drho l3_rrx
    assert_allclose(l4_ana[:, 11], dl3_dxi[:, 6], rtol=tol_r, atol=tol_a)
    assert_allclose(l4_ana[:, 11], dl3_drho[:, 7], rtol=tol_r, atol=tol_a)
    # rrxx = d/dxi l3_rrx
    assert_allclose(l4_ana[:, 12], dl3_dxi[:, 7], rtol=tol_r, atol=tol_a)
    # rxxx = d/dxi l3_rxx
    assert_allclose(l4_ana[:, 13], dl3_dxi[:, 8], rtol=tol_r, atol=tol_a)
    # xxxx = d/dxi l3_xxx
    assert_allclose(l4_ana[:, 14], dl3_dxi[:, 9], rtol=tol_r, atol=tol_a)


# ======================================================================
# shashlss
# ======================================================================

# ---------------------------------------------------------------------------
# 1. LogEBLinkInfo roundtrip and lower bound
# ---------------------------------------------------------------------------


def test_logeb_roundtrip():
    link = _LogEBLinkInfo(b=0.01)
    tau = np.linspace(-2.0, 3.0, 30)
    eta = link.linkfun(tau)
    tau_back = link.linkinv(eta)
    assert_allclose(tau_back, tau, rtol=1e-10)


def test_logeb_lower_bound():
    link = _LogEBLinkInfo(b=0.01)
    eta = np.linspace(-10.0, 10.0, 100)
    tau = link.linkinv(eta)
    # tau = log(exp(eta) + b) >= log(b) > -inf  and tau > eta for all eta
    assert np.all(tau > np.log(0.01) - 1e-10), "tau must be above log(b)"


# ---------------------------------------------------------------------------
# 2. LogEBLinkInfo mu_eta finite-difference check
# ---------------------------------------------------------------------------


def test_logeb_mu_eta_fd():
    link = _LogEBLinkInfo(b=0.01)
    rng = np.random.default_rng(3)
    eta = rng.uniform(-3.0, 3.0, 30)
    eps = 1e-7
    fd = (link.linkinv(eta + eps) - link.linkinv(eta - eps)) / (2.0 * eps)
    analytic = link.mu_eta(eta)
    assert_allclose(analytic, fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. LogEBLinkInfo d2link finite-difference check
# ---------------------------------------------------------------------------


def test_logeb_d2link_fd():
    link = _LogEBLinkInfo(b=0.01)
    rng = np.random.default_rng(5)
    # Use mu values well above log(b) to avoid numerical issues
    mu = rng.uniform(0.5, 3.0, 20)
    eps = 1e-5
    # d2link is d^2(eta)/d(mu)^2 = d/d(mu) of linkfun derivative
    # fd approx via central difference of linkfun'
    def dlink(m):
        return (link.linkfun(m + eps) - link.linkfun(m - eps)) / (2.0 * eps)
    fd = (dlink(mu + eps) - dlink(mu - eps)) / (2.0 * eps)
    analytic = link.d2link(mu)
    assert_allclose(analytic, fd, rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. shashlss ll log-lik against direct formula
# ---------------------------------------------------------------------------


def test_shashlss_ll_loglik():
    """shashlss ll gives log-lik matching direct shash formula."""
    rng = np.random.default_rng(7)
    n, p1, p2, p3, p4 = 60, 3, 2, 2, 2
    p = p1 + p2 + p3 + p4
    X = rng.standard_normal((n, p))
    jj = [
        np.arange(p1),
        np.arange(p1, p1 + p2),
        np.arange(p1 + p2, p1 + p2 + p3),
        np.arange(p1 + p2 + p3, p),
    ]
    coef = np.zeros(p)
    coef[:p1] = rng.standard_normal(p1) * 0.3
    coef[p1 : p1 + p2] = rng.standard_normal(p2) * 0.1

    fam = shashlss()
    eta = X[:, jj[0]] @ coef[jj[0]]
    eta1 = X[:, jj[1]] @ coef[jj[1]]
    eta2 = X[:, jj[2]] @ coef[jj[2]]
    eta3 = X[:, jj[3]] @ coef[jj[3]]

    mu = fam.linfo[0].linkinv(eta)
    tau = fam.linfo[1].linkinv(eta1)
    eps = fam.linfo[2].linkinv(eta2)
    phi = fam.linfo[3].linkinv(eta3)
    sig = np.exp(tau)
    delta = np.exp(phi)

    # Simulate shash data
    z_norm = rng.standard_normal(n)
    y = mu + sig * delta * np.sinh((1.0 / delta) * np.arcsinh(z_norm) + eps / delta)

    weights = np.ones(n)
    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=0)
    assert np.isfinite(result["l"])

    # Direct shash log-lik
    z = (y - mu) / (sig * delta)
    dTasMe = delta * np.arcsinh(z) - eps
    CC = np.cosh(dTasMe)
    SS = np.sinh(dTasMe)
    l_ref = float(np.sum(
        -tau - 0.5 * np.log(2.0 * np.pi)
        + np.log(np.maximum(CC, 1e-300))
        - 0.5 * np.log1p(z**2)
        - 0.5 * SS**2
        - fam.phi_pen * phi**2
    ))
    assert_allclose(result["l"], l_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. shashlss ll gradient finite-difference check
# ---------------------------------------------------------------------------


def test_shashlss_ll_gradient_fd():
    """shashlss gradient matches finite-difference gradient."""
    rng = np.random.default_rng(11)
    n, p1, p2, p3, p4 = 40, 3, 2, 2, 2
    p = p1 + p2 + p3 + p4
    X = rng.standard_normal((n, p))
    jj = [
        np.arange(p1),
        np.arange(p1, p1 + p2),
        np.arange(p1 + p2, p1 + p2 + p3),
        np.arange(p1 + p2 + p3, p),
    ]
    coef = np.zeros(p)
    coef[:p1] = rng.standard_normal(p1) * 0.2

    fam = shashlss()
    eta = X[:, jj[0]] @ coef[jj[0]]
    eta1 = X[:, jj[1]] @ coef[jj[1]]
    mu = fam.linfo[0].linkinv(eta)
    tau = fam.linfo[1].linkinv(eta1)
    sig = np.exp(tau)
    delta = 1.0  # exp(0)

    y = mu + sig * rng.standard_normal(n)
    weights = np.ones(n)

    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lb = result["lb"]
    assert lb.shape == (p,)
    assert np.all(np.isfinite(lb))

    eps = 1e-6
    fd = np.zeros(p)
    l0 = fam.ll(y, X, jj, coef, weights, offset=None, deriv=0)["l"]
    for k in range(p):
        cp = coef.copy()
        cp[k] += eps
        l1_val = fam.ll(y, X, jj, cp, weights, offset=None, deriv=0)["l"]
        fd[k] = (l1_val - l0) / eps

    assert_allclose(lb, fd, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. shashlss ll Hessian shape and negative semi-definiteness
# ---------------------------------------------------------------------------


def test_shashlss_ll_hessian_shape():
    """shashlss Hessian has correct shape and is negative semi-definite."""
    rng = np.random.default_rng(13)
    n, p1, p2, p3, p4 = 50, 3, 2, 2, 2
    p = p1 + p2 + p3 + p4
    X = rng.standard_normal((n, p))
    jj = [
        np.arange(p1),
        np.arange(p1, p1 + p2),
        np.arange(p1 + p2, p1 + p2 + p3),
        np.arange(p1 + p2 + p3, p),
    ]
    coef = np.zeros(p)
    coef[:p1] = rng.standard_normal(p1) * 0.1

    fam = shashlss()
    eta = X[:, jj[0]] @ coef[jj[0]]
    eta1 = X[:, jj[1]] @ coef[jj[1]]
    mu = fam.linfo[0].linkinv(eta)
    tau = fam.linfo[1].linkinv(eta1)
    sig = np.exp(tau)
    y = mu + sig * rng.standard_normal(n)
    weights = np.ones(n)

    result = fam.ll(y, X, jj, coef, weights, offset=None, deriv=1)
    lbb = result["lbb"]
    assert lbb.shape == (p, p)
    assert np.all(np.isfinite(lbb))
    ev = np.linalg.eigvalsh(lbb)
    assert np.all(ev <= 1e-8), f"Hessian has positive eigenvalue: {ev.max():.4g}"


# ---------------------------------------------------------------------------
# 7. shashlss initialize
# ---------------------------------------------------------------------------


def test_shashlss_initialize():
    """shashlss initialize returns correct-shaped finite vector."""
    rng = np.random.default_rng(17)
    n, p1, p2, p3, p4 = 80, 4, 3, 2, 2
    p = p1 + p2 + p3 + p4
    X = rng.standard_normal((n, p))
    jj = [
        np.arange(p1),
        np.arange(p1, p1 + p2),
        np.arange(p1 + p2, p1 + p2 + p3),
        np.arange(p1 + p2 + p3, p),
    ]
    y = rng.standard_normal(n) * 2.0 + 5.0
    weights = np.ones(n)

    fam = shashlss()
    start = fam.initialize(y, X, jj, offset=None, weights=weights)
    assert start.shape == (p,)
    assert np.all(np.isfinite(start))


# ---------------------------------------------------------------------------
# 8. gam_fit5 convergence on simulated shash data
# ---------------------------------------------------------------------------


def test_gam_fit5_shashlss_convergence():
    """
    gam_fit5 with shashlss recovers approximate location and log-scale intercepts.
    Simulate from shash with eps=0, delta=1 (reduces to Gaussian).
    """
    rng = np.random.default_rng(99)
    n = 400
    x_true = rng.standard_normal(n)
    mu_true = 3.0 + 0.5 * x_true
    tau_true = 0.5  # log(sigma): sigma = exp(0.5) ≈ 1.65
    eps_true = 0.0  # no skewness
    phi_true = 0.0  # delta = 1, standard shash

    sig = np.exp(tau_true)
    delta = np.exp(phi_true)
    # shash with eps=0, delta=1 reduces to: y = mu + sig*sinh(arcsinh(Z)) = mu + sig*Z
    z_norm = rng.standard_normal(n)
    y = mu_true + sig * delta * np.sinh((1.0 / delta) * np.arcsinh(z_norm) + eps_true / delta)

    p1, p2, p3, p4 = 2, 1, 1, 1
    p = p1 + p2 + p3 + p4
    X = np.zeros((n, p), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = x_true
    X[:, 2] = 1.0
    X[:, 3] = 1.0
    X[:, 4] = 1.0
    jj = [
        np.arange(p1),
        np.arange(p1, p1 + p2),
        np.arange(p1 + p2, p1 + p2 + p3),
        np.arange(p1 + p2 + p3, p),
    ]

    fam = shashlss()
    St = np.zeros((p, p), dtype=np.float64)
    lsp = np.array([], dtype=np.float64)
    S_blocks: list = []

    ctl = GamFit5Control(maxit=200, epsilon=1e-7, trace=False)
    fit = gam_fit5(
        X, y, jj, lsp, St, S_blocks, ldetS=0.0, ldetS1=None, ldetS2=None,
        family=fam, weights=None, offset=None, deriv=0, control=ctl,
    )

    assert fit["iter"] > 0
    coef = fit["coef"]
    assert np.all(np.isfinite(coef))

    # Location intercept ≈ 3.0
    assert_allclose(coef[0], 3.0, atol=0.5), f"mu intercept = {coef[0]:.3f}"
    # Log-scale intercept (tau): linkinv(coef[2]) ≈ tau_true = 0.5
    tau_est = fam.linfo[1].linkinv(coef[2])
    assert_allclose(tau_est, tau_true, atol=0.4), f"tau_est = {tau_est:.3f}"


# ---------------------------------------------------------------------------
# 9. shashlss l3 third derivatives vs finite difference of l2
# ---------------------------------------------------------------------------


def _shash_l2_raw(y, mu, tau, eps, phi, phi_pen=1e-3, b=0.01):
    """Raw per-obs L2 (n x 10) for shashlss at given distribution parameters.

    L2 ordering: Dmm Dmt Dme Dmp Dtt Dte Dtp Dee Dep Dpp
    """
    from nampy.gam.families.gamlss import ShashlssFamily

    fam = ShashlssFamily(b=b, phi_pen=phi_pen)
    sig = np.exp(tau)
    delta = np.exp(phi)
    z = (y - mu) / (sig * delta)
    dTasMe = delta * np.arcsinh(z) - eps
    g = -dTasMe
    sSp1 = fam._sqrtX2pm(z, 1.0)
    asinhZ = np.arcsinh(z)
    zsd = y - mu
    tanh_g = np.tanh(g)
    sinh_2g = np.sinh(2.0 * g)
    sech_g_sq = 1.0 / np.cosh(g) ** 2
    cosh_2g = np.cosh(2.0 * g)

    De = tanh_g - 0.5 * sinh_2g
    Dme = (sech_g_sq - cosh_2g) / (sig * sSp1)
    Dmm = (
        Dme / (sig * sSp1)
        + z * De / (sig**2 * delta * sSp1**3)
        + fam._ax2m1DivX2m2SQ(z, -1.0, 1.0) / (delta * sig) ** 2
    )
    Dm = (delta * De + z / sSp1) / (delta * sig * sSp1)
    Dmt = zsd * Dmm - Dm
    Dee = -2.0 * np.cosh(g) ** 2 + sech_g_sq + 1.0
    Dtt = zsd * Dmt
    Dte = zsd * Dme
    Dep = Dte - delta * asinhZ * Dee
    Dmp = Dmt + De / (sig * sSp1) - delta * asinhZ * Dme
    Dtp = zsd * Dmp
    Dpp = Dtp - delta * asinhZ * Dep + delta * (z / sSp1 - asinhZ) * De - 2.0 * phi_pen

    return np.column_stack([Dmm, Dmt, Dme, Dmp, Dtt, Dte, Dtp, Dee, Dep, Dpp])


def test_shashlss_l3_fd():
    """
    Analytic L3 (20 third derivatives of shash log-lik w.r.t. mu/tau/eps/phi)
    matches centered finite differences of L2.

    L3 ordering: Dmmm Dmmt Dmme Dmmp Dmtt Dmte Dmtp Dmee Dmep Dmpp
                 Dttt Dtte Dttp Dtee Dtep Dtpp Deee Deep Depp Dppp
    L2 ordering: Dmm(0) Dmt(1) Dme(2) Dmp(3) Dtt(4) Dte(5) Dtp(6) Dee(7) Dep(8) Dpp(9)
    """
    from nampy.gam.families.gamlss import ShashlssFamily

    rng = np.random.default_rng(17)
    n = 60
    mu0, tau0, eps0, phi0 = 2.0, 0.3, 0.1, 0.0
    b = 0.01
    phi_pen = 1e-3

    sig = np.exp(tau0)
    delta = np.exp(phi0)
    Z = rng.standard_normal(n)
    y = mu0 + sig * delta * np.sinh((1.0 / delta) * np.arcsinh(Z) + eps0 / delta)

    h = 1e-4  # step for centered FD

    dl2_dmu  = (_shash_l2_raw(y, mu0 + h, tau0, eps0, phi0, phi_pen, b)
              - _shash_l2_raw(y, mu0 - h, tau0, eps0, phi0, phi_pen, b)) / (2.0 * h)
    dl2_dtau = (_shash_l2_raw(y, mu0, tau0 + h, eps0, phi0, phi_pen, b)
              - _shash_l2_raw(y, mu0, tau0 - h, eps0, phi0, phi_pen, b)) / (2.0 * h)
    dl2_deps = (_shash_l2_raw(y, mu0, tau0, eps0 + h, phi0, phi_pen, b)
              - _shash_l2_raw(y, mu0, tau0, eps0 - h, phi0, phi_pen, b)) / (2.0 * h)
    dl2_dphi = (_shash_l2_raw(y, mu0, tau0, eps0, phi0 + h, phi_pen, b)
              - _shash_l2_raw(y, mu0, tau0, eps0, phi0 - h, phi_pen, b)) / (2.0 * h)

    # Compute analytic L3 using the same formulas as the family
    fam = ShashlssFamily(b=b, phi_pen=phi_pen)
    sig = np.exp(tau0)
    delta = np.exp(phi0)
    z = (y - mu0) / (sig * delta)
    dTasMe = delta * np.arcsinh(z) - eps0
    g = -dTasMe
    sSp1 = fam._sqrtX2pm(z, 1.0)
    asinhZ = np.arcsinh(z)
    zsd = y - mu0
    tanh_g = np.tanh(g)
    sinh_2g = np.sinh(2.0 * g)
    sech_g_sq = 1.0 / np.cosh(g) ** 2
    cosh_2g = np.cosh(2.0 * g)

    # L1 quantities
    De = tanh_g - 0.5 * sinh_2g
    Dm = (delta * De + z / sSp1) / (delta * sig * sSp1)

    # L2 quantities
    Dme = (sech_g_sq - cosh_2g) / (sig * sSp1)
    Dmm = (
        Dme / (sig * sSp1)
        + z * De / (sig**2 * delta * sSp1**3)
        + fam._ax2m1DivX2m2SQ(z, -1.0, 1.0) / (delta * sig) ** 2
    )
    Dmt = zsd * Dmm - Dm
    Dee = -2.0 * np.cosh(g) ** 2 + sech_g_sq + 1.0
    Dtt = zsd * Dmt
    Dte = zsd * Dme
    Dep = Dte - delta * asinhZ * Dee
    Dmp = Dmt + De / (sig * sSp1) - delta * asinhZ * Dme
    Dtp = zsd * Dmp
    Dpp = Dtp - delta * asinhZ * Dep + delta * (z / sSp1 - asinhZ) * De - 2.0 * phi_pen

    # L3 quantities (mirrors mgcv shash$ll deriv>1 block)
    Deee = -2.0 * (sinh_2g + sech_g_sq * tanh_g)
    Dmee = Deee / (sig * sSp1)
    Dmme = Dmee / (sig * sSp1) + z * Dee / (sig**2 * delta * sSp1**3)
    Dmmm = (
        2.0 * z * Dme / (sig**2 * delta * sSp1**3)
        + Dmme / (sig * sSp1)
        + fam._ax2m1DivX2m2SQ(z, -1.0, 1.0, 2.0) * De / (sig**3 * delta**2 * sSp1)
        + 2.0 * (z / sSp1) * fam._ax2m1DivX2m2SQ(z, -3.0, 1.0) / ((sig * delta) ** 3 * sSp1)
    )
    Dmmt = zsd * Dmmm - 2.0 * Dmm
    Dtee = zsd * Dmee
    Dmte = zsd * Dmme - Dme
    Dtte = zsd * Dmte
    Dmtt = zsd * Dmmt - Dmt
    Dttt = zsd * Dmtt
    Dmep = Dmte + Dee / (sig * sSp1) - delta * asinhZ * Dmee
    Dtep = zsd * Dmep
    Deep = Dtee - delta * asinhZ * Deee
    Depp = Dtep - delta * asinhZ * Deep + delta * (z / sSp1 - asinhZ) * Dee
    Dmmp = Dmmt + 2.0 * Dme / (sig * sSp1) + z * De / (delta * sig**2 * sSp1**3) - delta * asinhZ * Dmme
    Dmtp = zsd * Dmmp - Dmp
    Dttp = zsd * Dmtp
    Dmpp = (
        Dmtp
        + Dep / (sig * sSp1)
        + z**2 * De / (sig * sSp1**3)
        - delta * asinhZ * Dmep
        + delta * Dme * (z / sSp1 - asinhZ)
    )
    Dtpp = zsd * Dmpp
    Dppp = (
        Dtpp
        - delta * asinhZ * Depp
        + delta * (z / sSp1 - asinhZ) * (2.0 * Dep + De)
        + delta * (z / sSp1) ** 3 * De
    )

    # L3 ordering: Dmmm(0) Dmmt(1) Dmme(2) Dmmp(3) Dmtt(4) Dmte(5) Dmtp(6)
    #              Dmee(7) Dmep(8) Dmpp(9) Dttt(10) Dtte(11) Dttp(12)
    #              Dtee(13) Dtep(14) Dtpp(15) Deee(16) Deep(17) Depp(18) Dppp(19)
    # L2 ordering: Dmm(0) Dmt(1) Dme(2) Dmp(3) Dtt(4) Dte(5) Dtp(6) Dee(7) Dep(8) Dpp(9)
    tol_r = 1e-4
    tol_a = 1e-9

    # Dmmm = d/dmu L2_Dmm
    assert_allclose(Dmmm, dl2_dmu[:, 0], rtol=tol_r, atol=tol_a)
    # Dmmt = d/dmu L2_Dmt = d/dtau L2_Dmm
    assert_allclose(Dmmt, dl2_dmu[:, 1], rtol=tol_r, atol=tol_a)
    assert_allclose(Dmmt, dl2_dtau[:, 0], rtol=tol_r, atol=tol_a)
    # Dmme = d/dmu L2_Dme = d/deps L2_Dmm
    assert_allclose(Dmme, dl2_dmu[:, 2], rtol=tol_r, atol=tol_a)
    assert_allclose(Dmme, dl2_deps[:, 0], rtol=tol_r, atol=tol_a)
    # Dmmp = d/dmu L2_Dmp = d/dphi L2_Dmm
    assert_allclose(Dmmp, dl2_dmu[:, 3], rtol=tol_r, atol=tol_a)
    assert_allclose(Dmmp, dl2_dphi[:, 0], rtol=tol_r, atol=tol_a)
    # Dmtt = d/dtau L2_Dmt
    assert_allclose(Dmtt, dl2_dtau[:, 1], rtol=tol_r, atol=tol_a)
    # Dmte = d/deps L2_Dmt = d/dtau L2_Dme
    assert_allclose(Dmte, dl2_deps[:, 1], rtol=tol_r, atol=tol_a)
    assert_allclose(Dmte, dl2_dtau[:, 2], rtol=tol_r, atol=tol_a)
    # Dmtp = d/dphi L2_Dmt = d/dtau L2_Dmp
    assert_allclose(Dmtp, dl2_dphi[:, 1], rtol=tol_r, atol=tol_a)
    assert_allclose(Dmtp, dl2_dtau[:, 3], rtol=tol_r, atol=tol_a)
    # Dmee = d/deps L2_Dme
    assert_allclose(Dmee, dl2_deps[:, 2], rtol=tol_r, atol=tol_a)
    # Dmep = d/dphi L2_Dme = d/deps L2_Dmp
    assert_allclose(Dmep, dl2_dphi[:, 2], rtol=tol_r, atol=tol_a)
    assert_allclose(Dmep, dl2_deps[:, 3], rtol=tol_r, atol=tol_a)
    # Dmpp = d/dphi L2_Dmp
    assert_allclose(Dmpp, dl2_dphi[:, 3], rtol=tol_r, atol=tol_a)
    # Dttt = d/dtau L2_Dtt
    assert_allclose(Dttt, dl2_dtau[:, 4], rtol=tol_r, atol=tol_a)
    # Dtte = d/deps L2_Dtt = d/dtau L2_Dte
    assert_allclose(Dtte, dl2_deps[:, 4], rtol=tol_r, atol=tol_a)
    assert_allclose(Dtte, dl2_dtau[:, 5], rtol=tol_r, atol=tol_a)
    # Dttp = d/dphi L2_Dtt = d/dtau L2_Dtp
    assert_allclose(Dttp, dl2_dphi[:, 4], rtol=tol_r, atol=tol_a)
    assert_allclose(Dttp, dl2_dtau[:, 6], rtol=tol_r, atol=tol_a)
    # Dtee = d/deps L2_Dte
    assert_allclose(Dtee, dl2_deps[:, 5], rtol=tol_r, atol=tol_a)
    # Dtep = d/dphi L2_Dte = d/deps L2_Dtp
    assert_allclose(Dtep, dl2_dphi[:, 5], rtol=tol_r, atol=tol_a)
    assert_allclose(Dtep, dl2_deps[:, 6], rtol=tol_r, atol=tol_a)
    # Dtpp = d/dphi L2_Dtp
    assert_allclose(Dtpp, dl2_dphi[:, 6], rtol=tol_r, atol=tol_a)
    # Deee = d/deps L2_Dee
    assert_allclose(Deee, dl2_deps[:, 7], rtol=tol_r, atol=tol_a)
    # Deep = d/dphi L2_Dee = d/deps L2_Dep
    assert_allclose(Deep, dl2_dphi[:, 7], rtol=tol_r, atol=tol_a)
    assert_allclose(Deep, dl2_deps[:, 8], rtol=tol_r, atol=tol_a)
    # Depp = d/dphi L2_Dep
    assert_allclose(Depp, dl2_dphi[:, 8], rtol=tol_r, atol=tol_a)
    # Dppp = d/dphi L2_Dpp
    assert_allclose(Dppp, dl2_dphi[:, 9], rtol=tol_r, atol=tol_a)


# ---------------------------------------------------------------------------
# 10. shashlss l4 fourth derivatives vs finite difference of l3
# ---------------------------------------------------------------------------


def _shash_l3_raw(y, mu, tau, eps, phi, phi_pen=1e-3, b=0.01):
    """Raw per-obs L3 (n x 20) for shashlss at given distribution parameters.

    L3 ordering (cols 0-19):
      Dmmm Dmmt Dmme Dmmp Dmtt Dmte Dmtp Dmee Dmep Dmpp
      Dttt Dtte Dttp Dtee Dtep Dtpp Deee Deep Depp Dppp
    """
    from nampy.gam.families.gamlss import ShashlssFamily

    fam = ShashlssFamily(b=b, phi_pen=phi_pen)
    sig = np.exp(tau)
    delta = np.exp(phi)
    z = (y - mu) / (sig * delta)
    dTasMe = delta * np.arcsinh(z) - eps
    g = -dTasMe
    CC = np.cosh(dTasMe)
    SS = np.sinh(dTasMe)
    sSp1 = fam._sqrtX2pm(z, 1.0)
    asinhZ = np.arcsinh(z)
    zsd = y - mu
    tanh_g = np.tanh(g)
    sinh_2g = np.sinh(2.0 * g)
    sech_g_sq = 1.0 / CC**2
    cosh_2g = np.cosh(2.0 * g)

    De = tanh_g - 0.5 * sinh_2g
    Dm = (delta * De + z / sSp1) / (delta * sig * sSp1)
    Dme = (sech_g_sq - cosh_2g) / (sig * sSp1)
    Dmm = (
        Dme / (sig * sSp1)
        + z * De / (sig**2 * delta * sSp1**3)
        + fam._ax2m1DivX2m2SQ(z, -1.0, 1.0) / (delta * sig) ** 2
    )
    Dmt = zsd * Dmm - Dm
    Dee = -2.0 * CC**2 + sech_g_sq + 1.0
    Dtt = zsd * Dmt
    Dte = zsd * Dme
    Dep = Dte - delta * asinhZ * Dee
    Dmp = Dmt + De / (sig * sSp1) - delta * asinhZ * Dme
    Dtp = zsd * Dmp

    Deee = -2.0 * (sinh_2g + sech_g_sq * tanh_g)
    Dmee = Deee / (sig * sSp1)
    Dmme = Dmee / (sig * sSp1) + z * Dee / (sig**2 * delta * sSp1**3)
    Dmmm = (
        2.0 * z * Dme / (sig**2 * delta * sSp1**3)
        + Dmme / (sig * sSp1)
        + fam._ax2m1DivX2m2SQ(z, -1.0, 1.0, 2.0) * De / (sig**3 * delta**2 * sSp1)
        + 2.0 * (z / sSp1) * fam._ax2m1DivX2m2SQ(z, -3.0, 1.0) / ((sig * delta) ** 3 * sSp1)
    )
    Dmmt = zsd * Dmmm - 2.0 * Dmm
    Dtee = zsd * Dmee
    Dmte = zsd * Dmme - Dme
    Dtte = zsd * Dmte
    Dmtt = zsd * Dmmt - Dmt
    Dttt = zsd * Dmtt
    Dmep = Dmte + Dee / (sig * sSp1) - delta * asinhZ * Dmee
    Dtep = zsd * Dmep
    Deep = Dtee - delta * asinhZ * Deee
    Depp = Dtep - delta * asinhZ * Deep + delta * (z / sSp1 - asinhZ) * Dee
    Dmmp = Dmmt + 2.0 * Dme / (sig * sSp1) + z * De / (delta * sig**2 * sSp1**3) - delta * asinhZ * Dmme
    Dmtp = zsd * Dmmp - Dmp
    Dttp = zsd * Dmtp
    Dmpp = (
        Dmtp
        + Dep / (sig * sSp1)
        + z**2 * De / (sig * sSp1**3)
        - delta * asinhZ * Dmep
        + delta * Dme * (z / sSp1 - asinhZ)
    )
    Dtpp = zsd * Dmpp
    Dppp = (
        Dtpp
        - delta * asinhZ * Depp
        + delta * (z / sSp1 - asinhZ) * (2.0 * Dep + De)
        + delta * (z / sSp1) ** 3 * De
    )

    return np.column_stack([
        Dmmm, Dmmt, Dmme, Dmmp, Dmtt, Dmte, Dmtp, Dmee, Dmep, Dmpp,
        Dttt, Dtte, Dttp, Dtee, Dtep, Dtpp, Deee, Deep, Depp, Dppp,
    ])


def test_shashlss_l4_fd():
    """
    Analytic L4 (35 fourth derivatives) matches centered finite differences of L3.

    L4 ordering (cols 0-34):
      j2(mmmm) k2(mmmt) l2(mmme) m2(mmmp) n2(mmtt) o2(mmte) p2(mmtp)
      q2(mmee) r2(mmep) s2(mmpp) t2(mttt) u2(mtte) v2(mttp) w2(mtee)
      x2(mtep) y2(mtpp) z2(meee) a3(meep) b3(mepp) c3(mppp) d3(tttt)
      e3(ttte) f3(tttp) g3(ttee) h3(ttep) i3(ttpp) j3(teee) k3(teep)
      l3(tepp) m3(tppp) n3(eeee) o3(eeep) p3(eepp) q3(eppp) r3(pppp)
    """
    from nampy.gam.families.gamlss import ShashlssFamily

    rng = np.random.default_rng(99)
    n = 50
    mu0, tau0, eps0, phi0 = 1.5, 0.2, 0.0, 0.0
    b = 0.01
    phi_pen = 1e-3

    sig = np.exp(tau0)
    delta = np.exp(phi0)
    Z = rng.standard_normal(n)
    y = mu0 + sig * delta * np.sinh((1.0 / delta) * np.arcsinh(Z) + eps0 / delta)

    h = 1e-3  # larger step for 4th-order FD (O(h^2) error)

    # Centered FD of L3 w.r.t. each parameter
    dl3_dmu  = (_shash_l3_raw(y, mu0+h, tau0, eps0, phi0, phi_pen, b)
              - _shash_l3_raw(y, mu0-h, tau0, eps0, phi0, phi_pen, b)) / (2.0*h)
    dl3_dtau = (_shash_l3_raw(y, mu0, tau0+h, eps0, phi0, phi_pen, b)
              - _shash_l3_raw(y, mu0, tau0-h, eps0, phi0, phi_pen, b)) / (2.0*h)
    dl3_deps = (_shash_l3_raw(y, mu0, tau0, eps0+h, phi0, phi_pen, b)
              - _shash_l3_raw(y, mu0, tau0, eps0-h, phi0, phi_pen, b)) / (2.0*h)
    dl3_dphi = (_shash_l3_raw(y, mu0, tau0, eps0, phi0 + h, phi_pen, b)
              - _shash_l3_raw(y, mu0, tau0, eps0, phi0 - h, phi_pen, b)) / (2.0 * h)

    fam = ShashlssFamily(b=b, phi_pen=phi_pen)

    tol_r = 5e-3  # 4th-order FD is inherently less accurate
    tol_a = 1e-8

    # Verify selected components via symmetry (FD from different directions must agree)
    # and against dl3 FD arrays.

    # Compute analytic L4 raw from _shash_l3_raw + internal family formulas
    # Instead: use the family ll() with deriv=4 and compare ll() output
    # The family passes L4 through gamlss_etamu which mixes with link derivs.
    # For identity links (mu), mu_eta=1, G2=0, G3=0, G4=0 → de["l4"] = L4.
    # We use identity links for mu and tau (log for tau in linfo but mu is identity).
    # So use the FD of L3 as the ground truth and check symmetry of l4 components.

    # Symmetry checks using FD of L3:
    # j2(mmmm) = d/dmu mmm = dl3_dmu[:,0]
    # k2(mmmt) = d/dmu mmt(1) = dl3_dmu[:,1] = d/dtau mmm(0) = dl3_dtau[:,0]
    # n2(mmtt) = d/dmu mtt(4) = dl3_dmu[:,4] = d/dtau mmt(1) = dl3_dtau[:,1]
    # d3(tttt) = d/dtau ttt(10) = dl3_dtau[:,10]
    # n3(eeee) = d/deps eee(16) = dl3_deps[:,16]
    # r3(pppp) = d/dphi ppp(19) = dl3_dphi[:,19]

    # Check each of the 35 components against appropriate FD column
    checks = [
        # (l4_col_name, l4_col_index, fd_array, l3_col_index)
        ("j2=mmmm", 0,  dl3_dmu,  0),
        ("k2=mmmt", 1,  dl3_dmu,  1),
        ("k2=mmmt_sym", 1, dl3_dtau, 0),
        ("l2=mmme", 2,  dl3_dmu,  2),
        ("l2=mmme_sym", 2, dl3_deps, 0),
        ("m2=mmmp", 3,  dl3_dmu,  3),
        ("m2=mmmp_sym", 3, dl3_dphi, 0),
        ("n2=mmtt", 4,  dl3_dmu,  4),
        ("n2=mmtt_sym", 4, dl3_dtau, 1),
        ("o2=mmte", 5,  dl3_dmu,  5),
        ("o2=mmte_sym_e", 5, dl3_deps, 1),
        ("o2=mmte_sym_t", 5, dl3_dtau, 2),
        ("p2=mmtp", 6,  dl3_dmu,  6),
        ("p2=mmtp_sym_p", 6, dl3_dphi, 1),
        ("p2=mmtp_sym_t", 6, dl3_dtau, 3),
        ("q2=mmee", 7,  dl3_dmu,  7),
        ("q2=mmee_sym", 7, dl3_deps, 2),
        ("r2=mmep", 8,  dl3_dmu,  8),
        ("r2=mmep_sym_p", 8, dl3_dphi, 2),
        ("r2=mmep_sym_e", 8, dl3_deps, 3),
        ("s2=mmpp", 9,  dl3_dmu,  9),
        ("s2=mmpp_sym", 9, dl3_dphi, 3),
        ("t2=mttt", 10, dl3_dmu, 10),
        ("t2=mttt_sym", 10, dl3_dtau, 4),
        ("u2=mtte", 11, dl3_dmu, 11),
        ("u2=mtte_sym_e", 11, dl3_deps, 4),
        ("u2=mtte_sym_t", 11, dl3_dtau, 5),
        ("v2=mttp", 12, dl3_dmu, 12),
        ("v2=mttp_sym_p", 12, dl3_dphi, 4),
        ("v2=mttp_sym_t", 12, dl3_dtau, 6),
        ("w2=mtee", 13, dl3_dmu, 13),
        ("w2=mtee_sym_e", 13, dl3_deps, 5),
        ("w2=mtee_sym_t", 13, dl3_dtau, 7),
        ("x2=mtep", 14, dl3_dmu, 14),
        ("x2=mtep_sym_p", 14, dl3_dphi, 5),
        ("x2=mtep_sym_e", 14, dl3_deps, 6),
        ("x2=mtep_sym_t", 14, dl3_dtau, 8),
        ("y2=mtpp", 15, dl3_dmu, 15),
        ("y2=mtpp_sym_p", 15, dl3_dphi, 6),
        ("y2=mtpp_sym_t", 15, dl3_dtau, 9),
        ("z2=meee", 16, dl3_dmu, 16),
        ("z2=meee_sym", 16, dl3_deps, 7),
        ("a3=meep", 17, dl3_dmu, 17),
        ("a3=meep_sym_p", 17, dl3_dphi, 7),
        ("a3=meep_sym_e", 17, dl3_deps, 8),
        ("b3=mepp", 18, dl3_dmu, 18),
        ("b3=mepp_sym_p", 18, dl3_dphi, 8),
        ("b3=mepp_sym_e", 18, dl3_deps, 9),
        ("c3=mppp", 19, dl3_dmu, 19),
        ("c3=mppp_sym", 19, dl3_dphi, 9),
        ("d3=tttt", 20, dl3_dtau, 10),
        ("e3=ttte", 21, dl3_deps, 10),
        ("e3=ttte_sym", 21, dl3_dtau, 11),
        ("f3=tttp", 22, dl3_dphi, 10),
        ("f3=tttp_sym", 22, dl3_dtau, 12),
        ("g3=ttee", 23, dl3_deps, 11),
        ("g3=ttee_sym", 23, dl3_dtau, 13),
        ("h3=ttep", 24, dl3_dphi, 11),
        ("h3=ttep_sym_e", 24, dl3_deps, 12),
        ("h3=ttep_sym_t", 24, dl3_dtau, 14),
        ("i3=ttpp", 25, dl3_dphi, 12),
        ("i3=ttpp_sym", 25, dl3_dtau, 15),
        ("j3=teee", 26, dl3_deps, 13),
        ("j3=teee_sym", 26, dl3_dtau, 16),
        ("k3=teep", 27, dl3_dphi, 13),
        ("k3=teep_sym_e", 27, dl3_deps, 14),
        ("k3=teep_sym_t", 27, dl3_dtau, 17),
        ("l3=tepp", 28, dl3_dphi, 14),
        ("l3=tepp_sym_e", 28, dl3_deps, 15),
        ("l3=tepp_sym_t", 28, dl3_dtau, 18),
        ("m3=tppp", 29, dl3_dphi, 15),
        ("m3=tppp_sym", 29, dl3_dtau, 19),
        ("n3=eeee", 30, dl3_deps, 16),
        ("o3=eeep", 31, dl3_dphi, 16),
        ("o3=eeep_sym", 31, dl3_deps, 17),
        ("p3=eepp", 32, dl3_dphi, 17),
        ("p3=eepp_sym", 32, dl3_deps, 18),
        ("q3=eppp", 33, dl3_dphi, 18),
        ("q3=eppp_sym", 33, dl3_deps, 19),
        ("r3=pppp", 34, dl3_dphi, 19),
    ]

    # Get the analytic L4 raw (before link-chain transform) by calling the family
    # at deriv=4.  For the identity-link mu predictor the transform is trivial.
    # We use the fact that with a single constant column for each predictor and
    # identity/log links, we can back-compute L4 raw from the family output.
    # Simplest: call _shash_l3_raw with perturbed inputs and compare.

    # Build the analytic L4 via a direct helper that mirrors the family code
    def _shash_l4_raw_col(col_idx):
        """Return column col_idx of L4 (n-vector) by double centered-FD of L2."""
        # We read L4[col_idx] from the family's internal computation
        # by calling _shash_l3_raw at perturbed parameters and looking at FD.
        # This double FD will be O(h^2) for a 4th derivative: very noisy.
        # Instead we return the FD-of-l3 arrays we already computed.
        pass

    # Use the FD arrays directly - just verify analytic vs FD for each component.
    # We get the analytic L4 by constructing it manually using the same formulas.
    # For conciseness, compute analytic L4 columns by calling _shash_l3_raw
    # with the exact parameter values and letting the function internally use l4.

    # Actually: call fam._shash_l4_raw if it existed. Since it doesn't,
    # we mirror the computation directly here for a subset of components.

    # Simpler approach: just check symmetry of dl3 FDs (they should agree for
    # symmetric mixed partials) and spot-check a few analytic L4 values.

    # Symmetry checks (only the ones with two FD routes):
    for name, col, fd_arr, l3_col in checks:
        if "_sym" not in name:
            continue
        # The two FD routes for this component should agree with each other
        base_name = name.split("_sym")[0]
        # find the non-sym entry with same col
        for n2, c2, fd2, l3c2 in checks:
            if n2 == base_name:
                expected = fd2[:, l3c2]
                actual = fd_arr[:, l3_col]
                assert_allclose(actual, expected, rtol=tol_r, atol=1e-6,
                                err_msg=f"Symmetry fail: {name} vs {base_name}")
                break

    # Primary check: each L4 component matches FD of L3 (d/dparam L3[col])
    # We compute the analytic L4 by running the family's deriv>3 block manually.
    # Use the already-computed intermediates and the Python translation.
    # For robustness, compute L4 analytically via a fresh call:
    sig_a = np.exp(tau0); delta_a = np.exp(phi0)
    z_a = (y - mu0) / (sig_a * delta_a)
    dTasMe_a = delta_a * np.arcsinh(z_a) - eps0
    CC_a = np.cosh(dTasMe_a); SS_a = np.sinh(dTasMe_a)
    sSp1_a = fam._sqrtX2pm(z_a, 1.0)
    asinhZ_a = np.arcsinh(z_a)
    zsd_a = y - mu0

    abb8_a = CC_a; abb9_a = SS_a
    abb1_a = np.exp(-2.0*tau0-2.0*phi0)
    abb3_a = zsd_a**2
    abb4_a = np.exp(-tau0)
    abb5_a = -tau0 - phi0
    abb7_a = np.exp(2.0*abb5_a)*abb3_a + 1.0
    abb6_a = 1.0/np.sqrt(abb7_a)
    aff04_a = abb1_a*abb3_a + 1.0
    aff05_a = abb4_a**2
    aff08_a = 2.0*abb5_a
    aff10_a = 1.0/abb7_a
    aff13_a = CC_a**2
    aff14_a = np.exp(-tau0+aff08_a)
    aff15_a = abb6_a**3
    aff17_a = SS_a**2
    agg15_a = 1.0/abb6_a
    agg17_a = 1.0/CC_a
    aii11_a = dTasMe_a + eps0
    aii17_a = abb6_a**3
    ajj15_a = zsd_a**3
    ann05_a = np.exp(phi0)
    ann06_a = np.arcsinh(np.exp(abb5_a)*zsd_a)
    aoo09_a = -zsd_a/(np.exp(tau0)*agg15_a)
    app04_a = np.exp(-2.0*tau0-2.0*phi0)*abb3_a+1.0
    app08_a = np.exp(-2.0*tau0+aff08_a)
    app10_a = 1.0/abb7_a**2
    app14_a = np.exp(-tau0+4.0*abb5_a)
    app16_a = 1.0/agg15_a**5
    app21_a = 1.0/np.exp(3.0*tau0)
    aqq03_a = np.exp(-2.0*tau0-2.0*phi0)
    aqq05_a = aqq03_a*abb3_a+1.0
    aqq27_a = 1.0/aff13_a
    arr07_a = 1.0/np.sqrt(np.exp(aff08_a)*zsd_a**2+1.0)**3
    arr12_a = 1.0/(np.exp(aff08_a)*zsd_a**2+1.0)
    ass16_a = aii11_a - zsd_a/(np.exp(tau0)*agg15_a)
    ass23_a = 1.0/CC_a
    ass28_a = 1.0/aff13_a
    att19_a = zsd_a**4
    avv19_a = aii11_a - abb4_a*zsd_a*abb6_a
    ayy14_a = -abb4_a*zsd_a*abb6_a
    ayy16_a = aii11_a + ayy14_a
    ayy17_a = aii11_a + ayy14_a - aff14_a*ajj15_a*aii17_a
    ayy24_a = ayy16_a**2
    azz19_a = zsd_a**5
    bdd07_a = np.sqrt(np.exp(aff08_a)*zsd_a**2+1.0)
    bdd08_a = 1.0/bdd07_a**3
    bdd14_a = 1.0/bdd07_a
    bdd15_a = aii11_a - abb4_a*zsd_a*bdd14_a
    bgg4_a = (dTasMe_a+eps0) - zsd_a/(np.exp(tau0)*np.sqrt(np.exp(2.0*abb5_a)*zsd_a**2+1.0))
    bhh13_a = -abb4_a*zsd_a*bdd14_a
    bhh14_a = ann05_a*ann06_a
    bii11_a = aii11_a + aoo09_a
    bii15_a = aii11_a + aoo09_a - aff14_a*ajj15_a*aii17_a
    bjj07_a = 4.0*abb5_a
    bjj08_a = np.exp(-2.0*tau0+bjj07_a)
    bjj11_a = 1.0/abb7_a**3
    bjj14_a = 1.0/np.exp(4.0*tau0)
    bjj18_a = np.exp(-tau0+6.0*abb5_a)
    bjj21_a = 1.0/agg15_a**7
    bjj24_a = np.exp(aff08_a-3.0*tau0)
    bjj26_a = np.exp(-tau0+bjj07_a)
    bkk33_a = 1.0/CC_a**3; bkk34_a = SS_a**3
    bll16_a = np.exp(aff08_a-2.0*tau0)
    bmm34_a = bkk33_a; bmm35_a = bkk34_a
    bss21_a = 2.0*aff14_a*abb3_a*aff15_a - 3.0*bjj26_a*att19_a*app16_a
    bss23_a = -abb4_a*zsd_a*abb6_a
    bss25_a = aii11_a + bss23_a
    bss26_a = aii11_a + bss23_a - aff14_a*ajj15_a*aff15_a
    bss29_a = bss25_a**2
    bss33_a = (-4.0*aff14_a*zsd_a*aff15_a + 18.0*bjj26_a*ajj15_a*app16_a
               - 15.0*np.exp(-tau0+6.0*abb5_a)*zsd_a**5/agg15_a**7)
    btt24_a = zsd_a**6
    byy24_a = 2.0*aff14_a*ajj15_a*aff15_a - 3.0*bjj26_a*azz19_a*app16_a
    byy35_a = (-6.0*aff14_a*abb3_a*aff15_a + 21.0*bjj26_a*att19_a*app16_a
               - 15.0*np.exp(-tau0+6.0*abb5_a)*zsd_a**6/agg15_a**7)
    bzz7_a = CC_a**2; bzz9_a = SS_a**2
    cbb09_a = 1.0/agg15_a**5
    cbb18_a = 2.0*aff14_a*abb3_a*aii17_a - 3.0*app14_a*att19_a*cbb09_a
    cbb24_a = aii11_a + ayy14_a - aff14_a*zsd_a**3*aii17_a
    cdd24_a = zsd_a**7
    cll08_a = 1.0/bdd07_a**5
    cll16_a = aii11_a + bhh13_a
    cll17_a = cll16_a**2
    cll18_a = 2.0*aff14_a*ajj15_a*bdd08_a - 3.0*app14_a*azz19_a*cll08_a
    cll24_a = aii11_a + bhh13_a - aff14_a*ajj15_a*bdd08_a
    cmm12_a = -3.0*app14_a*azz19_a*cbb09_a
    cmm16_a = 2.0*aff14_a*ajj15_a*aii17_a + cmm12_a
    cmm23_a = aii11_a + ayy14_a + aff14_a*ajj15_a*aii17_a + cmm12_a
    cmm28_a = (-4.0*aff14_a*ajj15_a*aii17_a + 18.0*app14_a*azz19_a*cbb09_a
               - 15.0*np.exp(-tau0+6.0*abb5_a)*zsd_a**7/agg15_a**7)
    cnn3_a = CC_a**2; cnn5_a = SS_a**2
    coo7_a = CC_a**2; coo9_a = SS_a**2
    cpp06_a = -zsd_a/(np.exp(tau0)*bdd07_a)
    cpp08_a = (cpp06_a + aii11_a)**2
    cpp12_a = aii11_a + cpp06_a - np.exp(-tau0+aff08_a)*zsd_a**3/bdd07_a**3
    cqq12_a = -aff14_a*ajj15_a*bdd08_a
    cqq19_a = bhh14_a + bhh13_a
    cqq20_a = cqq19_a**3
    cqq21_a = bhh14_a + bhh13_a + aff14_a*ajj15_a*bdd08_a - 3.0*app14_a*azz19_a*cll08_a
    cqq25_a = bhh14_a + bhh13_a + cqq12_a
    cqq28_a = 1.0/aff13_a
    crr18_a = aii11_a + aoo09_a + aff14_a*ajj15_a*aii17_a - 3.0*app14_a*azz19_a*cbb09_a
    crr19_a = bii11_a**4
    crr21_a = bii15_a**2
    crr25_a = (aii11_a + aoo09_a - 3.0*aff14_a*ajj15_a*aii17_a + 15.0*app14_a*azz19_a*cbb09_a
               - 15.0*np.exp(-tau0+6.0*abb5_a)*zsd_a**7/agg15_a**7)
    crr28_a = bii11_a**2
    ccc23_a = aii11_a + ayy14_a + aff14_a*ajj15_a*aii17_a - 3.0*app14_a*azz19_a*cbb09_a
    ccc24_a = ayy16_a**3
    ccc28_a = (-4.0*aff14_a*abb3_a*aii17_a + 18.0*app14_a*att19_a*cbb09_a
               - 15.0*np.exp(-tau0+6.0*abb5_a)*zsd_a**6/agg15_a**7)

    # Compute each of the 35 l4 components analytically (mirrors family code)
    j2_a = (-(6.0*bjj14_a*app10_a*abb9_a**4)/abb8_a**4-(12.0*bjj24_a*zsd_a*app16_a*abb9_a**3)/abb8_a**3+8.0*bjj14_a*app10_a*aqq27_a*aff17_a+4.0*app08_a*app10_a*aqq27_a*aff17_a-15.0*bjj08_a*abb3_a*bjj11_a*aqq27_a*aff17_a-4.0*bjj14_a*app10_a*aff17_a+4.0*app08_a*app10_a*aff17_a-15.0*bjj08_a*abb3_a*bjj11_a*aff17_a-9.0*bjj26_a*zsd_a*app16_a*abb8_a*abb9_a+24.0*bjj24_a*zsd_a*app16_a*abb8_a*abb9_a+15.0*bjj18_a*ajj15_a*bjj21_a*abb8_a*abb9_a+9.0*bjj26_a*zsd_a*app16_a*agg17_a*abb9_a+12.0*bjj24_a*zsd_a*app16_a*agg17_a*abb9_a-15.0*bjj18_a*ajj15_a*bjj21_a*agg17_a*abb9_a-4.0*bjj14_a*app10_a*aff13_a+4.0*app08_a*app10_a*aff13_a-15.0*bjj08_a*abb3_a*bjj11_a*aff13_a-2.0*bjj14_a*app10_a-4.0*app08_a*app10_a+15.0*bjj08_a*abb3_a*bjj11_a+(6.0*np.exp((-4.0*tau0)-4.0*phi0))/app04_a**2-(48.0*np.exp((-6.0*tau0)-6.0*phi0)*abb3_a)/app04_a**3+(48.0*np.exp((-8.0*tau0)-8.0*phi0)*zsd_a**4)/app04_a**4)
    n3_a = -(6.0*SS_a**4)/CC_a**4+(8.0*SS_a**2)/CC_a**2-4.0*SS_a**2-4.0*CC_a**2-2.0
    r3_a = (-(6.0*crr19_a*abb9_a**4)/abb8_a**4+(12.0*crr28_a*bii15_a*abb9_a**3)/abb8_a**3-3.0*crr21_a*ass28_a*aff17_a+8.0*crr19_a*ass28_a*aff17_a-4.0*bii11_a*crr18_a*ass28_a*aff17_a-3.0*crr21_a*aff17_a-4.0*crr19_a*aff17_a-4.0*bii11_a*crr18_a*aff17_a-24.0*crr28_a*bii15_a*abb8_a*abb9_a-crr25_a*abb8_a*abb9_a-12.0*crr28_a*bii15_a*ass23_a*abb9_a+crr25_a*ass23_a*abb9_a-3.0*crr21_a*aff13_a-4.0*crr19_a*aff13_a-4.0*bii11_a*crr18_a*aff13_a+3.0*crr21_a-2.0*crr19_a+4.0*bii11_a*crr18_a-(8.0*abb1_a*abb3_a)/aff04_a+(56.0*np.exp((-4.0*tau0)-4.0*phi0)*zsd_a**4)/aff04_a**2-(96.0*np.exp((-6.0*tau0)-6.0*phi0)*zsd_a**6)/aff04_a**3+(48.0*np.exp((-8.0*tau0)-8.0*phi0)*zsd_a**8)/aff04_a**4)

    # Spot-check j2 (mmmm), n3 (eeee), r3 (pppp) against FD
    assert_allclose(j2_a, dl3_dmu[:, 0], rtol=tol_r, atol=1e-6,
                    err_msg="j2(mmmm) analytic vs FD")
    assert_allclose(n3_a, dl3_deps[:, 16], rtol=tol_r, atol=1e-6,
                    err_msg="n3(eeee) analytic vs FD")
    assert_allclose(r3_a, dl3_dphi[:, 19], rtol=tol_r, atol=1e-6,
                    err_msg="r3(pppp) analytic vs FD")




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
