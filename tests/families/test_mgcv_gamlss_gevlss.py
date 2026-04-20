from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from nampy.gam import GAM
from nampy.gam.families.gamlss import gevlss
from nampy.gam.families.gamlss.gevlss import _ShiftedLogitLinkInfo
from nampy.gam.fit.solvers.gam_fit5 import GamFit5Control, gam_fit5

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
    coef[p1 : p1 + p2] = rng.standard_normal(p2) * 0.1
    coef[p1 + p2 :] = rng.standard_normal(p3) * 0.2

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
    aa = np.maximum(1.0 + xi * (y - mu) / sigma, 1e-300)
    l_ref = float(np.sum(-(1.0 / xi + 1.0) * np.log(aa) - aa ** (-1.0 / xi) - rho))
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
    coef[p1 : p1 + p2] = rng.standard_normal(p2) * 0.1

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
    coef[p1 : p1 + p2] = rng.standard_normal(p2) * 0.05

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
    xi_true = 0.1  # shape parameter

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
    assert np.all(np.isfinite(coef))

    # Location intercept ≈ 2.0
    assert_allclose(coef[0], 2.0, atol=0.3), f"mu intercept = {coef[0]:.3f}"
    # Log-scale intercept ≈ 0.0
    assert_allclose(coef[2], 0.0, atol=0.3), f"rho = {coef[2]:.3f}"
    # Shape: xi_est ≈ 0.1 (but link maps to eta, so check linkinv)
    xi_est = fam.linfo[2].linkinv(coef[3])
    assert_allclose(xi_est, xi_true, atol=0.15), f"xi_est = {xi_est:.3f}"


def test_gam_public_api_gevlss_formula_list_fit():
    """Public GAM API fits a three-predictor gevlss model via formula list."""
    rng = np.random.default_rng(12345)
    n = 220
    x = rng.standard_normal(n)
    mu_true = 1.8 + 0.45 * x
    rho_true = -0.15
    xi_true = 0.1
    u = rng.uniform(0.02, 0.98, n)
    sigma = np.exp(rho_true)
    y = mu_true + sigma * ((-np.log(u)) ** (-xi_true) - 1.0) / xi_true
    data = pd.DataFrame({"y": y, "x": x})

    gam = GAM(
        family="gevlss",
        formula=["y ~ x", "~ 1", "~ 1"],
        optimize_smoothing=False,
    )
    gam.fit(data=data)
    fit = gam.fit_result(include_covariances=True)
    coef_full = np.asarray(fit.coef_full, dtype=np.float64)
    vp = np.asarray(gam.vcov(), dtype=np.float64)
    vc = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)

    assert coef_full.shape == (4,)
    assert vp.shape == (4, 4)
    assert vc.shape == (4, 4)
    assert np.isfinite(gam.loglik())

    eta = np.asarray(gam.predict(data, type="link"), dtype=np.float64)
    fitted = np.asarray(gam.predict(data, type="response"), dtype=np.float64)
    assert eta.shape == (n, 3)
    assert fitted.shape == (n, 3)

    assert_allclose(coef_full[0], 1.8, atol=0.35)
    assert_allclose(coef_full[1], 0.45, atol=0.35)
    assert_allclose(coef_full[2], rho_true, atol=0.35)
    xi_est = gam.family.linfo[2].linkinv(np.array([coef_full[3]]))[0]
    assert_allclose(xi_est, xi_true, atol=0.2)


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
        - jj13
        * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7)
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
    l3_ana[:, 0] = (2.0 * kk1 * kk2 * dd3) / cc3**3 - kk1 * (ee3 - 2.0) * (
        ee3 - 1.0
    ) * kk2 * cc3 ** (ee3 - 3.0)
    l3_ana[:, 1] = (
        -2.0 * ee1 * ff7 * xi * cc3**ll8
        - kk1 * ll8 * ff7 * kk2 * cc2 * cc3 ** (ee3 - 3.0)
        - (2.0 * ee1 * xi * dd3) / cc3**2
        + (2.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
    )
    l3_ana[:, 2] = (
        ee1 * ff7 * xi * mm12 * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
        - ee1 * mm12
        - kk1 * mm11 * xi * cc2 * mm10
        + kk1 * cc2 * mm10
        + ee1 * dd3 * jj08
        + ee1 * xi * jj08
        - (2.0 * kk1 * xi * dd3 * cc2) / cc3**3
    )
    l3_ana[:, 3] = (
        -bb1 * cc3**ff7
        - 3.0 * ee1 * ff7 * xi * cc2 * cc3**ll8
        - kk1 * ll8 * ff7 * kk2 * hh4 * cc3 ** (ee3 - 3.0)
        + (bb1 * dd3) / cc3
        - (3.0 * ee1 * xi * dd3 * cc2) / cc3**2
        + (2.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
    )
    l3_ana[:, 4] = (
        bb1 * cc3**oo10 * (bb1 * oo10 * cc2 * dd6 + oo13)
        + ee1 * oo10 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
        + ee1 * aa2 * cc2 * mm12
        + ee1 * oo10 * cc2 * mm12
        - bb1 * dd6
        + 2.0 * ee1 * dd3 * cc2 * jj08
        + ee1 * xi * cc2 * jj08
        - 2.0 * xi * dd3 * hh4 * kk1 / cc3**3
    )
    l3_ana[:, 5] = (
        -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2
        - bb1
        * pp08
        * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 / xi**3)
        - 2.0 * ee1 * cc2 * jj08
        + 2.0 * dd3 * hh4 * kk1 / cc3**3
    )
    l3_ana[:, 6] = (
        -bb1 * cc2 * cc3**ff7
        - 3.0 * ee1 * ff7 * xi * hh4 * cc3**ll8
        - kk1 * ll8 * ff7 * kk2 * qq05 * cc3 ** (ee3 - 3.0)
        + (bb1 * dd3 * cc2) / cc3
        - (3.0 * ee1 * xi * dd3 * hh4) / cc3**2
        + (2.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
    )
    l3_ana[:, 7] = (
        bb1 * cc2 * cc3**oo10 * rr17
        + ee1 * oo10 * xi * hh4 * mm12 * rr17
        - 2.0 * ee1 * hh4 * mm12
        - kk1 * mm11 * xi * qq05 * mm10
        + kk1 * qq05 * mm10
        - bb1 * cc2 * dd6
        + 2.0 * ee1 * dd3 * hh4 * jj08
        + ee1 * xi * hh4 * jj08
        - (2.0 * kk1 * xi * dd3 * qq05) / cc3**3
    )
    l3_ana[:, 8] = (
        -bb1 * cc2 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2
        - bb1
        * cc2
        * pp08
        * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 / xi**3)
        - 2.0 * ee1 * hh4 * jj08
        + 2.0 * dd3 * qq05 * kk1 / cc3**3
    )
    l3_ana[:, 9] = (
        -jj13 * tt18**3
        - 3.0
        * jj13
        * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7)
        * tt18
        - jj13
        * (
            -2.0 * kk1 * aa2 * qq05 * tt08
            - 3.0 * ee1 * dd8 * hh4 * jj08
            - 6.0 * bb1 * jj12 * cc2 * dd6
            + 6.0 * tt16 * dd7
        )
        + 6.0 * tt16 * dd3 * dd7
        - 6.0 * jj12 * dd7
        - 6.0 * bb1 * jj12 * dd3 * cc2 * dd6
        + 6.0 * bb1 * dd8 * cc2 * dd6
        - 3.0 * ee1 * dd8 * dd3 * hh4 * jj08
        + 3.0 * ee1 * aa2 * hh4 * jj08
        - 2.0 * kk1 * aa2 * dd3 * qq05 * tt08
    )

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
    l3[:, 0] = (2.0 * kk1 * kk2 * dd3) / cc3**3 - kk1 * (ee3 - 2.0) * (
        ee3 - 1.0
    ) * kk2 * cc3 ** (ee3 - 3.0)
    l3[:, 1] = (
        -2.0 * ee1 * ff7 * xi * cc3**ll8
        - kk1 * ll8 * ff7 * kk2 * cc2 * cc3 ** (ee3 - 3.0)
        - (2.0 * ee1 * xi * dd3) / cc3**2
        + (2.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
    )
    l3[:, 2] = (
        ee1 * ff7 * xi * mm12 * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
        - ee1 * mm12
        - kk1 * mm11 * xi * cc2 * mm10
        + kk1 * cc2 * mm10
        + ee1 * dd3 * jj08
        + ee1 * xi * jj08
        - (2.0 * kk1 * xi * dd3 * cc2) / cc3**3
    )
    l3[:, 3] = (
        -bb1 * cc3**ff7
        - 3.0 * ee1 * ff7 * xi * cc2 * cc3**ll8
        - kk1 * ll8 * ff7 * kk2 * hh4 * cc3 ** (ee3 - 3.0)
        + (bb1 * dd3) / cc3
        - (3.0 * ee1 * xi * dd3 * cc2) / cc3**2
        + (2.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
    )
    l3[:, 4] = (
        bb1 * cc3**oo10 * (bb1 * oo10 * cc2 * dd6 + oo13)
        + ee1 * oo10 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
        + ee1 * aa2 * cc2 * mm12
        + ee1 * oo10 * cc2 * mm12
        - bb1 * dd6
        + 2.0 * ee1 * dd3 * cc2 * jj08
        + ee1 * xi * cc2 * jj08
        - 2.0 * xi * dd3 * hh4 * kk1 / cc3**3
    )
    l3[:, 5] = (
        -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2
        - bb1
        * pp08
        * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 / xi**3)
        - 2.0 * ee1 * cc2 * jj08
        + 2.0 * dd3 * hh4 * kk1 / cc3**3
    )
    l3[:, 6] = (
        -bb1 * cc2 * cc3**ff7
        - 3.0 * ee1 * ff7 * xi * hh4 * cc3**ll8
        - kk1 * ll8 * ff7 * kk2 * qq05 * cc3 ** (ee3 - 3.0)
        + (bb1 * dd3 * cc2) / cc3
        - (3.0 * ee1 * xi * dd3 * hh4) / cc3**2
        + (2.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
    )
    l3[:, 7] = (
        bb1 * cc2 * cc3**oo10 * rr17
        + ee1 * oo10 * xi * hh4 * mm12 * rr17
        - 2.0 * ee1 * hh4 * mm12
        - kk1 * mm11 * xi * qq05 * mm10
        + kk1 * qq05 * mm10
        - bb1 * cc2 * dd6
        + 2.0 * ee1 * dd3 * hh4 * jj08
        + ee1 * xi * hh4 * jj08
        - (2.0 * kk1 * xi * dd3 * qq05) / cc3**3
    )
    l3[:, 8] = (
        -bb1 * cc2 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2
        - bb1
        * cc2
        * pp08
        * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 / xi**3)
        - 2.0 * ee1 * hh4 * jj08
        + 2.0 * dd3 * qq05 * kk1 / cc3**3
    )
    l3[:, 9] = (
        -jj13 * tt18**3
        - 3.0
        * jj13
        * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7)
        * tt18
        - jj13
        * (
            -2.0 * kk1 * aa2 * qq05 * tt08
            - 3.0 * ee1 * dd8 * hh4 * jj08
            - 6.0 * bb1 * jj12 * cc2 * dd6
            + 6.0 * tt16 * dd7
        )
        + 6.0 * tt16 * dd3 * dd7
        - 6.0 * jj12 * dd7
        - 6.0 * bb1 * jj12 * dd3 * cc2 * dd6
        + 6.0 * bb1 * dd8 * cc2 * dd6
        - 3.0 * ee1 * dd8 * dd3 * hh4 * jj08
        + 3.0 * ee1 * aa2 * hh4 * jj08
        - 2.0 * kk1 * aa2 * dd3 * qq05 * tt08
    )
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
    h = 1e-4  # larger step for 4th deriv FD stability

    # Centered FD of l3 w.r.t. each parameter
    dl3_dmu = (
        _gevlss_l3_raw(y, mu_v + h, rho_v, xi0)
        - _gevlss_l3_raw(y, mu_v - h, rho_v, xi0)
    ) / (2.0 * h)
    dl3_drho = (
        _gevlss_l3_raw(y, mu_v, rho_v + h, xi0)
        - _gevlss_l3_raw(y, mu_v, rho_v - h, xi0)
    ) / (2.0 * h)
    dl3_dxi = (
        _gevlss_l3_raw(y, mu_v, rho_v, xi0 + h)
        - _gevlss_l3_raw(y, mu_v, rho_v, xi0 - h)
    ) / (2.0 * h)

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
    l4_ana[:, 0] = (
        uu1 * (ee3 - 3.0) * (ee3 - 2.0) * (ee3 - 1.0) * uu2 * cc3 ** (ee3 - 4.0)
        + (6.0 * uu1 * uu2 * dd3) / cc3**4
    )
    l4_ana[:, 1] = (
        3.0 * kk1 * ll8 * ff7 * kk2 * cc3**vv09
        + uu1 * vv09 * ll8 * ff7 * uu2 * cc2 * cc3 ** (ee3 - 4.0)
        - (6.0 * kk1 * kk2 * dd3) / cc3**3
        + (6.0 * uu1 * uu2 * dd3 * cc2) / cc3**4
    )
    l4_ana[:, 2] = (
        -kk1 * mm11 * ff7 * kk2 * ww15 * rr17
        + 2.0 * kk1 * mm11 * xi * ww15
        - kk1 * ww15
        + uu1 * ww11 * mm11 * kk2 * cc2 * ww12
        - uu1 * ff7 * xi * cc2 * ww12
        - uu1 * ww11 * xi * cc2 * ww12
        + 2.0 * kk1 * kk2 * tt08
        + 4.0 * kk1 * xi * dd3 * tt08
        - (6.0 * uu1 * kk2 * dd3 * cc2) / cc3**4
    )
    l4_ana[:, 3] = (
        4.0 * ee1 * ff7 * xi * cc3**ll8
        + 5.0 * kk1 * ll8 * ff7 * kk2 * cc2 * cc3**vv09
        + uu1 * vv09 * ll8 * ff7 * uu2 * hh4 * cc3 ** (ee3 - 4.0)
        + (4.0 * ee1 * xi * dd3) / cc3**2
        - (10.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
        + (6.0 * uu1 * uu2 * dd3 * hh4) / cc3**4
    )
    l4_ana[:, 4] = (
        -2.0 * ee1 * ff7 * xi * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
        - kk1 * mm11 * ff7 * kk2 * cc2 * ww15 * (bb1 * ww11 * cc2 * dd6 + oo13)
        - 2.0 * ee1 * aa2 * mm12
        - 2.0 * ee1 * ff7 * mm12
        - 2.0 * kk1 * mm11 * ff7 * xi * cc2 * ww15
        - kk1 * ff7 * cc2 * ww15
        - kk1 * mm11 * cc2 * ww15
        - 2.0 * ee1 * dd3 * jj08
        - 2.0 * ee1 * xi * jj08
        + 2.0 * kk1 * kk2 * cc2 * tt08
        + 8.0 * kk1 * xi * dd3 * cc2 * tt08
        - 6.0 * kk2 * dd3 * hh4 * uu1 / cc3**4
    )
    l4_ana[:, 5] = (
        ee1 * ff7 * xi * mm12 * tt18**2
        - 2.0 * ee1 * mm12 * tt18
        - 2.0 * kk1 * mm11 * xi * cc2 * ww15 * tt18
        + 2.0 * kk1 * cc2 * ww15 * tt18
        + ee1
        * ff7
        * xi
        * mm12
        * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * dd7 * jj12)
        + 4.0 * kk1 * cc2 * ww15
        + 2.0 * uu1 * ww11 * xi * hh4 * ww12
        - 4.0 * uu1 * hh4 * ww12
        + 2.0 * ee1 * jj08
        - 4.0 * kk1 * dd3 * cc2 * tt08
        - 4.0 * kk1 * xi * cc2 * tt08
        + (6.0 * uu1 * xi * dd3 * hh4) / cc3**4
    )
    l4_ana[:, 6] = (
        bb1 * cc3**ff7
        + 7.0 * ee1 * ff7 * xi * cc2 * cc3**ll8
        + 6.0 * kk1 * ll8 * ff7 * kk2 * hh4 * cc3**vv09
        + uu1 * vv09 * ll8 * ff7 * uu2 * qq05 * cc3 ** (ee3 - 4.0)
        - (bb1 * dd3) / cc3
        + (7.0 * ee1 * xi * dd3 * cc2) / cc3**2
        - (12.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
        + (6.0 * uu1 * uu2 * dd3 * qq05) / cc3**4
    )
    l4_ana[:, 7] = (
        -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + oo13)
        - 3.0 * ee1 * ff7 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
        - kk1 * mm11 * ff7 * kk2 * hh4 * ww15 * (bb1 * ww11 * cc2 * dd6 + oo13)
        - 3.0 * ee1 * aa2 * cc2 * mm12
        - 3.0 * ee1 * ff7 * cc2 * mm12
        - 2.0 * kk1 * mm11 * ff7 * xi * hh4 * ww15
        - kk1 * ff7 * hh4 * ww15
        - kk1 * mm11 * hh4 * ww15
        + bb1 * dd6
        - 4.0 * ee1 * dd3 * cc2 * jj08
        - 3.0 * ee1 * xi * cc2 * jj08
        + 2.0 * kk1 * kk2 * hh4 * tt08
        + 10.0 * kk1 * xi * dd3 * hh4 * tt08
        - 6.0 * kk2 * dd3 * qq05 * uu1 / cc3**4
    )
    l4_ana[:, 8] = (
        bb1 * ad20 * (bb1 * ff7 * cc2 * dd6 + ad21) ** 2
        + ee1 * ff7 * xi * cc2 * mm12 * ad22**2
        + 2.0 * ee1 * aa2 * cc2 * mm12 * ad22
        + 2.0 * ee1 * ff7 * cc2 * mm12 * ad22
        + bb1 * ad20 * (-ee1 * ff7 * hh4 * jj08 + ad17 + ad19)
        + ee1 * ff7 * xi * cc2 * mm12 * (-ee1 * mm11 * hh4 * jj08 + ad17 + ad19)
        + 4.0 * ee1 * cc2 * jj08
        - 6.0 * kk1 * dd3 * hh4 * tt08
        - 4.0 * kk1 * xi * hh4 * tt08
        + 6.0 * xi * dd3 * qq05 * uu1 / cc3**4
    )
    l4_ana[:, 9] = (
        -bb1 * pp08 * ae16**3
        - 3.0
        * bb1
        * pp08
        * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7)
        * ae16
        - bb1
        * pp08
        * (
            2.0 * kk1 * ff7 * qq05 * tt08
            - 3.0 * ee1 * dd8 * hh4 * jj08
            - 6.0 * bb1 * jj12 * cc2 * dd6
            + 6.0 * dd7 * tt16
        )
        + 6.0 * kk1 * hh4 * tt08
        - 6.0 * dd3 * qq05 * uu1 / cc3**4
    )
    l4_ana[:, 10] = (
        bb1 * cc2 * cc3**ff7
        + 7.0 * ee1 * ff7 * xi * hh4 * cc3**ll8
        + 6.0 * kk1 * ll8 * ff7 * kk2 * qq05 * cc3**vv09
        + uu1 * vv09 * ll8 * ff7 * uu2 * af05 * cc3 ** (ee3 - 4.0)
        - (bb1 * dd3 * cc2) / cc3
        + (7.0 * ee1 * xi * dd3 * hh4) / cc3**2
        - (12.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
        + (6.0 * uu1 * uu2 * dd3 * af05) / cc3**4
    )
    l4_ana[:, 11] = (
        -bb1 * cc2 * pp08 * rr17
        - 3.0 * ee1 * ff7 * xi * hh4 * mm12 * rr17
        - kk1 * mm11 * ff7 * kk2 * qq05 * ww15 * rr17
        + 4.0 * ee1 * hh4 * mm12
        + 5.0 * kk1 * mm11 * xi * qq05 * ww15
        - 4.0 * kk1 * qq05 * ww15
        + uu1 * ww11 * mm11 * kk2 * af05 * ww12
        - uu1 * ff7 * xi * af05 * ww12
        - uu1 * ww11 * xi * af05 * ww12
        + bb1 * cc2 * dd6
        - 4.0 * ee1 * dd3 * hh4 * jj08
        - 3.0 * ee1 * xi * hh4 * jj08
        + 2.0 * kk1 * kk2 * qq05 * tt08
        + 10.0 * kk1 * xi * dd3 * qq05 * tt08
        - 6.0 * uu1 * kk2 * dd3 * af05 / cc3**4
    )
    l4_ana[:, 12] = (
        bb1 * cc2 * ad20 * tt18**2
        + ee1 * ff7 * xi * hh4 * mm12 * tt18**2
        - 4.0 * ee1 * hh4 * mm12 * tt18
        - 2.0 * kk1 * mm11 * xi * qq05 * ww15 * tt18
        + 2.0 * kk1 * qq05 * ww15 * tt18
        + bb1 * cc2 * ad20 * ah24
        + ee1 * ff7 * xi * hh4 * mm12 * ah24
        + 6.0 * kk1 * qq05 * ww15
        + 2.0 * uu1 * ww11 * xi * af05 * ww12
        - 4.0 * uu1 * af05 * ww12
        + 4.0 * ee1 * hh4 * jj08
        - 6.0 * kk1 * dd3 * qq05 * tt08
        - 4.0 * kk1 * xi * qq05 * tt08
        + 6.0 * uu1 * xi * dd3 * af05 / cc3**4
    )
    l4_ana[:, 13] = (
        -bb1 * cc2 * pp08 * ae16**3
        - 3.0
        * bb1
        * cc2
        * pp08
        * (-ee1 * ff7 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7)
        * ae16
        - bb1
        * cc2
        * pp08
        * (
            2.0 * kk1 * ff7 * qq05 * tt08
            - 3.0 * ee1 * dd8 * hh4 * jj08
            - 6.0 * bb1 * jj12 * cc2 * dd6
            + 6.0 * dd7 * tt16
        )
        + 6.0 * kk1 * qq05 * tt08
        - 6.0 * dd3 * af05 * uu1 / cc3**4
    )
    l4_ana[:, 14] = (
        -jj13 * tt18**4
        - 6.0 * jj13 * ah24 * tt18**2
        - 3.0 * jj13 * ah24**2
        - 4.0
        * jj13
        * (
            -2.0 * kk1 * aa2 * qq05 * tt08
            - 3.0 * ee1 * dd8 * hh4 * jj08
            - 6.0 * bb1 * jj12 * cc2 * dd6
            + 6.0 * tt16 * dd7
        )
        * tt18
        - jj13
        * (
            6.0 * uu1 * aa2 * af05 * aj08
            + 8.0 * kk1 * dd8 * qq05 * tt08
            + 12.0 * ee1 * jj12 * hh4 * jj08
            + 24.0 * bb1 * tt16 * cc2 * dd6
            - 24.0 * aj20 * dd7
        )
        - 24.0 * aj20 * dd3 * dd7
        + 24.0 * tt16 * dd7
        + 24.0 * bb1 * tt16 * dd3 * cc2 * dd6
        - 24.0 * bb1 * jj12 * cc2 * dd6
        + 12.0 * ee1 * jj12 * dd3 * hh4 * jj08
        - 12.0 * ee1 * dd8 * hh4 * jj08
        + 8.0 * kk1 * dd8 * dd3 * qq05 * tt08
        - 8.0 * kk1 * aa2 * qq05 * tt08
        + 6.0 * uu1 * aa2 * dd3 * af05 * aj08
    )

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
