"""
Basic smoke tests for ziplss family (Zero-inflated Poisson location-scale).

Tests:
1. _l1ee and _lee1 value correctness
2. _ldg gradient finite-difference check
3. _lde gradient finite-difference check
4. _zipll log-lik matches direct formula
5. ziplss ll gradient finite-difference check
6. ziplss ll Hessian shape and negative semi-definiteness
7. ziplss initialize returns correct-shaped finite vector
8. gam_fit5 convergence on simulated ZIP data

Mirrors mgcv ziplss behaviour at a high level.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import gammaln

from nampy.gam.families.gamlss import ZiplssFamily, ziplss, _l1ee, _lee1, _ldg, _lde, _zipll
from nampy.gam.fit.solvers.gam_fit5 import GamFit5Control, gam_fit5


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
