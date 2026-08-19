from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from nampy.gam import GAM
from nampy.gam.families.gamlss import gaulss
from nampy.gam.fit.solvers.general_family.newton import (
    GeneralNewtonControl,
    solve_general_newton_fit,
)

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

    # Direct formula: N(mu, 1/tau^2)
    eta0 = X[:, jj[0]] @ coef[jj[0]]
    eta1 = X[:, jj[1]] @ coef[jj[1]]
    mu_ref = eta0  # identity link
    tau_ref = 1.0 / (np.exp(eta1) + 0.01)  # logb link
    l_ref = float(
        np.sum(
            -0.5 * (y - mu_ref) ** 2 * tau_ref**2
            - 0.5 * np.log(2 * np.pi)
            + np.log(tau_ref)
        )
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
    # Hessian of log-lik should be negative semi-definite
    ev = np.linalg.eigvalsh(lbb)
    assert np.all(
        ev <= 1e-10
    ), f"Hessian of log-lik has positive eigenvalue: {ev.max():.4g}"


def test_gaulss_supports_sqrt_mean_link():
    """gaulss supports mgcv's sqrt link for the mean predictor."""
    fam = gaulss(link=("sqrt", "logb"))
    eta = np.array([0.4, 0.8, 1.2], dtype=np.float64)
    mu = fam.linfo[0].linkinv(eta)

    assert_allclose(mu, eta**2, rtol=1e-12, atol=1e-12)
    assert_allclose(fam.linfo[0].mu_eta(eta), 2.0 * eta, rtol=1e-12, atol=1e-12)
    assert fam.link_names == ("sqrt", "logb")


# ---------------------------------------------------------------------------
# 6. solve_general_newton_fit end-to-end on simulated data
# ---------------------------------------------------------------------------


def test_gam_fit5_simple_convergence():
    """
    solve_general_newton_fit with gaulss should converge to sensible estimates on simulated data.
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
    assert_allclose(coef_full[0], 0.5, atol=0.25)
    assert_allclose(coef_full[1], 1.5, atol=0.25)
    tau_est = gam.family.linfo[1].linkinv(np.array([coef_full[2]]))[0]
    assert_allclose(tau_est, 1.0 / sigma_true, atol=0.35)
