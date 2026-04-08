from __future__ import annotations

import numpy as np
import pytest

from nampy.basemodels.gam import GAM
from nampy.gam.smoothing_selection.criteria.dispatch import (
    criterion_gradient,
    criterion_gradient_numerical,
    criterion_hessian,
    criterion_hessian_numerical,
)
from nampy.gam.smoothing_selection.criteria.pirls_deriv import (
    _gamma_saturated_loglik_scale_derivatives,
    _pirls_laplace_term_derivatives,
    criterion_gradient_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_exact,
)
from nampy.gam.smoothing_selection.criteria.penalty import _static_penalty_null_dim
from nampy.gam.smoothing_selection.criteria.laplace import _penalty_derivative_matrices
from nampy.gam.smoothing_selection.criteria.pirls import _ensure_penalty_reparameterization
from nampy.gam.smoothing_selection.criteria.pirls_reml_derivative_blocks import (
    _penalty_quadratic_and_sp_derivatives,
    _deviance_chained_to_smoothing,
    _pearson_coefficient_derivatives,
    _working_weight_derivatives_wrt_linpred,
)

from mgcv_parity_utils import _make_binomial_data, _make_gamma_data, _make_poisson_data


def _fit_reml_model(data, formula, family):
    gam = GAM(formula=formula, family=family, smoothing_method="REML")
    gam.fit(data=data)
    y = gam.family.validate_y(np.asarray(data["y"], dtype=np.float64))
    sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
    log_sp = np.log(np.clip(sp, 1e-300, None))
    return gam, y, log_sp


def _gamma_pirls_smoothing_derivative_state(gam, y, sp):
    _ensure_penalty_reparameterization(gam)
    sol = gam._solve_pirls_given_smoothing(y, sp)
    X = np.asarray(sol["X"], dtype=np.float64)
    beta = np.asarray(sol["coef_full"], dtype=np.float64)
    eta = np.asarray(sol["eta"], dtype=np.float64)
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    P_derivs = _penalty_derivative_matrices(gam, sp)
    dW_eta, _ = _working_weight_derivatives_wrt_linpred(gam, y, eta, sol["mu"], W)

    n_sp = len(P_derivs)
    dbeta = []
    d2beta_mat = [[None] * n_sp for _ in range(n_sp)]
    dA = []
    for Pj in P_derivs:
        dbj = -(A_inv @ (Pj @ beta)) if np.any(Pj) else np.zeros_like(beta)
        dbeta.append(dbj)
        dA.append(X.T @ ((dW_eta * (X @ dbj))[:, None] * X) + Pj)

    for j, Pj in enumerate(P_derivs):
        for k in range(j, n_sp):
            delta = 1.0 if j == k else 0.0
            d2b = -(A_inv @ (dA[k] @ dbeta[j] + Pj @ dbeta[k] + delta * (Pj @ beta)))
            d2beta_mat[j][k] = d2b
            d2beta_mat[k][j] = d2b

    return sol, dbeta, d2beta_mat


@pytest.mark.parametrize(
    ("family", "data_factory", "seed"),
    [
        ("poisson", _make_poisson_data, 321),
        ("binomial", _make_binomial_data, 322),
    ],
)
def test_exact_pirls_gradient_matches_finite_difference(family, data_factory, seed):
    data = data_factory(seed=seed, n=160)
    gam, y, log_sp = _fit_reml_model(
        data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family,
    )

    exact = criterion_gradient_ml_reml_pirls_exact(gam, y, log_sp, "REML")
    fd = criterion_gradient_numerical(gam, y, log_sp, method="reml", eps_abs=1e-6, eps_rel=1e-5)

    np.testing.assert_allclose(exact, fd, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        criterion_gradient(gam, y, log_sp, method="reml"),
        exact,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("family", "data_factory", "seed"),
    [
        ("poisson", _make_poisson_data, 421),
        ("binomial", _make_binomial_data, 422),
    ],
)
def test_exact_pirls_hessian_matches_finite_difference(family, data_factory, seed):
    data = data_factory(seed=seed, n=140)
    gam, y, log_sp = _fit_reml_model(
        data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family,
    )

    exact = criterion_hessian_ml_reml_pirls_exact(gam, y, log_sp, "REML")
    fd = criterion_hessian_numerical(gam, y, log_sp, method="reml", eps_abs=1e-4, eps_rel=1e-3)

    np.testing.assert_allclose(exact, fd, rtol=2e-4, atol=5e-5)
    np.testing.assert_allclose(
        criterion_hessian(gam, y, log_sp, method="reml"),
        exact,
        rtol=0.0,
        atol=1e-9,
    )


def test_exact_pirls_gamma_gradient_matches_finite_difference():
    data = _make_gamma_data(seed=511, n=120)
    gam, y, log_sp = _fit_reml_model(
        data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
    )

    exact = criterion_gradient_ml_reml_pirls_exact(gam, y, log_sp, "REML")
    fd = criterion_gradient_numerical(gam, y, log_sp, method="reml", eps_abs=1e-6, eps_rel=1e-5)

    np.testing.assert_allclose(exact, fd, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        criterion_gradient(gam, y, log_sp, method="reml"),
        exact,
        rtol=0.0,
        atol=1e-12,
    )


def test_gamma_pirls_gradient_records_direct_laplace_k1_decomposition():
    data = _make_gamma_data(seed=513, n=120)
    gam, y, log_sp = _fit_reml_model(
        data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
    )

    grad = criterion_gradient_ml_reml_pirls_exact(gam, y, log_sp, "REML")
    state = getattr(gam, "_pirls_reml_gamma_state_", None)
    assert state is not None

    phi = float(state["phi"])
    Dp1 = np.asarray(state["Dp1"], dtype=np.float64)
    K1 = np.asarray(state["K1"], dtype=np.float64)

    fixed_mask = (
        np.zeros(gam.n_smoothing_params_, dtype=bool)
        if gam.smoothing_fixed_mask_ is None
        else np.asarray(gam.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~fixed_mask

    reconstructed = (
        Dp1[free_mask] / (2.0 * phi)
        + K1[free_mask]
    )
    np.testing.assert_allclose(grad, reconstructed, rtol=0.0, atol=1e-12)


def test_gamma_pirls_hessian_records_direct_laplace_k2_decomposition():
    data = _make_gamma_data(seed=514, n=120)
    gam, y, log_sp = _fit_reml_model(
        data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
    )

    sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
    sol = gam._solve_pirls_given_smoothing(y, sp)
    X = np.asarray(sol["X"], dtype=np.float64)
    beta = np.asarray(sol["coef_full"], dtype=np.float64)
    eta = np.asarray(sol["eta"], dtype=np.float64)
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    from nampy.gam.smoothing_selection.criteria.laplace import _penalty_derivative_matrices
    from nampy.gam.smoothing_selection.criteria.pirls import _ensure_penalty_reparameterization
    from nampy.gam.smoothing_selection.criteria.pirls_reml_derivative_blocks import (
        _working_weight_derivatives_wrt_linpred,
    )

    _ensure_penalty_reparameterization(gam)
    P_derivs = _penalty_derivative_matrices(gam, sp)
    dW_eta, d2W_eta = _working_weight_derivatives_wrt_linpred(gam, y, eta, sol["mu"], W)
    n_sp = len(P_derivs)
    dbeta = []
    d2beta_mat = [[None] * n_sp for _ in range(n_sp)]
    dA = []
    for Pj in P_derivs:
        dbj = -(A_inv @ (Pj @ beta)) if np.any(Pj) else np.zeros_like(beta)
        dbeta.append(dbj)
        dA.append(X.T @ ((dW_eta * (X @ dbj))[:, None] * X) + Pj)
    for j, Pj in enumerate(P_derivs):
        for k in range(j, n_sp):
            delta = 1.0 if j == k else 0.0
            d2b = -(A_inv @ (dA[k] @ dbeta[j] + Pj @ dbeta[k] + delta * (Pj @ beta)))
            d2beta_mat[j][k] = d2b
            d2beta_mat[k][j] = d2b

    _, K1, K2 = _pirls_laplace_term_derivatives(
        gam, sol, sp, dbeta, d2beta_mat, dW_eta, d2W_eta, method="REML"
    )

    steps = np.maximum(1e-4, 1e-3 * (1.0 + np.abs(log_sp)))
    fd = np.zeros_like(K2)
    for j, h in enumerate(steps):
        rho_plus = log_sp.copy()
        rho_minus = log_sp.copy()
        rho_plus[j] += h
        rho_minus[j] -= h
        sp_plus = np.exp(rho_plus)
        sp_minus = np.exp(rho_minus)
        sol_plus = gam._solve_pirls_given_smoothing(y, sp_plus)
        sol_minus = gam._solve_pirls_given_smoothing(y, sp_minus)

        def _k1(sol_local, sp_local):
            X_local = np.asarray(sol_local["X"], dtype=np.float64)
            beta_local = np.asarray(sol_local["coef_full"], dtype=np.float64)
            eta_local = np.asarray(sol_local["eta"], dtype=np.float64)
            W_local = np.asarray(sol_local["working_weights"], dtype=np.float64)
            A_inv_local = np.asarray(sol_local["A_inv"], dtype=np.float64)
            P_derivs_local = _penalty_derivative_matrices(gam, sp_local)
            dW_eta_local, d2W_eta_local = _working_weight_derivatives_wrt_linpred(
                gam, y, eta_local, sol_local["mu"], W_local
            )
            n_local = len(P_derivs_local)
            dbeta_local = []
            d2beta_local = [[None] * n_local for _ in range(n_local)]
            dA_local = []
            for Pm in P_derivs_local:
                dbm = -(A_inv_local @ (Pm @ beta_local)) if np.any(Pm) else np.zeros_like(beta_local)
                dbeta_local.append(dbm)
                dA_local.append(X_local.T @ ((dW_eta_local * (X_local @ dbm))[:, None] * X_local) + Pm)
            for a, Pa in enumerate(P_derivs_local):
                for b in range(a, n_local):
                    delta_ab = 1.0 if a == b else 0.0
                    d2b_val = -(
                        A_inv_local
                        @ (dA_local[b] @ dbeta_local[a] + Pa @ dbeta_local[b] + delta_ab * (Pa @ beta_local))
                    )
                    d2beta_local[a][b] = d2b_val
                    d2beta_local[b][a] = d2b_val
            return _pirls_laplace_term_derivatives(
                gam,
                sol_local,
                sp_local,
                dbeta_local,
                d2beta_local,
                dW_eta_local,
                d2W_eta_local,
                method="REML",
            )[1]

        k1_plus = _k1(sol_plus, sp_plus)
        k1_minus = _k1(sol_minus, sp_minus)
        fd[:, j] = (k1_plus - k1_minus) / (2.0 * h)
    fd = 0.5 * (fd + fd.T)

    np.testing.assert_allclose(K2, fd, rtol=2e-4, atol=5e-5)


def test_gamma_pirls_pearson_eta_derivatives_match_finite_difference():
    data = _make_gamma_data(seed=515, n=120)
    gam, y, log_sp = _fit_reml_model(
        data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
    )

    sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
    sol, _, _ = _gamma_pirls_smoothing_derivative_state(gam, y, sp)
    eta = np.asarray(sol["eta"], dtype=np.float64)
    mu = np.asarray(sol["mu"], dtype=np.float64)
    X = np.asarray(sol["X"], dtype=np.float64)
    pearson, pearson_grad, pearson_hess = _pearson_coefficient_derivatives(
        gam,
        y,
        eta,
        mu,
        X,
    )
    family = gam.family
    g1 = 1.0 / np.clip(np.asarray(family.mu_eta(eta), dtype=np.float64), 1e-14, None)
    V = np.clip(np.asarray(family.variance(mu), dtype=np.float64), 1e-14, None)
    V1 = np.asarray(family.dvar(mu), dtype=np.float64) / V
    V2 = np.asarray(family.d2var(mu), dtype=np.float64) / V
    g2 = np.asarray(family.d2link(mu), dtype=np.float64) / g1
    resid = y - mu
    xx = resid / V
    pe1 = -xx * (2.0 + resid * V1) / g1
    pe2 = (
        -pe1 * g2 / g1
        + (2.0 / V + 2.0 * xx * V1 - pe1 * V1 * g1 - xx * resid * (V2 - V1 * V1))
        / (g1 * g1)
    )

    h = 1e-6
    fd1 = np.zeros_like(pe1)
    fd2 = np.zeros_like(pe2)
    for i in range(y.size):
        eta_plus = eta.copy()
        eta_minus = eta.copy()
        eta_plus[i] += h
        eta_minus[i] -= h
        mu_plus = np.asarray(family.inverse_link(eta_plus), dtype=np.float64)
        mu_minus = np.asarray(family.inverse_link(eta_minus), dtype=np.float64)
        var_plus = np.clip(np.asarray(family.variance(mu_plus), dtype=np.float64), 1e-14, None)
        var_minus = np.clip(np.asarray(family.variance(mu_minus), dtype=np.float64), 1e-14, None)
        f0 = float(((y[i] - mu[i]) ** 2) / V[i])
        fp = float(((y[i] - mu_plus[i]) ** 2) / var_plus[i])
        fm = float(((y[i] - mu_minus[i]) ** 2) / var_minus[i])
        fd1[i] = (fp - fm) / (2.0 * h)
        fd2[i] = (fp - 2.0 * f0 + fm) / (h * h)

    np.testing.assert_allclose(pe1, fd1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pe2, fd2, rtol=2e-4, atol=5e-4)

    expected_grad = X.T @ pe1
    expected_hess = X.T @ (pe2[:, None] * X)
    np.testing.assert_allclose(pearson_grad, expected_grad, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(pearson_hess, expected_hess, rtol=0.0, atol=1e-12)


def test_gamma_penalty_quadratic_second_derivatives_match_finite_difference():
    data = _make_gamma_data(seed=516, n=120)
    gam, y, log_sp = _fit_reml_model(
        data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
    )

    sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
    sol, dbeta, d2beta_mat = _gamma_pirls_smoothing_derivative_state(gam, y, sp)
    beta = np.asarray(sol["coef_full"], dtype=np.float64)
    P_derivs = _penalty_derivative_matrices(gam, sp)
    _, bSb1, bSb2 = _penalty_quadratic_and_sp_derivatives(
        beta=beta,
        P_total=np.asarray(sol["P"], dtype=np.float64),
        P_derivs=P_derivs,
        dbeta_cols=dbeta,
        d2beta_mat=d2beta_mat,
    )

    def _bSb1_at(log_sp_local):
        sp_local = np.exp(log_sp_local)
        sol_local, dbeta_local, d2beta_local = _gamma_pirls_smoothing_derivative_state(gam, y, sp_local)
        beta_local = np.asarray(sol_local["coef_full"], dtype=np.float64)
        P_derivs_local = _penalty_derivative_matrices(gam, sp_local)
        _, bSb1_local, _ = _penalty_quadratic_and_sp_derivatives(
            beta=beta_local,
            P_total=np.asarray(sol_local["P"], dtype=np.float64),
            P_derivs=P_derivs_local,
            dbeta_cols=dbeta_local,
            d2beta_mat=d2beta_local,
        )
        return bSb1_local

    steps = np.maximum(1e-4, 1e-3 * (1.0 + np.abs(log_sp)))
    fd_hess = np.zeros_like(bSb2)
    for j, h in enumerate(steps):
        plus = log_sp.copy()
        minus = log_sp.copy()
        plus[j] += h
        minus[j] -= h
        g_plus = _bSb1_at(plus)
        g_minus = _bSb1_at(minus)
        fd_hess[:, j] = (g_plus - g_minus) / (2.0 * h)
    fd_hess = 0.5 * (fd_hess + fd_hess.T)

    np.testing.assert_allclose(bSb2, fd_hess, rtol=2e-4, atol=5e-5)


def test_gamma_pirls_hessian_dispatch_matches_finite_difference():
    data = _make_gamma_data(seed=512, n=120)
    gam, y, log_sp = _fit_reml_model(
        data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
    )

    exact = criterion_hessian_ml_reml_pirls_exact(gam, y, log_sp, "REML")
    fd = criterion_hessian_numerical(gam, y, log_sp, method="reml", eps_abs=1e-4, eps_rel=1e-3)

    np.testing.assert_allclose(exact, fd, rtol=2e-4, atol=5e-5)
    np.testing.assert_allclose(
        criterion_hessian(gam, y, log_sp, method="reml"),
        exact,
        rtol=0.0,
        atol=1e-9,
    )


