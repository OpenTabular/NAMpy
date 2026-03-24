"""Equivalence checks for Gaussian REML Laplace score algebra."""

import numpy as np
import pytest

from nampy.gam.smoothness.criteria.gaussian_reml_algebra import (
    deviance_method_scale_estimate,
    gaussian_reml_laplace_score,
    gaussian_reml_saturation_terms_wrt_variance,
    gaussian_reml_weighted_degrees_and_log_weight_term,
    gaussian_weighted_residual_sum_squares,
    pearson_method_scale_estimate,
    prior_weights_diagonal_from_fit,
    profiled_gaussian_reml_variance,
    quadratic_form_penalty,
)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_gaussian_reml_laplace_matches_profiled_closed_form(seed: int) -> None:
    """Concentrated REML equals Laplace assembly for unit weights, gamma=1."""
    rng = np.random.default_rng(seed)
    n = 40
    Mp = 3.0
    rss_bSb = float(rng.uniform(5.0, 80.0))
    logdet_A = float(rng.uniform(-2.0, 6.0))
    logdet_S = float(rng.uniform(-4.0, 2.0))
    denom = float(n - Mp)
    scale = rss_bSb / denom
    weights = np.ones(n, dtype=np.float64)

    closed = (
        rss_bSb / scale + (n - Mp) * np.log(2.0 * np.pi * scale) + logdet_A - logdet_S
    ) / 2.0

    dev = rss_bSb - float(
        rng.uniform(0.0, min(rss_bSb * 0.5, rss_bSb - 1e-6))
    )
    penalty_P = rss_bSb - dev

    assembled = gaussian_reml_laplace_score(
        dev,
        penalty_P,
        scale,
        logdet_A - logdet_S,
        Mp,
        weights,
        gamma=1.0,
        reml=True,
    )

    np.testing.assert_allclose(assembled, closed, rtol=0.0, atol=2e-14)


def test_deviance_penalty_scale_match_gaussian_solve() -> None:
    """Weighted RSS, quadratic penalty, and deviance-method scale identities."""
    rng = np.random.default_rng(7)
    n = 25
    q = 6
    y = rng.standard_normal(n)
    mu = rng.standard_normal(n)
    w = np.ones(n, dtype=np.float64)
    dev = gaussian_weighted_residual_sum_squares(y, mu, w)
    rss = float(np.sum((y - mu) ** 2))
    np.testing.assert_allclose(dev, rss, rtol=0.0, atol=1e-15)

    beta = rng.standard_normal(q)
    P = np.eye(q, dtype=np.float64)
    P[0, 1] = P[1, 0] = 0.25
    P = (P + P.T) * 0.5
    pen = quadratic_form_penalty(beta, P)
    np.testing.assert_allclose(pen, float(beta @ (P @ beta)), rtol=0.0, atol=1e-15)

    tr_a = 4.2
    sig2_est = deviance_method_scale_estimate(dev, tr_a, float(n), dev_extra=0.0)
    np.testing.assert_allclose(sig2_est, dev / (n - tr_a), rtol=0.0, atol=1e-15)

    Mp = 2.0
    sp = profiled_gaussian_reml_variance(dev, pen, float(n), Mp)
    np.testing.assert_allclose(sp, (dev + pen) / (n - Mp), rtol=0.0, atol=1e-15)


def test_pearson_scale_fletcher_identity() -> None:
    """Fletcher branch is a no-op when dV/V is zero (Gaussian)."""
    rng = np.random.default_rng(2)
    n = 12
    y = rng.standard_normal(n)
    mu = rng.standard_normal(n)
    pearson = float(np.sum((y - mu) ** 2))
    tr_a = 3.0
    s0 = pearson_method_scale_estimate(pearson, tr_a, float(n), fletcher=False)
    s1 = pearson_method_scale_estimate(
        pearson,
        tr_a,
        float(n),
        fletcher=True,
        y=y,
        mu=mu,
        dvar_over_var=np.zeros(n, dtype=np.float64),
    )
    np.testing.assert_allclose(s0, s1, rtol=0.0, atol=1e-15)


def test_weighted_gaussian_reml_laplace_matches_expanded_score() -> None:
    """Laplace score with non-unit weights matches expanded saturation form."""
    rng = np.random.default_rng(5)
    n = 18
    Mp = 2.0
    w = rng.uniform(0.4, 1.9, size=n).astype(np.float64)
    rss_bSb = float(rng.uniform(4.0, 40.0))
    logdet_A = float(rng.uniform(-1.0, 4.0))
    logdet_S = float(rng.uniform(-3.0, 1.5))
    denom = float(n - Mp)
    scale = rss_bSb / denom
    dev = rss_bSb - float(rng.uniform(0.0, min(rss_bSb * 0.4, rss_bSb - 1e-6)))
    penalty_P = rss_bSb - dev
    assembled = gaussian_reml_laplace_score(
        dev,
        penalty_P,
        scale,
        logdet_A - logdet_S,
        Mp,
        w,
        gamma=1.0,
        reml=True,
    )
    ls0 = gaussian_reml_saturation_terms_wrt_variance(w, scale)[0]
    closed = (
        (dev + penalty_P) / (2.0 * scale)
        - ls0
        + 0.5 * (logdet_A - logdet_S)
        - Mp * (np.log(2.0 * np.pi * scale) / 2.0)
    )
    np.testing.assert_allclose(assembled, closed, rtol=0.0, atol=2e-14)


def test_prior_weights_diagonal_from_fit() -> None:
    n = 5
    w = np.array([1.0, 2.0, 0.5, 1.25, 1.0], dtype=np.float64)
    got = prior_weights_diagonal_from_fit({"working_weights": w}, n)
    np.testing.assert_array_equal(got, w)
    np.testing.assert_array_equal(
        prior_weights_diagonal_from_fit({"working_weights": w[:3]}, n),
        np.ones(n, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        prior_weights_diagonal_from_fit({}, n),
        np.ones(n, dtype=np.float64),
    )


def test_gaussian_saturation_terms_match_exponential_family_saturated() -> None:
    from nampy.gam.families.exponential import GaussianIdentityFamily

    fam = GaussianIdentityFamily()
    w = np.array([1.0, 2.0, 0.5, 0.0, 1.0])
    scale = 0.37
    vec = gaussian_reml_saturation_terms_wrt_variance(w, scale)
    ls = fam.saturated_loglik(np.zeros(len(w)), weights=w, scale=scale)
    np.testing.assert_allclose(vec[0], ls, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("seed", [0, 3, 19])
def test_joint_gaussian_reml_matches_laplace_score(seed: int) -> None:
    """
    Doubled joint Wood-style objective equals twice the Laplace REML score at any σ²
    when effective row count defaults to ``n_row``.
    """
    rng = np.random.default_rng(seed)
    n = 26
    w = rng.uniform(0.25, 2.2, size=n).astype(np.float64)
    Mp = float(rng.uniform(1.5, 4.0))
    F = float(rng.uniform(8.0, 120.0))
    sigma2 = float(rng.uniform(0.04, 3.0))
    logdet_diff = float(rng.uniform(-4.0, 6.0))
    dev = F * float(rng.uniform(0.25, 0.92))
    penalty_P = F - dev
    nu, sum_log_s = gaussian_reml_weighted_degrees_and_log_weight_term(w, float(n), Mp)
    joint = (
        F / sigma2 + nu * np.log(2.0 * np.pi * sigma2) - sum_log_s + logdet_diff
    ) / 2.0
    laplace = gaussian_reml_laplace_score(
        dev,
        penalty_P,
        sigma2,
        logdet_diff,
        Mp,
        w,
        gamma=1.0,
        reml=True,
    )
    np.testing.assert_allclose(joint, laplace, rtol=0.0, atol=1e-14)


def test_joint_gaussian_reml_zero_weight_rows_reduce_nu() -> None:
    """Effective ν uses ``sum(w > 0)`` when some weights are zero."""
    n = 8
    w = np.array([1.0, 0.0, 2.0, 0.5, 0.0, 1.25, 0.0, 1.0], dtype=np.float64)
    Mp = 1.0
    nu, _ = gaussian_reml_weighted_degrees_and_log_weight_term(w, float(n), Mp)
    n_ls = float(np.sum(w > 0.0))
    np.testing.assert_allclose(nu, n_ls - Mp, rtol=0.0, atol=1e-15)
