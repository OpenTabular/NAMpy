"""Exact BCCG score/observed Hessian and fixed-smoothing assembly gates."""

from itertools import combinations_with_replacement
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.gamlss.bccg import BCCGLikelihood, _log_normalizer
from nampy.gam.families.registry import make_gam_family
from nampy.gam.fit.selection import supports_smoothing_method


def _mp_loglikelihood(mp, y, log_mu, log_sigma, nu):
    lr = mp.log(y) - log_mu
    sigma = mp.exp(log_sigma)
    # Entire hypergeometric identity avoids the removable division at nu=0.
    z = lr / sigma * mp.hyp1f1(1, 2, nu * lr)
    log_c = 0 if nu == 0 else mp.log(mp.erfc(-1 / (sigma * abs(nu) * mp.sqrt(2))) / 2)
    return nu * lr - log_sigma - z * z / 2 - mp.log(y) - mp.log(2 * mp.pi) / 2 - log_c


@pytest.mark.parametrize(
    "eta",
    [
        [0.2, -0.4, -2.0],
        [0.3, 0.2, 2.0],
        [-0.4, 1.0, -0.1],
        [0.2, -0.4, -1e-12],
        [0.2, -0.4, 0.0],
        [0.2, -0.4, 1e-12],
        [0.7, 0.3, 0.01],
    ],
)
def test_exact_fourth_order_tensors_match_high_precision_differentiation(eta):
    mp = pytest.importorskip("mpmath")
    family = BCCGLikelihood()
    value = family._log_density(
        np.array([1.2]), *[Taylor(np.array([v]), 3, 4, k) for k, v in enumerate(eta)]
    )
    with mp.workdps(80):
        point = tuple(mp.mpf(v) for v in eta)

        def function(*z):
            return _mp_loglikelihood(mp, mp.mpf(1.2), *z)

        assert_allclose(value.value, float(function(*point)), atol=2e-13)
        for order in range(1, 5):
            expected = []
            for combination in combinations_with_replacement(range(3), order):
                degrees = tuple(combination.count(j) for j in range(3))
                expected.append(float(mp.diff(function, point, degrees)))
            assert_allclose(
                value.packed_derivatives(order)[0], expected, atol=2e-11, rtol=2e-10
            )


def test_weighted_coefficient_score_and_hessian_include_offsets_shared_columns_and_fixed_penalty():
    family = BCCGLikelihood()
    X = np.random.default_rng(34).normal(size=(7, 4))
    jj = [np.array([k, 3]) for k in range(3)]
    offsets = [np.linspace(-0.2, 0.1, 7), np.full(7, -0.5), np.linspace(-0.3, 0.3, 7)]
    beta = np.array([0.1, -0.2, 0.01, 0.05])
    weights = np.array([1.0, 0.0, 2.0, 0.4, 1.0, 3.0, 0.5])
    y = np.linspace(0.5, 2.0, 7)
    penalty = np.diag([0.1, 0.3, 0.2, 0.5])
    result = family.ll(y, X, jj, beta, weights, offset=offsets, deriv=1)

    def objective(b):
        return np.array(
            [
                family.ll(y, X, jj, b, weights, offset=offsets)["l"]
                - 0.5 * b @ penalty @ b
            ]
        )

    def score(b):
        return (
            family.ll(y, X, jj, b, weights, offset=offsets, deriv=1)["lb"] - penalty @ b
        )

    assert_allclose(score(beta), approx_derivative(objective, beta).ravel(), atol=2e-8)
    assert_allclose(result["lbb"] - penalty, approx_derivative(score, beta), atol=2e-7)
    repeated = np.repeat(np.arange(7), [1, 0, 2, 0, 1, 3, 0])
    integer_weights = np.array([1, 0, 2, 0, 1, 3, 0])
    weighted = family.ll(y, X, jj, beta, integer_weights, offset=offsets, deriv=1)
    duplicate = family.ll(
        y[repeated],
        X[repeated],
        jj,
        beta,
        None,
        offset=[o[repeated] for o in offsets],
        deriv=1,
    )
    for key in ["l", "lb", "lbb"]:
        assert_allclose(weighted[key], duplicate[key], atol=2e-13)


def test_likelihood_matches_shared_probability_law_and_moment_prediction():
    family = BCCGLikelihood()
    eta = np.array([[0.1, -0.5, -3], [0.4, 0.2, 0], [0.7, -0.4, 1]])
    parameters = family.distribution_parameters_from_eta(eta)
    value = family._log_density(
        np.array([0.4, 1.2, 2.0]), *[Taylor(eta[:, k], 3, 0) for k in range(3)]
    )
    assert_allclose(
        value.value,
        family.logpdf_from_parameters([0.4, 1.2, 2.0], parameters),
        atol=2e-13,
    )
    assert_allclose(
        family.point_prediction_from_parameters(parameters),
        family._distribution(parameters).mean(),
    )
    assert_allclose(
        family.variance_from_parameters(parameters),
        family._distribution(parameters).variance(),
    )


def test_fourth_order_capabilities_and_unsupported_higher_orders_are_explicit():
    family = BCCGLikelihood()
    assert family.max_derivative_order == 4
    with pytest.raises(ValueError, match="Unknown GAM family"):
        make_gam_family("bccg")
    assert isinstance(make_gam_family("bccg_lss"), BCCGLikelihood)
    for method in ["reml", "ml", "laml"]:
        assert supports_smoothing_method(SimpleNamespace(family=family), method)
    with pytest.raises(ValueError, match="three distinct"):
        family.initialize(
            np.ones(4), np.ones((4, 3)), [np.array([k]) for k in range(3)]
        )
    with pytest.raises(NotImplementedError, match="only through order four"):
        _log_normalizer(Taylor(np.array([0.5]), 1, 5, 0))


def test_recorded_r_scores_agree_but_working_curvature_is_not_observed_hessian():
    reference = pd.read_csv(
        Path(__file__).parents[1]
        / "reference_fixtures"
        / "bccg"
        / "derivative_hooks.csv"
    )
    eta = np.column_stack([np.log(reference.mu), np.log(reference.sigma), reference.nu])
    value = BCCGLikelihood()._log_density(
        reference.y.to_numpy(), *[Taylor(eta[:, k], 3, 2, k) for k in range(3)]
    )
    natural_score = value.packed_derivatives(1) / np.column_stack(
        [reference.mu, reference.sigma, np.ones(len(reference))]
    )
    assert_allclose(
        natural_score[:-1],
        reference[["score_mu", "score_sigma", "score_nu"]].to_numpy()[:-1],
        atol=2e-13,
    )
    assert (
        np.isnan(reference.score_nu.iloc[-1]) and np.isfinite(natural_score[-1]).all()
    )
    natural_mu_curvature = (
        value.packed_derivatives(2)[:, 0] - value.packed_derivatives(1)[:, 0]
    ) / reference.mu**2
    assert np.any(np.abs(natural_mu_curvature - reference.working_mu_mu) > 1)


def test_normalizer_chain_does_not_underflow_before_a_large_predictor_jacobian():
    mp = pytest.importorskip("mpmath")
    t = Taylor(np.array([1 / 41]), 1, 2)
    t.coefficients[(1,)] = np.array([1e200])
    value = _log_normalizer(t)
    with mp.workdps(450):
        x, chain = mp.mpf(1 / 41), mp.mpf(1e200)
        a = 1 / x
        ratio = mp.exp(-a * a / 2) / mp.sqrt(2 * mp.pi) / (mp.erfc(-a / mp.sqrt(2)) / 2)
        first = -ratio * a * a * chain
        second = ratio * a**3 * (2 - a * a - a * ratio) * chain**2
    assert_allclose(value.packed_derivatives(1), [[float(first)]], rtol=3e-12, atol=0)
    assert_allclose(value.packed_derivatives(2), [[float(second)]], rtol=3e-12, atol=0)


@pytest.mark.parametrize("x", [0.0, 1e-14, -1e-14, -4.0, 4.0, -10.0, 10.0])
def test_generic_expm1_over_x_taylor_primitive_through_fourth_order(x):
    mp = pytest.importorskip("mpmath")
    value = Taylor(np.array([x]), 1, 4, 0).expm1_over_x()
    with mp.workdps(70):
        for order in range(5):
            expected = mp.quad(
                lambda t, order=order: t**order * mp.exp(mp.mpf(x) * t), [0, 1]
            )
            actual = value.value if order == 0 else value.packed_derivatives(order)[0]
            assert_allclose(actual, float(expected), atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("point", [-1 / 41, 0.0, 1 / 41])
def test_fourth_order_normalizer_flat_limit_and_scaled_chain_against_high_precision(
    point,
):
    mp = pytest.importorskip("mpmath")
    chain = 1e60
    t = Taylor(np.array([point]), 1, 4)
    t.coefficients[(1,)] = np.array([chain])
    value = _log_normalizer(t)
    with mp.workdps(450):

        def log_c(v):
            return (
                mp.mpf(0) if v == 0 else mp.log(mp.erfc(-1 / (abs(v) * mp.sqrt(2))) / 2)
            )

        for degree in range(1, 5):
            expected = mp.diff(log_c, mp.mpf(point), degree) * mp.mpf(chain) ** degree
            assert_allclose(
                value.packed_derivatives(degree),
                [[float(expected)]],
                rtol=5e-12,
                atol=0,
            )
