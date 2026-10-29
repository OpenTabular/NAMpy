"""Normalized BCCG probability law, tails and capability boundaries."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, stats

from nampy.distributions import BCCGLaw, DistributionPrediction, LogNormalLaw
from nampy.diagnostics import prepare_centiles


def _distribution(mu=1.0, sigma=0.4, nu=1.0):
    mu, sigma, nu = np.broadcast_arrays(
        np.atleast_1d(mu), np.atleast_1d(sigma), np.atleast_1d(nu)
    )
    return DistributionPrediction(BCCGLaw(), {"mu": mu, "sigma": sigma, "nu": nu})


def test_saved_r_probability_reference_and_support_discrepancies():
    root = Path(__file__).parents[1] / "reference_fixtures" / "bccg"
    reference = pd.read_csv(root / "probability.csv")
    dist = _distribution(reference.mu, reference.sigma, reference.nu)
    assert_allclose(
        dist.log_prob(reference.y), reference.log_prob, atol=3e-12, rtol=2e-12
    )
    assert_allclose(dist.cdf(reference.y), reference.cdf, atol=3e-14)
    assert_allclose(dist.survival(reference.y), reference.survival, atol=3e-14)
    for row, p in enumerate(reference.probability):
        assert_allclose(
            dist.quantile(p)[row, 0], reference["quantile"][row], atol=3e-12, rtol=3e-12
        )
    quirks = pd.read_csv(root / "source_quirks.csv")
    assert quirks.log_prob.iloc[0] == quirks.log_prob.iloc[1] == 0
    assert np.isneginf(_distribution().log_prob([-1])).all()
    assert np.isneginf(_distribution().log_prob(0)).all()
    assert np.isneginf(_distribution().log_cdf(0)).all()
    assert _distribution().log_survival(0)[0] == 0
    assert np.isneginf(quirks.log_cdf.iloc[2])
    assert np.isneginf(quirks.log_survival.iloc[3])
    assert np.isfinite(_distribution(sigma=1.0, nu=1.0).log_cdf(1e-200)[0])
    assert np.isfinite(_distribution(sigma=1.0, nu=-1.0).log_survival(1e200)[0])


@pytest.mark.parametrize("nu", [-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0])
def test_density_normalizes_with_response_jacobian_and_truncation(nu):
    dist = _distribution(mu=2.0, sigma=1.4, nu=nu)

    # Integrating in log-response also independently tests the dy Jacobian.
    def integrand(log_y):
        return np.exp(dist.log_prob(np.exp(log_y))[0] + log_y)

    total, error = integrate.quad(
        integrand, -700, 700, epsabs=2e-11, epsrel=2e-11, points=[np.log(2)], limit=200
    )
    assert error < 2e-9
    assert_allclose(total, 1, atol=2e-10)


def test_nu_one_is_positive_truncated_normal_not_mu_mean_or_median():
    mu, sigma = 2.0, 1.2
    dist = _distribution(mu, sigma, 1.0)
    reference = stats.truncnorm(-1 / sigma, np.inf, loc=mu, scale=mu * sigma)
    y = np.array([0.01, 0.2, 1, 2, 4, 10])
    repeated = _distribution(np.full(len(y), mu), sigma, 1.0)
    assert_allclose(repeated.log_prob(y), reference.logpdf(y), atol=2e-14)
    assert_allclose(repeated.cdf(y), reference.cdf(y), atol=2e-14)
    assert_allclose(
        dist.quantile([0.01, 0.5, 0.99])[0],
        reference.ppf([0.01, 0.5, 0.99]),
        atol=3e-13,
    )
    mean = integrate.quad(lambda y: y * np.exp(dist.log_prob(y)[0]), 0, np.inf)[0]
    assert_allclose(mean, reference.mean(), atol=2e-11)
    assert abs(mean - mu) > 0.1 and abs(dist.quantile(0.5)[0, 0] - mu) > 0.1


@pytest.mark.parametrize("nu", [-1e-12, -1e-300, 0.0, 1e-300, 1e-12])
def test_zero_power_and_small_nonzero_power_limit_remain_lognormal(nu):
    y = np.array([0.2, 0.7, 1.3, 3.0])
    mu, sigma = np.full(4, 1.2), np.full(4, 0.6)
    reference = DistributionPrediction(
        LogNormalLaw(), {"loc": np.log(mu), "scale": sigma}
    )
    dist = _distribution(mu, sigma, nu)
    for operation in [
        "log_prob",
        "cdf",
        "cdf_left",
        "log_cdf",
        "survival",
        "log_survival",
    ]:
        assert_allclose(
            getattr(dist, operation)(y),
            getattr(reference, operation)(y),
            atol=2e-11,
            rtol=2e-11,
        )
    assert_allclose(
        dist.quantile([0.01, 0.5, 0.99]),
        reference.quantile([0.01, 0.5, 0.99]),
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.parametrize("nu", [-2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0])
def test_quantile_inverse_cdf_tails_and_exact_support(nu):
    dist = _distribution(mu=1.3, sigma=1.2, nu=nu)
    probabilities = np.array([1e-20, 1e-12, 0.01, 0.2, 0.5, 0.9, 1 - 1e-12])
    quantiles = dist.quantile(probabilities)[0]
    repeated = _distribution(np.full(len(quantiles), 1.3), 1.2, nu)
    lower = probabilities <= 0.5
    assert_allclose(
        repeated.log_cdf(quantiles)[lower],
        np.log(probabilities[lower]),
        atol=3e-10,
        rtol=3e-12,
    )
    assert_allclose(
        repeated.log_survival(quantiles)[~lower],
        np.log1p(-probabilities[~lower]),
        atol=3e-10,
        rtol=3e-12,
    )
    assert np.all(np.diff(quantiles) >= 0)
    assert_array_equal(dist.quantile([0, 1]), [[0, np.inf]])
    assert dist.quantile([]).shape == (1, 0)
    assert dist.log_prob(np.inf)[0] == -np.inf
    assert_array_equal(dist.cdf([-1]), [0])
    assert_array_equal(dist.cdf(np.inf), [1])
    assert_array_equal(dist.cdf_left(0), [0])


@pytest.mark.parametrize(
    "nu,y", [(1.0, 1e-200), (-1.0, 1e200), (2.0, 1e-100), (-2.0, 1e100)]
)
def test_boundary_log_tails_against_high_precision_normal_interval(nu, y):
    mp = pytest.importorskip("mpmath")
    sigma = 1.1
    with mp.workdps(260):
        ss, nn, yy = mp.mpf(str(sigma)), mp.mpf(str(nu)), mp.mpf(str(y))
        a = 1 / (ss * abs(nn))
        delta = yy**nn / (ss * abs(nn))

        def phi(x):
            return mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi)

        mass = mp.quad(lambda s: phi(-a + delta * s), [0, 1]) * delta
        log_tail = float(mp.log(mass) - mp.log(mp.erfc(-a / mp.sqrt(2)) / 2))
    dist = _distribution(sigma=sigma, nu=nu)
    actual = dist.log_cdf(y) if nu > 0 else dist.log_survival(y)
    assert_allclose(actual, log_tail, atol=2e-12, rtol=2e-14)
    assert np.isfinite(actual).all()


def test_normal_far_tail_is_not_log_of_an_underflowed_survival():
    dist = _distribution(mu=1.0, sigma=1.0, nu=1.0)
    assert dist.survival(50)[0] == 0
    assert_allclose(
        dist.log_survival(50), stats.norm.logsf(49) - stats.norm.logcdf(1), atol=1e-12
    )


@pytest.mark.parametrize("factor", [1e-200, 1e200])
def test_response_units_preserve_probabilities_and_density_jacobian(factor):
    mu = np.array([1.0, 2.0, 3.0])
    sigma, nu, y = (
        np.array([0.4, 0.6, 1.2]),
        np.array([-1.0, 0.0, 2.0]),
        np.array([0.2, 2.1, 7.0]),
    )
    base = _distribution(mu, sigma, nu)
    scaled = _distribution(mu * factor, sigma, nu)
    assert_allclose(
        scaled.log_prob(y * factor), base.log_prob(y) - np.log(factor), atol=3e-12
    )
    for operation in ["log_cdf", "log_survival"]:
        assert_allclose(
            getattr(scaled, operation)(y * factor),
            getattr(base, operation)(y),
            atol=3e-12,
        )
    assert_allclose(
        scaled.quantile([0.1, 0.5, 0.9]) / factor,
        base.quantile([0.1, 0.5, 0.9]),
        rtol=3e-12,
    )


def test_interior_quantile_can_underflow_the_float64_outcome_range():
    sigma, nu, p = 1.2, 0.1, 1e-100
    a = 1 / (sigma * nu)
    # Density is increasing just above -a, so interval mass >= phi(a)*delta.
    # This gives an upper bound on log(y), proving the outcome is unrepresentable.
    log_delta_upper = np.log(p) + stats.norm.logcdf(a) - stats.norm.logpdf(a)
    assert np.exp(log_delta_upper) < a
    log_y_upper = (np.log(sigma * nu) + log_delta_upper) / nu
    assert log_y_upper < np.log(np.nextafter(0.0, 1.0))
    assert _distribution(sigma=sigma, nu=nu).quantile(p)[0, 0] == 0


def test_sampling_rowwise_shapes_reproducibility_and_shared_diagnostics():
    dist = _distribution(mu=[1, 2, 3], sigma=[0.3, 0.2, 0.2], nu=[0, 1, -2])
    draws = dist.sample(4000, random_state=71)
    assert draws.shape == (4000, 3) and np.all(draws > 0)
    assert_array_equal(draws, dist.sample(4000, random_state=71))
    median = dist.quantile(0.5)[:, 0]
    assert_allclose(np.mean(draws <= median, axis=0), 0.5, atol=0.025)
    probabilities = np.array([0.2, 0.5, 0.8])
    y = np.array([dist.quantile(p)[j, 0] for j, p in enumerate(probabilities)])
    residual = dist.quantile_residuals(y)
    assert_allclose(residual.residuals, stats.norm.ppf(probabilities), atol=2e-12)
    assert not residual.randomized.any()
    assert_allclose(
        prepare_centiles(dist, [0, 1, 2]).quantiles, dist.quantile([0.05, 0.5, 0.95])
    )


def test_scalar_capabilities_and_missing_rows_are_explicit():
    dist = _distribution(
        mu=[1, np.nan, 2], sigma=[0.4, np.nan, 0.6], nu=[1, np.nan, -1]
    )
    assert (
        np.isnan(dist.cdf(1)[1])
        and np.isnan(dist.sample(3, random_state=0)[:, 1]).all()
    )
    assert dist.spec.response_kind == "continuous"
    assert {"mean", "variance"} <= dist.spec.operations
    for operation in ["mean", "variance"]:
        assert np.isnan(getattr(dist, operation)()[1])
    with pytest.raises(ValueError, match="shape"):
        dist.log_prob([1, 2])


@pytest.mark.parametrize(
    "mu,sigma,nu", [(0, 0.2, 1), (-1, 0.2, 1), (1, 0, 1), (1, -1, 1), (1, 0.2, np.inf)]
)
def test_invalid_parameters_do_not_get_clipped(mu, sigma, nu):
    with pytest.raises(ValueError):
        _distribution(mu, sigma, nu)


def test_probability_support_does_not_enable_or_alias_statistical_fitting():
    from nampy.gam.families.bcg import CensoredBoxCoxGaussianFamily
    from nampy.gam.families.registry import make_gam_family

    with pytest.raises(ValueError, match="Unknown GAM family"):
        make_gam_family("bccg")
    assert isinstance(make_gam_family("bcg"), CensoredBoxCoxGaussianFamily)
