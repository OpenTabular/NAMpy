"""Finite BCCG moments, mathematical divergence and numerical boundaries."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from nampy.distributions import (
    AtomMixtureLaw,
    BCCGLaw,
    DistributionPrediction,
    LogNormalLaw,
)


def distribution(mu=1.0, sigma=0.4, nu=1.0):
    mu, sigma, nu = np.broadcast_arrays(
        np.atleast_1d(mu), np.atleast_1d(sigma), np.atleast_1d(nu)
    )
    return DistributionPrediction(BCCGLaw(), {"mu": mu, "sigma": sigma, "nu": nu})


@pytest.mark.parametrize(
    "sigma,nu",
    [
        (0.4, 1.0),
        (1.2, 1.0),
        (0.7, 0.2),
        (2.0, 3.0),
        (0.4, -3.0),
        (2.0, -2.01),
        (0.3, -10.0),
        (0.8, -2.00000001),
        (0.8, np.nextafter(-2.0, -np.inf)),
    ],
)
def test_moments_match_independent_parabolic_cylinder_identity(sigma, nu):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(80):
        s, n = mp.mpf(sigma), mp.mpf(nu)
        t = abs(s * n)
        a = 1 / t
        c = mp.erfc(-a / mp.sqrt(2)) / 2

        def moment(k):
            r = k / n
            return (
                t**r
                * mp.gamma(r + 1)
                * mp.exp(-a * a / 4)
                * mp.pcfd(-r - 1, -a)
                / (mp.sqrt(2 * mp.pi) * c)
            )

        mean = moment(1)
        variance = moment(2) - mean * mean
    dist = distribution(sigma=sigma, nu=nu)
    assert_allclose(dist.mean(), float(mean), rtol=3e-10)
    assert_allclose(dist.variance(), float(variance), rtol=3e-10)


@pytest.mark.parametrize("sigma", [0.03, 0.4, 2.0, 1e-5, 1e-8])
def test_nu_one_moments_equal_positive_truncated_normal_including_small_variance(sigma):
    reference = stats.truncnorm(-1 / sigma, np.inf, loc=2, scale=2 * sigma)
    dist = distribution(mu=2.0, sigma=sigma)
    assert_allclose(dist.mean(), reference.mean(), rtol=2e-11)
    assert_allclose(dist.variance(), reference.var(), rtol=3e-9)


@pytest.mark.parametrize("nu", [1e-6, 1e-12, 1e-100])
def test_positive_zero_power_limit_matches_lognormal_moments(nu):
    dist = distribution(mu=2.0, sigma=0.4, nu=nu)
    reference = DistributionPrediction(
        LogNormalLaw(), {"loc": [np.log(2)], "scale": [0.4]}
    )
    assert_allclose(dist.mean(), reference.mean(), rtol=2e-6)
    assert_allclose(dist.variance(), reference.variance(), rtol=4e-6)


def test_zero_power_exact_moments_and_heavy_tail_domains_are_rowwise():
    nu = np.array([0, -0.001, -0.5, -1.0, -1.5, -2.0, -3.0, np.nan])
    mu, sigma = np.ones(8), np.full(8, 0.4)
    mu[-1] = sigma[-1] = np.nan
    dist = distribution(mu, sigma, nu)
    means, variances = dist.mean(), dist.variance()
    assert_allclose(means[0], np.exp(0.4**2 / 2))
    assert np.isinf(means[1:4]).all() and np.isfinite(means[4:7]).all()
    assert np.isnan(variances[1:4]).all() and np.isinf(variances[4:6]).all()
    assert variances[6] > 0 and np.isnan(means[-1]) and np.isnan(variances[-1])


@pytest.mark.parametrize("nu", [0.0, 0.5, -3.0])
def test_moments_scale_with_response_units_without_intermediate_overflow(nu):
    base = distribution(sigma=0.4, nu=nu)
    scaled = distribution(mu=[1e-150, 1e150], sigma=0.4, nu=nu)
    assert_allclose(scaled.mean() / [1e-150, 1e150], base.mean()[0], rtol=2e-10)
    assert_allclose(scaled.variance() / [1e-300, 1e300], base.variance()[0], rtol=3e-10)


def test_numerical_failure_is_explicit_and_does_not_substitute_moments(monkeypatch):
    from nampy.distributions import _bccg_moments

    monkeypatch.setattr(_bccg_moments, "quad", lambda *args, **kwargs: (1.0, 1.0))
    with pytest.raises(NotImplementedError, match="relative accuracy"):
        distribution().mean()
    assert_array_equal(distribution(nu=-0.5).mean(), [np.inf])


@pytest.mark.parametrize("sigma,nu", [(1e300, 1.0), (1e-300, -3.0), (1e-12, -3.)])
def test_unsupported_numerical_moment_ranges_raise(sigma, nu):
    with pytest.raises(NotImplementedError, match="range"):
        distribution(sigma=sigma, nu=nu).variance()


def test_small_variance_with_large_mu_is_evaluated_without_intermediate_underflow():
    assert_allclose(
        distribution(mu=1e300, sigma=1e-300, nu=1.0).variance(), 1.0, rtol=2e-12
    )


def test_mean_remains_accurate_immediately_below_its_existence_boundary():
    mp = pytest.importorskip("mpmath")
    nu, sigma = -1.00000001, 0.8
    with mp.workdps(80):
        n, s = mp.mpf(nu), mp.mpf(sigma)
        t, r = abs(s * n), 1 / n
        a = 1 / t
        reference = (
            t**r
            * mp.gamma(r + 1)
            * mp.exp(-a * a / 4)
            * mp.pcfd(-r - 1, -a)
            / (mp.sqrt(2 * mp.pi) * mp.erfc(-a / mp.sqrt(2)) / 2)
        )
    assert_allclose(
        distribution(sigma=sigma, nu=nu).mean(), float(reference), rtol=3e-10
    )


def test_generic_atom_mixtures_inherit_moments_and_skip_an_inactive_divergent_base():
    law = AtomMixtureLaw(BCCGLaw(), (0.0,), ("p0",))
    mixed = DistributionPrediction(
        law,
        {"mu": [1.0, 1.0], "sigma": [0.4, 0.4], "nu": [-3.0, -0.5], "p0": [0.2, 1.0]},
    )
    base = distribution(nu=-3.0)
    assert_allclose(mixed.mean(), [0.8 * base.mean()[0], 0.0])
    assert_allclose(
        mixed.variance(), [0.8 * base.variance()[0] + 0.16 * base.mean()[0] ** 2, 0.0]
    )
