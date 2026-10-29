"""Scientific identities and numerical boundaries of the additional laws."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import poisson

from nampy.distributions import (
    BetaLaw,
    DistributionPrediction,
    NegativeBinomialLaw,
    StudentTLaw,
)
from nampy.gam.families.gamlss.distributional import (
    BetaLocationPrecision,
    NegativeBinomialLocationDispersion,
    StudentLocationScale,
)
from nampy.gam.predict.distribution_adapter import distribution_prediction


def test_student_cauchy_identity_and_undefined_moments():
    prediction = DistributionPrediction(
        StudentTLaw(),
        {"df": [1.0, 1.5, 2.0, 8.0], "loc": [2.0] * 4, "scale": [3.0] * 4},
    )
    y = np.array([5.0, 2.0, 2.0, 2.0])
    assert prediction.cdf(y)[0] == pytest.approx(0.75)
    assert prediction.log_prob(y)[0] == pytest.approx(-np.log(6 * np.pi))
    assert_allclose(prediction.mean(), [np.nan, 2.0, 2.0, 2.0])
    assert_allclose(prediction.variance(), [np.nan, np.inf, np.inf, 12.0])
    assert_allclose(prediction.quantile(0.5), 2.0)


def test_beta_uniform_and_polynomial_identities_and_support():
    uniform = DistributionPrediction(BetaLaw(), {"alpha": [1.0], "beta": [1.0]})
    assert uniform.log_prob(0.3)[0] == pytest.approx(0.0)
    assert uniform.cdf(0.3)[0] == pytest.approx(0.3)
    assert uniform.mean()[0] == pytest.approx(0.5)
    assert uniform.variance()[0] == pytest.approx(1 / 12)
    beta = DistributionPrediction(BetaLaw(), {"alpha": [2.0], "beta": [3.0]})
    y = 0.4
    assert beta.log_prob(y)[0] == pytest.approx(np.log(12 * y * (1 - y) ** 2))
    assert beta.cdf(y)[0] == pytest.approx(6 * y**2 - 8 * y**3 + 3 * y**4)
    assert_array_equal(beta.quantile([0.0, 1.0]), [[0.0, 1.0]])
    assert beta.cdf(-1)[0] == 0
    assert beta.cdf(2)[0] == 1
    assert np.isneginf(beta.log_prob(-1)[0])


def test_negative_binomial_geometric_mass_left_limits_and_quantiles():
    prediction = DistributionPrediction(
        NegativeBinomialLaw(), {"mean": [2.0, 2.0, 2.0, 2.0], "dispersion": [1.0] * 4}
    )
    y = np.array([0.0, 1.0, 1.2, -1.0])
    assert_allclose(prediction.cdf(y), [1 / 3, 5 / 9, 5 / 9, 0])
    assert_allclose(prediction.cdf_left(y), [0, 1 / 3, 5 / 9, 0])
    assert_allclose(prediction.log_prob(y)[:2], np.log([1 / 3, 2 / 9]))
    assert np.isneginf(prediction.log_prob(y)[2:]).all()
    assert_allclose(prediction.mean(), 2)
    assert_allclose(prediction.variance(), 6)
    assert_array_equal(prediction.quantile(0), np.zeros((4, 1)))
    for p in [0.01, 0.3, 0.8, 0.95]:
        q = prediction.quantile(p)[:, 0]
        assert np.all(prediction.cdf(q) >= p)
        assert np.all(prediction.cdf_left(q) <= p)


@pytest.mark.parametrize(
    "law,params,low,high",
    [
        (StudentTLaw(), {"df": [0.7], "loc": [1.0], "scale": [0.5]}, -np.inf, np.inf),
        (BetaLaw(), {"alpha": [0.6], "beta": [2.4]}, 0.0, 1.0),
    ],
)
def test_new_continuous_densities_normalize(law, params, low, high):
    prediction = DistributionPrediction(law, params)
    mass, _ = quad(lambda y: np.exp(prediction.log_prob(y)[0]), low, high)
    assert mass == pytest.approx(1.0, abs=2e-8)


@pytest.mark.parametrize(
    "law,params",
    [
        (StudentTLaw(), {"df": [8.0], "loc": [2.0], "scale": [0.5]}),
        (BetaLaw(), {"alpha": [0.6], "beta": [2.4]}),
        (NegativeBinomialLaw(), {"mean": [4.0], "dispersion": [0.2]}),
    ],
)
def test_new_law_draws_moments_and_quantile_inversion(law, params):
    prediction = DistributionPrediction(law, params)
    draws = prediction.sample(30000, random_state=19)
    assert_array_equal(draws, prediction.sample(30000, random_state=19))
    assert_allclose(draws.mean(axis=0), prediction.mean(), atol=0.025)
    assert_allclose(draws.var(axis=0), prediction.variance(), rtol=0.04)
    for p in [0.01, 0.5, 0.99]:
        q = prediction.quantile(p)[:, 0]
        if prediction.spec.response_kind == "continuous":
            assert_allclose(prediction.cdf(q), p, atol=1e-12)
        else:
            assert (prediction.cdf_left(q) <= p).all()
            assert (prediction.cdf(q) >= p).all()


def test_nb_poisson_limit_is_not_an_automatic_poisson_substitution():
    prediction = DistributionPrediction(
        NegativeBinomialLaw(),
        {"mean": np.full(20, 4.0), "dispersion": np.full(20, 1e-6)},
    )
    counts = np.arange(20)
    actual = prediction.log_prob(counts)
    assert_allclose(actual, poisson.logpmf(counts, 4), atol=1.2e-4)
    assert not np.array_equal(actual, poisson.logpmf(counts, 4))
    assert np.exp(actual).sum() == pytest.approx(1.0, abs=2e-7)
    with pytest.raises(NotImplementedError, match="representable"):
        DistributionPrediction(
            NegativeBinomialLaw(), {"mean": [1.0], "dispersion": [1e-20]}
        )


@pytest.mark.parametrize(
    "family,params",
    [
        (StudentLocationScale(df=11.0), [[0.3, 0.7], [1.0, 2.0], [np.nan, np.nan]]),
        (BetaLocationPrecision(), [[0.3, 4.0], [0.8, 2.0], [np.nan, np.nan]]),
        (
            NegativeBinomialLocationDispersion(),
            [[0.3, 0.7], [1.0, 2.0], [np.nan, np.nan]],
        ),
    ],
)
def test_statistical_parameter_mapping_and_missing_rows(family, params):
    distribution = distribution_prediction(family, params)
    y = (
        np.array([0.2, 0.8])
        if isinstance(family, BetaLocationPrecision)
        else np.array([0.0, 2.0])
    )
    assert_allclose(
        distribution.log_prob(np.r_[y, np.nan])[:2],
        family.logpdf_from_parameters(y, np.array(params)[:2]),
        atol=1e-13,
    )
    assert np.isnan(distribution.mean()[2])
    if isinstance(family, StudentLocationScale):
        assert_allclose(distribution.parameters["df"][:2], 11.0)


@pytest.mark.parametrize(
    "family,params",
    [
        (BetaLocationPrecision(), [[0.0, 1.0]]),
        (BetaLocationPrecision(), [[1.0, 1.0]]),
        (BetaLocationPrecision(), [[0.5, -1.0]]),
        (StudentLocationScale(), [[0.0, -1.0]]),
        (NegativeBinomialLocationDispersion(), [[-1.0, 1.0]]),
    ],
)
def test_adapter_domains_are_not_silently_repaired(family, params):
    with pytest.raises(ValueError):
        distribution_prediction(family, params)
