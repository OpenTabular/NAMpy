"""Shared positive-response laws preserve semantics across numerical backends."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import norm

from nampy import GAMLSS
from nampy.diagnostics import quantile_residuals
from nampy.distributions import DistributionPrediction, LogNormalLaw, WeibullLaw
from nampy.gam.families.gamlss.positive import (
    LogNormalLocationScale,
    WeibullLocationShape,
)
from nampy.gam.predict.distribution_adapter import (
    distribution_prediction as gam_distribution,
)


def test_lognormal_log_transform_jacobian_and_tail_operations():
    loc, scale = np.array([-1.0, 0.3]), np.array([0.4, 0.7])
    distribution = DistributionPrediction(LogNormalLaw(), {"loc": loc, "scale": scale})
    y = np.array([0.5, 2.0])
    assert_allclose(
        distribution.log_prob(y), norm.logpdf(np.log(y), loc, scale) - np.log(y)
    )
    assert_allclose(distribution.mean(), np.exp(loc + scale**2 / 2))
    assert_allclose(
        distribution.variance(), np.exp(2 * loc + scale**2) * np.expm1(scale**2)
    )
    assert_allclose(distribution.quantile(0.5)[:, 0], np.exp(loc))
    assert_array_equal(distribution.cdf(0), [0.0, 0.0])
    assert_array_equal(distribution.survival(-1), [1.0, 1.0])
    assert np.isneginf(distribution.log_prob(0)).all()
    assert_array_equal(distribution.quantile(0), [[0.0], [0.0]])
    assert np.isposinf(distribution.quantile(1)).all()
    assert_allclose(
        quantile_residuals(distribution, y).residuals, (np.log(y) - loc) / scale
    )
    # A large log-location must not overflow CDF/log-density through exp(loc).
    extreme = DistributionPrediction(LogNormalLaw(), {"loc": [800.0], "scale": [0.3]})
    assert np.isfinite(extreme.log_prob([1.0])).all()
    assert np.isfinite(extreme.log_cdf([1.0])).all()
    tiny_mean = DistributionPrediction(
        LogNormalLaw(), {"loc": [-1600.0], "scale": [40.0]}
    )
    assert_allclose(tiny_mean.variance(), [1.0])
    assert_allclose(
        LogNormalLocationScale().variance_from_parameters([[-1600.0, 40.0]]), [1.0]
    )


def test_weibull_exponential_limit_and_response_mean():
    distribution = DistributionPrediction(
        WeibullLaw(), {"scale": [2.0, 3.0], "shape": [1.0, 1.0]}
    )
    y = np.array([1.0, 2.0])
    assert_allclose(distribution.log_prob(y), -np.log([2.0, 3.0]) - y / [2.0, 3.0])
    assert_allclose(distribution.survival(y), np.exp(-y / [2.0, 3.0]))
    assert_allclose(distribution.mean(), [2.0, 3.0])
    assert_allclose(distribution.variance(), [4.0, 9.0])
    assert_allclose(distribution.quantile(0.5)[:, 0], np.array([2.0, 3.0]) * np.log(2))
    assert_array_equal(distribution.cdf_left(y), distribution.cdf(y))


@pytest.mark.parametrize(
    "law,parameters",
    [
        (LogNormalLaw(), {"loc": [-0.4], "scale": [0.5]}),
        (WeibullLaw(), {"scale": [1.4], "shape": [1.7]}),
    ],
)
def test_positive_law_normalization_draws_and_quantile_inverse(law, parameters):
    distribution = DistributionPrediction(law, parameters)
    mass, _ = quad(lambda y: np.exp(distribution.log_prob(y)[0]), 0, np.inf)
    assert mass == pytest.approx(1.0, abs=1e-9)
    draws = distribution.sample(40000, random_state=31)
    assert_array_equal(draws, distribution.sample(40000, random_state=31))
    assert_allclose(draws.mean(axis=0), distribution.mean(), rtol=0.02)
    assert_allclose(draws.var(axis=0), distribution.variance(), rtol=0.03)
    for probability in [0.01, 0.5, 0.99]:
        assert_allclose(
            distribution.cdf(distribution.quantile(probability)[:, 0]),
            probability,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "name,family",
    [("lognormal", LogNormalLocationScale), ("weibull", WeibullLocationShape)],
)
def test_probability_adapter_matches_neural_loss_and_respects_missing_rows(
    name, family
):
    torch = pytest.importorskip("torch")
    from nampy.neural.families import LogNormalDistribution, WeibullDistribution
    from nampy.neural.families.distribution_adapter import distribution_prediction

    neural_family = (
        LogNormalDistribution() if name == "lognormal" else WeibullDistribution()
    )
    raw = torch.tensor([[-0.4, 0.2], [0.6, -0.2]], requires_grad=True)
    parameters = neural_family(raw).detach().numpy()
    neural = distribution_prediction(neural_family, parameters)
    statistical = gam_distribution(family(), parameters)
    y = torch.tensor([0.6, 1.4])
    assert_allclose(
        neural.log_prob(y.detach().numpy()),
        statistical.log_prob(y.detach().numpy()),
        atol=1e-13,
    )
    assert_allclose(
        neural.quantile([0.1, 0.9]), statistical.quantile([0.1, 0.9]), atol=1e-13
    )
    loss = neural_family.compute_loss_per_sample(raw, y)
    assert_allclose(
        neural.log_prob(y.detach().numpy()), -loss.detach().numpy(), atol=2e-6
    )
    loss.sum().backward()
    assert torch.isfinite(raw.grad).all()
    missing = gam_distribution(family(), np.vstack([parameters, [np.nan, np.nan]]))
    assert np.isnan(missing.mean()[-1])
    assert GAMLSS(family=name)._resolved_family().name == name + "_lss"
