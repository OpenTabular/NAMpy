from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import gammaln

from nampy.gam.families.binomial import BinomialCauchitFamily
from nampy.gam.families.gamma import (
    GammaIdentityFamily,
    GammaInverseFamily,
    GammaLogFamily,
)
from nampy.gam.families.registry import make_gam_family

pytestmark = [
    pytest.mark.surface_derivatives,
    pytest.mark.surface_regression,
    pytest.mark.family_binomial,
]


def test_binomial_cauchit_registry_and_weighted_initialization():
    """Verify that binomial cauchit is available and uses mgcv weighted starts."""
    family = make_gam_family({"name": "binomial", "link": "cauchit"})
    assert isinstance(family, BinomialCauchitFamily)

    y = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    weights = np.array([1.0, 2.0, 5.0], dtype=np.float64)
    expected = (weights * y + 0.5) / (weights + 1.0)
    assert_allclose(
        family.initialize_mu(y, weights=weights),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.family_binomial
def test_grouped_binomial_loglik_aic_include_mgcv_binomial_coefficient():
    """Grouped-binomial likelihood mirrors stats::binomial()$aic via dbinom."""
    family = make_gam_family("binomial")
    n_trials = np.array([5.0, 8.0, 12.0], dtype=np.float64)
    successes = np.array([1.0, 3.0, 9.0], dtype=np.float64)
    y = successes / n_trials
    mu = np.array([0.25, 0.40, 0.70], dtype=np.float64)

    expected_loglik_obs = (
        gammaln(n_trials + 1.0)
        - gammaln(successes + 1.0)
        - gammaln(n_trials - successes + 1.0)
        + successes * np.log(mu)
        + (n_trials - successes) * np.log1p(-mu)
    )

    assert_allclose(
        family.loglik_obs(y, mu, n=n_trials),
        expected_loglik_obs,
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(
        family.aic(y, mu, n=n_trials),
        -2.0 * np.sum(expected_loglik_obs),
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(
        family.aic(y, mu, n=n_trials, weights=n_trials),
        -2.0 * np.sum(expected_loglik_obs),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.family_gamma
def test_gamma_registry_defaults_to_mgcv_inverse_link():
    """mgcv::Gamma() defaults to inverse link; explicit links remain honored."""
    assert isinstance(make_gam_family("gamma"), GammaInverseFamily)
    assert isinstance(make_gam_family({"name": "gamma"}), GammaInverseFamily)
    assert isinstance(
        make_gam_family({"name": "gamma", "link": "inverse"}),
        GammaInverseFamily,
    )
    assert isinstance(make_gam_family({"name": "gamma", "link": "log"}), GammaLogFamily)
    assert isinstance(
        make_gam_family({"name": "gamma", "link": "identity"}),
        GammaIdentityFamily,
    )


@pytest.mark.parametrize("link", ["probit", "cloglog", "cauchit"])
def test_binomial_nonlogit_working_weight_second_derivative_fd(link):
    """Verify non-logit binomial W'' implementations match finite differences."""
    family = make_gam_family({"name": "binomial", "link": link})
    eta = np.linspace(-1.0, 1.0, 11, dtype=np.float64)
    eps = 1e-5

    def _fisher_weight(eta_val):
        eta_val = np.asarray(eta_val, dtype=np.float64)
        mu = family.inverse_link(eta_val)
        return family.mu_eta(eta_val) ** 2 / family.variance(mu)

    fd2 = (
        _fisher_weight(eta + eps)
        - 2.0 * _fisher_weight(eta)
        + _fisher_weight(eta - eps)
    ) / (eps**2)

    assert_allclose(
        family.working_weight_second_derivative_eta(eta),
        fd2,
        rtol=1e-4,
        atol=1e-4,
    )
