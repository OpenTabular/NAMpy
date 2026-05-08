from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nampy.gam.families.family_base import BaseFamily
from nampy.gam.families.negbin import NegativeBinomialLogFamily
from nampy.gam.families.registry import make_gam_family

pytestmark = [
    pytest.mark.surface_derivatives,
    pytest.mark.surface_regression,
    pytest.mark.family_negbin,
]


def test_negbin_inherits_base_higher_order_link_derivatives():
    """Verify that negative-binomial inherits base higher order link derivatives."""
    family = NegativeBinomialLogFamily(theta=1.7)
    assert family.d2link.__func__ is BaseFamily.d2link
    assert family.d3link.__func__ is BaseFamily.d3link
    assert family.d4link.__func__ is BaseFamily.d4link

    eta = np.linspace(-2.0, 1.5, 25, dtype=np.float64)
    mu = np.exp(eta)

    assert_allclose(family.d2link(mu), -1.0 / (mu**2), rtol=1e-12, atol=1e-12)
    assert_allclose(family.d3link(mu), 2.0 / (mu**3), rtol=1e-12, atol=1e-12)
    assert_allclose(family.d4link(mu), -6.0 / (mu**4), rtol=1e-12, atol=1e-12)

    for order in range(1, 5):
        assert_allclose(
            family.inverse_link_derivatives(eta, order=order),
            mu,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("link", ["identity", "sqrt"])
def test_negbin_nonlog_link_working_weight_derivatives_match_finite_difference(link):
    """Verify that non-log negative-binomial link derivatives stay exact."""
    family = NegativeBinomialLogFamily(theta=1.7, link=link)
    eta = np.linspace(0.5, 1.8, 25, dtype=np.float64)
    eps = 1e-6

    def _fisher_weight(eta_val):
        eta_val = np.asarray(eta_val, dtype=np.float64)
        mu = family.inverse_link(eta_val)
        return family.mu_eta(eta_val) ** 2 / family.variance(mu)

    fd1 = (_fisher_weight(eta + eps) - _fisher_weight(eta - eps)) / (2.0 * eps)
    fd2 = (
        _fisher_weight(eta + eps)
        - 2.0 * _fisher_weight(eta)
        + _fisher_weight(eta - eps)
    ) / (eps**2)

    assert_allclose(
        family.working_weight_derivative_eta(eta),
        fd1,
        rtol=1e-5,
        atol=1e-5,
    )
    assert_allclose(
        family.working_weight_second_derivative_eta(eta),
        fd2,
        rtol=2e-3,
        atol=2e-3,
    )


def test_negbin_registry_preserves_requested_link():
    """Verify that family registry preserves mgcv negative-binomial links."""
    family = make_gam_family({"name": "negbin", "theta": 1.5, "link": "sqrt"})
    eta = np.array([0.5, 1.0], dtype=np.float64)

    assert family.link_name == "sqrt"
    assert_allclose(family.inverse_link(eta), eta**2, rtol=1e-12, atol=1e-12)


def test_negbin_registry_distinguishes_mgcv_negbin_and_nb_theta_semantics():
    """mgcv::negbin requires theta; mgcv::nb estimates only for NULL/zero/negative theta."""
    with pytest.raises(ValueError, match="requires explicit theta"):
        make_gam_family("negbin")
    with pytest.raises(ValueError, match="requires explicit theta"):
        make_gam_family({"name": "negbin"})

    fixed = make_gam_family({"name": "nb", "theta": 2.5})
    assert fixed.theta == 2.5
    assert fixed.estimate_theta is False

    estimated_default = make_gam_family("nb")
    assert estimated_default.theta == 1.0
    assert estimated_default.estimate_theta is True

    estimated_initial = make_gam_family(("nb", -3.0))
    assert estimated_initial.theta == 3.0
    assert estimated_initial.estimate_theta is True
