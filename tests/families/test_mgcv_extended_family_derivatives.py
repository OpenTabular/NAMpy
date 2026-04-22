from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nampy.gam import GAM
from nampy.gam.families.family_base import BaseFamily
from nampy.gam.families.negbin import NegativeBinomialLogFamily
from nampy.gam.smoothing_selection.criteria.pirls_deriv import (
    criterion_gradient_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_exact,
)
from tests.mgcv_parity_utils import _make_negbin_data

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


def test_negbin_exact_pirls_derivative_entrypoint_runs_with_inherited_link_derivatives():
    """
    Verify that negative-binomial exact PIRLS derivative entrypoint runs with inherited
    link derivatives.
    """
    data = _make_negbin_data(seed=77, n=120, theta=1.3)
    gam = GAM(
        family={"name": "negbin", "theta": 1.3},
        formula="y ~ s(x0, k=8)",
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)

    log_sp = np.log(np.asarray(gam.smoothing_params, dtype=np.float64))
    grad = criterion_gradient_ml_reml_pirls_exact(gam, gam.y_, log_sp, "REML")
    hess = criterion_hessian_ml_reml_pirls_exact(gam, gam.y_, log_sp, "REML")

    assert grad.shape == (1,)
    assert hess.shape == (1, 1)
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))
