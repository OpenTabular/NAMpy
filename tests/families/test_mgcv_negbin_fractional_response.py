"""Negative-binomial deviance and starts for accepted nonnegative responses."""

import numpy as np
import pytest

from nampy.gam.families.registry import make_gam_family
from tests.families.test_mgcv_family_kernel_parity import _run_mgcv_family_kernels


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("theta", [0.4, 3.2])
def test_negbin_fractional_deviance_and_tiny_starts_match_mgcv(link, theta):
    family = make_gam_family({"name": "nb", "link": link, "theta": theta})
    y = np.array([0, 1e-14, 0.1, 0.5, 1, 3.2])
    mu = np.array([0.2, 1e-14, 0.1, 0.8, 2, 2.5])
    weights = np.array([0, 0.5, 2, 1, 3, 0.8])
    expected = _run_mgcv_family_kernels(
        "negbin",
        link,
        y=y,
        mu=mu,
        eta=family.link(mu),
        wt=weights,
        scale=1,
        ltheta=float(np.log(theta)),
    )
    # mgcv accepts these responses and retains its own fractional-deviance
    # convention. Initialization must preserve positive values below FAMILY_EPS.
    np.testing.assert_array_equal(family.validate_y(y), y)
    np.testing.assert_allclose(
        family.deviance_obs(y, mu, weights),
        expected["dev_resids"],
        rtol=2e-13,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        family.deviance(y, mu, weights),
        np.sum(expected["dev_resids"]),
        rtol=2e-13,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        family.initialize_mu(y, weights),
        expected["mustart"],
        rtol=2e-15,
        atol=0,
    )
