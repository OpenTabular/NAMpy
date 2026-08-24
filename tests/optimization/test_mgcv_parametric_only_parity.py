"""Parity for models that have no smoothing parameters to select.

Upstream references: ``mgcv/R/mgcv.r::gam``, ``gam.setup``, and
``estimate.gam``.  In particular, ``estimate.gam`` supplies an empty ``lsp``
for a parametric-only model while retaining the Gaussian REML scale parameter
in the outer problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from nampy.gam import GAM
from tests.mgcv_parity_utils import (
    _make_gaussian_data,
    _make_poisson_data,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression]


def test_gaussian_parametric_only_reml_matches_mgcv() -> None:
    """REML remains a valid fit request when the formula has no smooths."""
    data = _make_gaussian_data(seed=1001, n=96)
    formula = "y ~ x0 + x1"

    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        "gaussian",
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )

    assert actual["fit"]["smoothing_params"] == []
    assert expected["fit"]["smoothing_params"] == []
    assert actual["fit"]["criterion_name"].lower() == expected["fit"][
        "criterion_name"
    ].lower()
    np.testing.assert_allclose(
        actual["fit"]["coef_full"],
        expected["fit"]["coef_full"],
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        actual["fit"]["scale"],
        expected["fit"]["scale"],
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        actual["fit"]["criterion_value"],
        expected["fit"]["criterion_value"],
        atol=2e-9,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        actual["predictions"]["response"],
        expected["predictions"]["response"],
        atol=2e-10,
        rtol=2e-10,
    )


def test_poisson_parametric_only_reml_matches_mgcv() -> None:
    """An empty smoothing search still records the requested REML criterion."""
    data = _make_poisson_data(seed=1002, n=120)
    formula = "y ~ x0 + x1"

    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        "poisson",
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )

    assert actual["fit"]["smoothing_params"] == []
    assert expected["fit"]["smoothing_params"] == []
    assert actual["fit"]["criterion_name"].lower() == expected["fit"][
        "criterion_name"
    ].lower()
    np.testing.assert_allclose(
        actual["fit"]["coef_full"],
        expected["fit"]["coef_full"],
        atol=2e-9,
        rtol=2e-9,
    )
    np.testing.assert_allclose(
        actual["fit"]["criterion_value"],
        expected["fit"]["criterion_value"],
        atol=2e-8,
        rtol=2e-9,
    )
    np.testing.assert_allclose(
        actual["predictions"]["response"],
        expected["predictions"]["response"],
        atol=2e-9,
        rtol=2e-9,
    )
