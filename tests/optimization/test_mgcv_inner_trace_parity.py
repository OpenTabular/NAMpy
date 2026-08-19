from __future__ import annotations

import numpy as np
import pytest

from nampy.gam import GAM
from tests.mgcv_parity_utils import _make_negbin_data
from tests.optimization._trace_parity_helpers import (
    _fit_nampy_negbin_inner_trace,
    _make_poisson_data,
    _run_mgcv_negbin_inner_trace,
)


def test_negbin_estimated_theta_fixed_sp_inner_trace_is_exposed():
    """Verify that negative-binomial estimated theta fixed sp inner trace is exposed."""
    data = _make_negbin_data(seed=2024, n=240, theta=1.0)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=1.0)'
    family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

    actual_rows, model = _fit_nampy_negbin_inner_trace(data, formula, family)
    expected = _run_mgcv_negbin_inner_trace(data, formula, family)

    a_theta = np.asarray([row["log_theta"] for row in actual_rows], dtype=np.float64)
    e_theta = np.asarray(
        [row["log_theta"] for row in expected["inner_trace"]], dtype=np.float64
    )

    assert a_theta.size >= 1
    assert e_theta.size >= 1
    assert e_theta[0] == pytest.approx(np.log(2.0), abs=1e-12)
    e_updates = e_theta[1:]
    assert a_theta.shape == e_updates.shape
    assert np.isfinite(a_theta).all()
    assert np.isfinite(e_theta).all()
    np.testing.assert_allclose(a_theta, e_updates, atol=1e-8, rtol=0.0)
    assert np.isfinite(float(model.family.theta))
    assert np.isfinite(float(expected["fit"]["family_theta"]))
    assert float(model.family.theta) == pytest.approx(
        float(expected["fit"]["family_theta"]), abs=1e-8
    )


def test_pirls_fixed_sp_reml_inner_trace_populates_optim_trace():
    """Verify that PIRLS fixed sp REML inner trace populates optim trace."""
    data = _make_poisson_data(seed=246, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8, sp=1.0)'
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    inner_trace = list(getattr(gam, "_pirls_last_inner_trace_", []) or [])
    optim_trace = list(getattr(gam, "_optim_trace", []) or [])

    assert len(inner_trace) >= 1
    assert len(optim_trace) == len(inner_trace)
    assert all(
        bool(row.get("rank_info", {}).get("pirls_inner", False)) for row in optim_trace
    )
    np.testing.assert_allclose(
        np.asarray([row["criterion"] for row in optim_trace], dtype=np.float64),
        np.asarray(
            [row["penalized_deviance_conv"] for row in inner_trace], dtype=np.float64
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert all(row.get("gradient", None) is not None for row in optim_trace)
