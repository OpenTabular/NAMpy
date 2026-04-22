from __future__ import annotations

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.parity import build_optimizer_trace
from tests.mgcv_parity_utils import _make_gamma_data, _make_negbin_data
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _assert_joint_negbin_trace_row_close,
    _assert_serialized_trace_matches_mgcv,
    _assert_trace_rows_close,
    _run_mgcv_outer_trace,
)

pytestmark = [pytest.mark.surface_trace, pytest.mark.surface_regression]

_JOINT_TRACE_GAPS = [
    pytest.param(
        "gamma_joint_scale_trace",
        id="gamma_joint_scale_trace",
        marks=[
            pytest.mark.status_known_gap,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "Gamma joint REML trace still exposes PIRLS-inner rows instead of "
                    "mgcv's joint log-scale outer rows."
                ),
            ),
        ],
    ),
    pytest.param(
        "negbin_joint_theta_trace_labels",
        id="negbin_joint_theta_trace_labels",
        marks=[
            pytest.mark.status_known_gap,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "Joint negbin outer trace still swaps the smoothing-parameter and "
                    "log-theta row labels on the serialized branch trace."
                ),
            ),
        ],
    ),
]


def _swap_negbin_trace_labels(row: dict) -> dict:
    out = dict(row)
    out["log_sp"] = row["log_theta"]
    out["log_theta"] = row["log_sp"]
    return out


@pytest.mark.parametrize("case_id", _JOINT_TRACE_GAPS)
def test_joint_branch_trace_rows_match_mgcv_with_correct_labels(case_id):
    """Verify that joint branch trace rows match mgcv with correct labels."""
    if case_id == "gamma_joint_scale_trace":
        data = _make_gamma_data(seed=123, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8)'
        expected = _run_mgcv_outer_trace(data, formula, "gamma", "REML", "newton")
        gam = GAM(
            family="gamma",
            formula=formula,
            optimize_smoothing=True,
            smoothing_method="REML",
        )
        gam.fit(data=data)

        actual_trace = list(getattr(gam, "_optim_trace", []) or [])
        expected_trace = list(expected["trace"])
        actual_outer = dict(getattr(gam._optim_result, "outer_info", {}) or {})
        expected_outer = expected["fit"]["outer_info"]
        _assert_trace_rows_close(actual_trace, expected_trace, atol=5e-7)
        np.testing.assert_allclose(
            np.asarray(actual_outer["score_hist"], dtype=np.float64),
            np.asarray(expected_outer["score_hist"], dtype=np.float64),
            atol=5e-7,
            rtol=0.0,
        )
        actual_serialized = build_optimizer_trace(gam)
        _assert_serialized_trace_matches_mgcv(actual_serialized, expected, atol=5e-7)
        return

    data = _make_negbin_data(seed=93, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 1.8, "estimate_theta": True}
    expected = _run_mgcv_outer_trace(
        data,
        formula,
        "negbin_est:1.8",
        "REML",
        "newton",
    )
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    actual_trace = list(getattr(gam, "_optim_trace", []) or [])
    expected_trace = [_swap_negbin_trace_labels(row) for row in expected["trace"]]
    actual_serialized = build_optimizer_trace(gam)

    _assert_trace_rows_close(actual_trace, expected_trace, atol=2e-5)
    _assert_serialized_trace_matches_mgcv(actual_serialized, expected, atol=2e-5)
    for actual_row, expected_row in zip(actual_serialized["trace"], expected["trace"]):
        _assert_joint_negbin_trace_row_close(actual_row, expected_row, atol=2e-5)

    np.testing.assert_allclose(
        float(gam._optim_result.joint_log_theta),
        float(expected["trace"][-1]["log_theta"]),
        atol=2e-5,
        rtol=0.0,
    )
