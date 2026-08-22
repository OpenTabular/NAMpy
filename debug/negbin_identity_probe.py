"""Probe the estimated-theta negative-binomial identity-link parity gap."""

from __future__ import annotations

import subprocess

import numpy as np

from nampy.gam.fit.selection.criteria.pirls.derivatives import (
    criterion_gradient_ml_reml_pirls_negbin_joint,
    criterion_hessian_ml_reml_pirls_negbin_joint,
)
from nampy.gam.fit.selection.criteria.pirls.value import (
    criterion_ml_reml_pirls_negbin_joint,
)
from nampy.gam.parity import build_optimizer_trace
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _make_negbin_data,
    _run_mgcv_snapshot,
)
from tests.optimization.test_mgcv_outer_optimization_parity import _run_mgcv_outer_trace


def _max_abs(actual, expected) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(actual, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64)
            )
        )
    )


def main() -> None:
    data = _make_negbin_data(seed=910, n=220, theta=1.4)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family_est = {
        "name": "negbin",
        "theta": 1.4,
        "estimate_theta": True,
        "link": "identity",
    }

    model = _fit_nampy_model(data, formula, family_est, "REML")
    actual = model.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(data, formula, family_est, "REML")

    actual_sp = np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
    expected_sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    actual_theta = float(actual["fit"]["family_theta"])
    expected_theta = float(expected["fit"]["family_theta"])

    print("actual_sp", actual_sp)
    print("expected_sp", expected_sp)
    print("actual_theta", actual_theta)
    print("expected_theta", expected_theta)
    print("actual_log_sp", np.log(actual_sp))
    print("expected_log_sp", np.log(expected_sp))
    print("actual_log_theta", np.log(actual_theta))
    print("expected_log_theta", np.log(expected_theta))
    print("full_response_max_abs", _max_abs(actual["predictions"]["response"], expected["predictions"]["response"]))
    print("full_link_max_abs", _max_abs(actual["predictions"]["link"], expected["predictions"]["link"]))
    print("actual_deviance", actual["fit"]["deviance"])
    print("expected_deviance", expected["fit"]["deviance"])
    print("actual_criterion", actual["fit"]["criterion_value"])
    print("expected_criterion", expected["fit"]["criterion_value"])
    try:
        expected_outer = _run_mgcv_outer_trace(
            data,
            formula,
            "negbin_est:1.4:identity",
            "REML",
            "newton",
        )
    except subprocess.CalledProcessError as exc:
        print("expected_outer_trace_stdout", exc.stdout)
        print("expected_outer_trace_stderr", exc.stderr)
        raise
    print("expected_outer_info", expected_outer["fit"]["outer_info"])
    print("expected_outer_trace_len", len(expected_outer["trace"]))
    for row in expected_outer["trace"]:
        print("expected_outer_trace_row", row)
    trace = build_optimizer_trace(model)
    print("trace_keys", trace.keys())
    print("trace_payload", trace)

    for label, sp, theta in [
        ("actual", actual_sp, actual_theta),
        ("expected", np.atleast_1d(expected_sp), expected_theta),
    ]:
        log_sp = np.log(np.asarray(sp, dtype=np.float64).ravel())
        log_theta = float(np.log(theta))
        val = criterion_ml_reml_pirls_negbin_joint(
            model,
            data["y"].to_numpy(dtype=np.float64),
            log_sp,
            log_theta,
            method="REML",
        )
        grad = criterion_gradient_ml_reml_pirls_negbin_joint(
            model,
            data["y"].to_numpy(dtype=np.float64),
            log_sp,
            log_theta,
            method="REML",
        )
        hess = criterion_hessian_ml_reml_pirls_negbin_joint(
            model,
            data["y"].to_numpy(dtype=np.float64),
            log_sp,
            log_theta,
            method="REML",
        )
        print(f"{label}_objective_at_point", val)
        print(f"{label}_gradient_theta_last", grad)
        print(f"{label}_hessian_theta_last", hess)

    fixed_family = {
        "name": "negbin",
        "theta": expected_theta,
        "link": "identity",
    }
    fixed_expected = _fit_nampy_model_fixed_sp(data, formula, fixed_family, expected_sp)
    fixed_expected_snapshot = fixed_expected.parity_snapshot(
        X=data,
        include_covariances=True,
    )
    print(
        "fixed_at_expected_response_max_abs",
        _max_abs(
            fixed_expected_snapshot["predictions"]["response"],
            expected["predictions"]["response"],
        ),
    )
    print(
        "fixed_at_expected_link_max_abs",
        _max_abs(
            fixed_expected_snapshot["predictions"]["link"],
            expected["predictions"]["link"],
        ),
    )
    print("fixed_at_expected_deviance", fixed_expected_snapshot["fit"]["deviance"])

    fixed_actual_mgcv = _run_mgcv_snapshot(
        data,
        formula.replace("k=8)", f"k=8, sp={actual_sp.tolist()})"),
        {
            "name": "negbin",
            "theta": actual_theta,
            "link": "identity",
        },
        "fixed",
    )
    print(
        "mgcv_fixed_at_actual_response_vs_nampy_max_abs",
        _max_abs(
            fixed_actual_mgcv["predictions"]["response"],
            actual["predictions"]["response"],
        ),
    )
    print(
        "mgcv_fixed_at_actual_link_vs_nampy_max_abs",
        _max_abs(fixed_actual_mgcv["predictions"]["link"], actual["predictions"]["link"]),
    )
    print("mgcv_fixed_at_actual_deviance", fixed_actual_mgcv["fit"]["deviance"])


if __name__ == "__main__":
    main()
