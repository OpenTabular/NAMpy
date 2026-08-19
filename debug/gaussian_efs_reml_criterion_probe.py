"""Diagnose Gaussian REML/EFS criterion parity without assuming trace identity.

Reference control flow:

- ``mgcv/R/mgcv.r::gam.outer``
- ``mgcv/R/gam.fit4.r::efsudr``
- ``mgcv/R/gam.fit3.r::gam.fit3``

Run from the repository root with::

    python3 debug/gaussian_efs_reml_criterion_probe.py
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria.gaussian_dyn import (
    criterion_ml_reml_gaussian_dynamic_joint,
)
from nampy.gam.fit.selection.criteria.ml_reml import criterion_ml_reml
from tests.mgcv_parity_utils import _make_gaussian_data
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _run_mgcv_outer_trace,
)


FORMULA = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'


def _run_case(data, label: str) -> dict:
    expected = _run_mgcv_outer_trace(
        data,
        FORMULA,
        "gaussian",
        "REML",
        "efs",
    )
    gam = GAM(
        family="gaussian",
        formula=FORMULA,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="efs",
    ).fit(data=data)

    log_sp = np.log(np.asarray(gam.smoothing_params, dtype=np.float64))
    fit_scale = float(gam.gam_result_.fit_core_solution.fit_result.scale)
    expected_last = expected["trace"][-1]
    expected_log_scale = float(expected_last["log_scale"])
    expected_criterion = float(expected_last["criterion"])
    profiled = float(criterion_ml_reml(gam, gam.y_, log_sp, method="REML"))
    joint_at_nampy_scale = float(
        criterion_ml_reml_gaussian_dynamic_joint(
            gam,
            gam.y_,
            log_sp,
            np.log(fit_scale),
            method="REML",
        )
    )
    joint_at_mgcv_scale = float(
        criterion_ml_reml_gaussian_dynamic_joint(
            gam,
            gam.y_,
            log_sp,
            expected_log_scale,
            method="REML",
        )
    )

    return {
        "label": label,
        "nampy_log_sp": log_sp.tolist(),
        "mgcv_log_sp": list(np.asarray(expected_last["log_sp"], dtype=float)),
        "log_sp_max_abs_delta": float(
            np.max(
                np.abs(
                    log_sp
                    - np.asarray(expected_last["log_sp"], dtype=np.float64)
                )
            )
        ),
        "nampy_log_scale": float(np.log(fit_scale)),
        "mgcv_log_scale": expected_log_scale,
        "log_scale_delta": float(np.log(fit_scale) - expected_log_scale),
        "mgcv_efs_criterion": expected_criterion,
        "nampy_efs_stored_criterion": float(gam.smoothing_score_),
        "nampy_profiled_reml": profiled,
        "nampy_joint_reml_at_nampy_scale": joint_at_nampy_scale,
        "nampy_joint_reml_at_mgcv_scale": joint_at_mgcv_scale,
        "stored_minus_mgcv": float(gam.smoothing_score_ - expected_criterion),
        "joint_at_nampy_scale_minus_mgcv": float(
            joint_at_nampy_scale - expected_criterion
        ),
        "joint_at_mgcv_scale_minus_mgcv": float(
            joint_at_mgcv_scale - expected_criterion
        ),
    }


def main() -> None:
    data = _make_gaussian_data(seed=127, n=140)
    rng = np.random.default_rng(90210)
    permuted = data.iloc[rng.permutation(len(data))].reset_index(drop=True)
    payload = [
        _run_case(data, "original"),
        _run_case(permuted, "row_permuted"),
    ]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
