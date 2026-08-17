"""Compare near-singular Gaussian REML derivatives at mgcv trace endpoints.

This probe evaluates NAMpy's direct port of the ``gam.fit3``/``gdi1`` joint
``(log(sp), log(scale))`` derivatives at the accepted points recorded by
``mgcv::newton``.  It avoids rerunning NAMpy's slow outer optimization while
localizing the first derivative component that diverges.
"""

from __future__ import annotations

import json

import numpy as np

from nampy.gam.smoothing_selection.criteria.gaussian_dyn import (
    _gaussian_dynamic_reml_derivative_terms,
)
from nampy.gam.smoothing_selection.optimize.objectives import (
    _GaussianRemlJointObjective,
)
from tests.mgcv_parity_utils import _make_random_effect_data
from tests.optimization.test_mgcv_fixed_inner_fit_parity import (
    _run_reference_fit3_fixed_sp,
)
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _compile_optimization_state,
    _run_mgcv_outer_trace,
)


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def main() -> None:
    data = _make_random_effect_data()
    formula = 'y ~ s(f, bs="re")'
    expected = _run_mgcv_outer_trace(
        data,
        formula,
        "gaussian",
        "REML",
        "newton",
    )
    model = _compile_optimization_state(data, formula, "gaussian", "REML")
    objective = _GaussianRemlJointObjective(model, model.y_, "REML")

    rows = list(expected.get("trace", []))
    selected = [row for row in rows if 10 <= int(row["iter"]) <= 18]
    if rows and rows[-1] not in selected:
        selected.append(rows[-1])

    out = []
    for row in selected:
        log_sp = np.asarray(row["log_sp"], dtype=np.float64).ravel()
        log_scale = float(row["log_scale"])
        x = np.concatenate([log_sp, np.array([log_scale], dtype=np.float64)])
        terms = _gaussian_dynamic_reml_derivative_terms(
            model,
            model.y_,
            log_sp,
            method="REML",
        )
        fixed_scale = _run_reference_fit3_fixed_sp(
            data,
            formula,
            "gaussian",
            np.exp(log_sp),
            score_type="REML",
        )
        out.append(
            {
                "iter": int(row["iter"]),
                "x": x,
                "mgcv": {
                    "criterion": row["criterion"],
                    "gradient": row["gradient_full"],
                    "hessian": row["hessian_full"],
                    "penalized_deviance_gradient": fixed_scale["D1"],
                    "penalized_deviance_hessian": fixed_scale["D2"],
                    "fixed_unit_scale_gradient": fixed_scale["REML1"],
                    "fixed_unit_scale_hessian": fixed_scale["REML2"],
                },
                "nampy": {
                    "criterion": objective.fun(x),
                    "gradient": objective.jac(x),
                    "hessian": objective.hess(x),
                    "profile_terms": terms,
                },
            }
        )

    print(json.dumps(_serializable(out), indent=2))


if __name__ == "__main__":
    main()
