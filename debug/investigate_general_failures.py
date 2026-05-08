"""Focused diagnostics for current general-family mgcv parity failures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nampy.gam.diagnostics.summary import summary_text
from nampy.gam.smoothing_selection import sp_vcov
from nampy.gam.smoothing_selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from tests.diagnostics.test_mgcv_general_family_secondary_diagnostics_parity import (
    _gaulss_two_smooth_data,
)
from tests.families.test_general_family_mgcv_parity import (
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _general_newdata,
    _gevlss_data,
    _gevlss_tensor_data,
)
from tests.mgcv_parity_utils import (
    _run_mgcv_gam_vcomp,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
)


def _max_abs(a, b) -> float:
    return float(
        np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))
    )


def _show(name, value) -> None:
    print(f"{name}: {value}")


def gevlss_cr_endpoint() -> None:
    print("\n== gevlss_cr endpoint ==")
    formula = ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"]
    data = _gevlss_data()
    expected = _run_mgcv_snapshot(data, formula, "gevlss", "ML")
    gam = _fit_nampy_model(data, formula, "gevlss", "ML")
    _show("actual log_sp", np.log(gam.smoothing_params))
    _show("mgcv log_sp", expected["fit"]["log_smoothing_params"])
    _show("actual score", gam.smoothing_score_)
    _show("mgcv score", expected["fit"]["criterion_value"])
    _show("actual coef head", gam.fit_core_solution_.fit_result.coef_full[:8])
    _show("mgcv coef head", np.asarray(expected["fit"]["coef_full"])[:8])
    newdata = _general_newdata(data)
    for pred_type in ("link", "response", "terms", "lpmatrix"):
        got = gam.predict(newdata, type=pred_type)
        r = _run_mgcv_predict_on_newdata(
            data,
            newdata,
            formula,
            family="gevlss",
            method="ML",
            type=pred_type,
            return_se=False,
        )["pred"]
        _show(f"{pred_type} max_abs", _max_abs(got, r))


def gevlss_t2_fixed_derivatives() -> None:
    print("\n== gevlss_t2_full_true fixed derivatives ==")
    formula = ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1", "~ 1"]
    data = _gevlss_tensor_data()
    expected = _run_mgcv_snapshot(data, formula, "gevlss", "ML")
    sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    log_sp = np.log(sp)
    gam = _fit_nampy_model_fixed_sp(data, formula, "gevlss", sp)
    _show("criterion abs diff", abs(float(criterion_value(gam, gam.y_, log_sp, method="ml")) - float(expected["fit"]["criterion_value"])))
    grad = criterion_gradient(gam, gam.y_, log_sp, method="ml")
    hess = criterion_hessian(gam, gam.y_, log_sp, method="ml")
    rgrad = np.asarray(expected["fit"]["outer_grad"], dtype=np.float64)
    rhess = np.asarray(expected["fit"]["outer_hess"], dtype=np.float64)
    _show("grad", grad)
    _show("mgcv grad", rgrad)
    _show("grad max_abs", _max_abs(grad, rgrad))
    _show("hess diff", np.asarray(hess) - rhess)
    _show("hess max_abs", _max_abs(hess, rhess))


def gaulss_two_cr_diagnostics() -> None:
    print("\n== gaulss_two_cr diagnostics ==")
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    data = _gaulss_two_smooth_data()
    expected = _run_mgcv_snapshot(data, formula, "gaulss", "ML")
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")
    _show("actual sp", gam.smoothing_params)
    _show("mgcv sp", expected["fit"]["smoothing_params"])
    _show("actual log_sp", np.log(gam.smoothing_params))
    _show("mgcv log_sp", expected["fit"]["log_smoothing_params"])
    _show("sp_vcov actual", sp_vcov(gam, edge_correct=False))
    _show("one_se actual", gam.one_se_rule())
    _show("one_se mgcv", expected["parity"]["diagnostics"]["one_se_rule"])
    _show(
        "gam_vcomp actual",
        gam.gam_vcomp(rescale=False),
    )
    _show(
        "gam_vcomp mgcv",
        _run_mgcv_gam_vcomp(data, formula, "gaulss", "ML", rescale=False),
    )
    _show("fit_result deviance", gam.fit_core_solution_.fit_result.deviance)
    _show("fit_summary deviance", gam.fit_result().deviance)
    _show("mgcv deviance", expected["fit"]["deviance"])
    print(summary_text(gam))


def main() -> None:
    gevlss_cr_endpoint()
    gevlss_t2_fixed_derivatives()
    gaulss_two_cr_diagnostics()


if __name__ == "__main__":
    main()
