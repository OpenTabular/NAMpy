from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data

from nampy.gam.model_state import _edf2, _edf_total, _fit_result
from tests.mgcv_parity_utils import _run_mgcv_snapshot


def main() -> None:
    case = MatrixCase(
        case_id="diagnostic_s_cr_negbin_est_fixed",
        formula='y ~ s(x0, bs="cr", k=8, sp=1.1)',
        family={"name": "negbin", "theta": 1.7, "estimate_theta": True},
        method="fixed",
        data_kind="count",
    )
    data = make_data(case.data_kind)
    gam = fit_model(case, data)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.method,
        allow_live_run=True,
    )
    fit_result = _fit_result(gam)
    edf2 = _edf2(gam)
    p_val, p_df = gam._loglik_value_and_effective_df()
    print("actual logLik", gam.loglik())
    print("expected logLik", expected["fit"].get("loglik"))
    print("actual AIC", gam.aic())
    print("expected AIC", expected["fit"].get("aic"))
    print("family_class", getattr(gam.family, "family_class", None))
    print("n_theta", getattr(gam.family, "n_theta", None))
    print("theta", getattr(gam.family, "theta", None))
    print("expected theta", expected["fit"].get("family_theta"))
    print("edf total", float(_edf_total(gam)))
    print("fit_result edf", float(fit_result.edf))
    print("expected edf total", expected["fit"].get("edf_total"))
    print("edf2 sum", None if edf2 is None else float(np.sum(edf2)))
    expected_edf2 = expected["fit"].get("edf2")
    print(
        "expected edf2 sum",
        None if expected_edf2 is None else float(np.sum(np.asarray(expected_edf2))),
    )
    print("p_val", p_val)
    print("p_df", p_df)


if __name__ == "__main__":
    main()
