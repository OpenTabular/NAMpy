from __future__ import annotations

import numpy as np

from nampy.gam.inference.anova import (
    _edf1_vector,
    _edf2,
    _edf_total,
    _residual_df_approx_mgcv,
)
from tests.families.test_general_family_mgcv_parity import _gevlss_data
from tests.mgcv_parity_utils import _fit_nampy_model, _run_mgcv_gam_vcomp
from tests.parity.test_mgcv_prediction_inference_diagnostics_parity import (
    _make_gamma_data,
)


def _print_vcomp_probe(data, formula, family, method) -> None:
    gam = _fit_nampy_model(data, formula, family, method)
    expected = _run_mgcv_gam_vcomp(data, formula, family, method, rescale=False)
    actual = gam.gam_vcomp(rescale=False)
    result = getattr(gam, "_optim_result", None)
    outer_info = {} if result is None else dict(getattr(result, "outer_info", {}) or {})
    fit_summary = getattr(gam, "fit_summary_", None)
    fit_result = getattr(gam, "fit_result_", None)
    fit_core = getattr(gam, "fit_core_solution_", None)
    fit_state = None if fit_core is None else getattr(fit_core, "fit_state", None)

    print("== vcomp probe ==")
    print(f"family={family!r} method={method!r}")
    print("smoothing_params", np.asarray(gam.smoothing_params, dtype=np.float64))
    print("fit_summary.scale", None if fit_summary is None else getattr(fit_summary, "scale", None))
    print("fit_result.scale", None if fit_result is None else getattr(fit_result, "scale", None))
    print("fit_core.fit_result.scale", None if fit_core is None else getattr(fit_core.fit_result, "scale", None))
    print("fit_state.loglik", None if fit_state is None else getattr(fit_state, "loglik", None))
    print(
        "result.fun",
        None if result is None else getattr(result, "fun", None),
    )
    print(
        "result.x",
        None if result is None else np.asarray(getattr(result, "x", None), dtype=np.float64),
    )
    print(
        "result.hess.shape",
        None if result is None or getattr(result, "hess", None) is None else np.asarray(result.hess).shape,
    )
    print("outer_info[hess].shape", None if outer_info.get("hess", None) is None else np.asarray(outer_info["hess"]).shape)
    print(
        "result.hess",
        None if result is None or getattr(result, "hess", None) is None else np.asarray(result.hess, dtype=np.float64),
    )
    print(
        "outer_info[hess]",
        None if outer_info.get("hess", None) is None else np.asarray(outer_info["hess"], dtype=np.float64),
    )
    print("expected", expected)
    print("actual", actual)
    print()


def _print_gamma_anova_probe() -> None:
    data = _make_gamma_data()
    formulas = [
        'y ~ s(x0, bs="cr", k=8)',
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
    ]
    gam0 = _fit_nampy_model(data, formulas[0], "gamma", "REML")
    gam1 = _fit_nampy_model(data, formulas[1], "gamma", "REML")
    out = gam0.anova(gam1, test="Chisq")

    print("== gamma anova probe ==")
    for idx, gam in enumerate((gam0, gam1)):
        edf1 = np.asarray(_edf1_vector(gam), dtype=np.float64)
        edf2 = np.asarray(_edf2(gam), dtype=np.float64)
        print(
            f"model{idx}",
            {
                "edf_total": float(_edf_total(gam)),
                "sum_edf1": float(np.sum(edf1)),
                "sum_edf2": float(np.sum(edf2)),
                "dfc": float(np.sum(edf2) - _edf_total(gam)),
                "resid_df_approx": float(_residual_df_approx_mgcv(gam)),
            },
        )
    print(out.table)
    print()


def main() -> None:
    _print_vcomp_probe(
        _gevlss_data(),
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        "gevlss",
        "ML",
    )
    _print_gamma_anova_probe()


if __name__ == "__main__":
    main()
