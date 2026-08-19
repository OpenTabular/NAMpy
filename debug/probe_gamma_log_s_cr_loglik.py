from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nampy.gam._model_state import _coef_column_offset, _fit_result
from nampy.gam.linalg.reindexing import permute_columns
from nampy.gam.fit.solvers.stacked_qr import _pivoted_economic_qr
from nampy.gam.fit.selection.criteria.pirls.value import (
    _solve_gamma_profile_scale,
)
from nampy.gam.fit.selection.reparam import _static_penalty_null_dim
from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data
from tests.mgcv_parity_utils import _run_mgcv_snapshot


def main() -> None:
    terms = {
        "s_cr": 's(x0, bs="cr", k=8, sp=1.1)',
        "te_3d_cr": 'te(x0, x1, x2, bs=["cr","cr","cr"], k=[5,5,5], sp=[1.0,1.2,1.4])',
    }
    term_name = sys.argv[1] if len(sys.argv) > 1 else "s_cr"
    rhs = terms[term_name]
    case = MatrixCase(
        case_id=f"diagnostic_{term_name}_gamma_log_fixed",
        formula=f"y ~ {rhs}",
        family={"name": "gamma", "link": "log"},
        method="fixed",
        data_kind="positive",
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

    y = np.asarray(data["y"], dtype=np.float64)
    mu = np.asarray(gam.predict(type="response"), dtype=np.float64)
    expected_mu = np.asarray(expected["predictions"]["response"], dtype=np.float64)
    weights = np.ones_like(y)
    fit_result = _fit_result(gam)
    scale = float(fit_result.scale)
    penalty = float(fit_result.penalty_quadratic or 0.0)
    mp = float(_static_penalty_null_dim(gam) + _coef_column_offset(gam))
    reml_scale = _solve_gamma_profile_scale(
        gam,
        y,
        float(fit_result.deviance) + penalty,
        mp,
        method="REML",
        init_scale=scale,
    )

    def gamma_aic_for(mu_values: np.ndarray, disp: float) -> float:
        shape = 1.0 / disp
        return (
            -2.0
            * float(
                np.sum(
                    weights
                    * (
                        (shape - 1.0) * np.log(y)
                        - y / (mu_values * disp)
                        - gammaln(shape)
                        - shape * np.log(mu_values * disp)
                    )
                )
            )
            + 2.0
        )

    gamma_aic = gamma_aic_for(mu, scale)
    gamma_aic_expected_mu = gamma_aic_for(expected_mu, scale)
    gamma_aic_expected_scale_mu = gamma_aic_for(
        expected_mu,
        float(expected["fit"].get("scale")),
    )
    gamma_aic_reml_scale = gamma_aic_for(expected_mu, reml_scale)
    object_aic_from_expected_loglik = 2.0 * (
        np.sum(fit_result.edf) + 1.0 - expected["fit"].get("loglik")
    )
    object_aic = gamma_aic + 2.0 * float(np.sum(fit_result.edf))
    loglik_from_aic = float(np.sum(fit_result.edf) + 1.0 - object_aic / 2.0)
    object_aic_reml = gamma_aic_reml_scale + 2.0 * float(np.sum(fit_result.edf))
    edf2 = None if fit_result.edf2 is None else float(np.sum(fit_result.edf2))
    weights_qr = gam.gam_result_.fit_core_solution.fit_state.fisher_weights
    if weights_qr is None:
        weights_qr = gam.gam_result_.fit_core_solution.fit_state.working_weights
    X_qr = np.asarray(gam.gam_result_.fit_core_solution.fit_state.X, dtype=np.float64)
    weights_qr = np.asarray(weights_qr, dtype=np.float64)
    WX = np.sqrt(weights_qr)[:, None] * X_qr
    _q_wx, r_wx_pivoted, pivot_wx = _pivoted_economic_qr(WX)
    R_wx = permute_columns(
        r_wx_pivoted,
        np.asarray(pivot_wx, dtype=np.int64),
        reverse=True,
    )
    RTR_wx = np.asarray(R_wx.T @ R_wx, dtype=np.float64)
    fixed_qr_edf2 = np.asarray(
        np.sum(np.asarray(fit_result.cov_bayes, dtype=np.float64) * RTR_wx, axis=1)
        / scale,
        dtype=np.float64,
    )
    expected_edf2 = expected["fit"].get("edf2")
    expected_edf2_sum = (
        None
        if expected_edf2 is None
        else float(np.sum(np.asarray(expected_edf2, dtype=np.float64)))
    )

    print("actual logLik", gam.loglik())
    print("expected logLik", expected["fit"].get("loglik"))
    print("actual AIC", gam.aic())
    print("expected AIC(fit)", expected["fit"].get("aic"))
    print("actual scale", scale)
    print("profile reml scale", reml_scale)
    print("penalty", penalty)
    print("mp", mp)
    print("expected scale", expected["fit"].get("scale"))
    print("actual deviance", fit_result.deviance)
    print("expected deviance", expected["fit"].get("deviance"))
    print("actual edf total", float(np.sum(fit_result.edf)))
    print("expected edf total", expected["fit"].get("edf_total"))
    print("actual edf2 sum", edf2)
    print("fixed QR edf2 sum", float(np.sum(fixed_qr_edf2)))
    print("expected edf2 sum", expected_edf2_sum)
    print("max response diff", float(np.max(np.abs(mu - expected_mu))))
    print("gamma aic", gamma_aic)
    print("gamma aic expected mu", gamma_aic_expected_mu)
    print("gamma aic expected scale and mu", gamma_aic_expected_scale_mu)
    print("gamma aic reml scale", gamma_aic_reml_scale)
    print("expected object aic from logLik", object_aic_from_expected_loglik)
    print("object aic", object_aic)
    print("object aic reml", object_aic_reml)
    print("logLik from object aic", loglik_from_aic)


if __name__ == "__main__":
    main()
