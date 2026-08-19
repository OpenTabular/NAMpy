from __future__ import annotations

import numpy as np
from nampy.gam.fit.solve_ops import solve_gaussian_given_smoothing
from nampy.gam.fit.selection.criteria.pirls_deriv import _gdi1_kernel

from nampy.gam import GAM
from nampy.gam._model_state import _fit_intercept, _n_coef, _penalty_blocks_seq
from nampy.gam.fit.solvers.stacked_qr import (
    _stacked_penalized_ls_nonneg_solution,
    solve_gaussian_penalized_ls_stacked_qr,
)
from nampy.gam.fit.state import (
    _restore_pirls_dbeta_to_original_parameterization,
    _restore_pirls_rank_root_to_original_parameterization,
)
from nampy.gam.fit.selection.criteria.gaussian import (
    criterion_ml_reml_exact,
    criterion_ml_reml_exact_dynamic,
)
from nampy.gam.fit.selection.criteria.gaussian_dyn import (
    _gaussian_penalty_quadratic_mgcv_style,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_hessian_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_joint,
)
from nampy.gam.fit.selection.criteria.gaussian_reml_algebra import (
    quadratic_form_penalty,
)
from nampy.gam.fit.selection.reparam import (
    _stable_penalty_logdet_derivatives,
    build_penalty_reparameterization_state,
)
from tests.mgcv_parity_utils import _make_mrf_data, _run_mgcv_snapshot
from tests.optimization.test_mgcv_fixed_inner_fit_parity import (
    _run_mgcv_fit3_fixed_sp,
    _run_mgcv_magic_fixed_sp,
)
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _run_mgcv_outer_trace,
)
from tests.optimization.test_mgcv_postprocessing_final_fit_parity import (
    ORDINARY_CASES,
    _fit_requested_case,
    _nampy_optimizer_name,
)


def inspect_gamma() -> None:
    case = next(c for c in ORDINARY_CASES if c.case_id == "gamma_log")
    expected = _run_mgcv_snapshot(
        data=case.data_factory(),
        formula=case.formula,
        family=case.family,
        method="REML",
        select=case.select,
        weights_column=case.weights_column,
    )
    optimizer = _nampy_optimizer_name(expected)
    data, gam, _warnings = _fit_requested_case(
        case,
        method="REML",
        optimizer=optimizer,
    )
    sol = gam.gam_result_.fit_core_solution
    fit = sol.fit_result
    sp = np.asarray(gam.smoothing_params, dtype=np.float64)
    sp_expected = np.atleast_1d(
        np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    )
    kernel = _gdi1_kernel(gam, gam.y_, sol, sp, method="REML")
    rank_root = _restore_pirls_rank_root_to_original_parameterization(
        kernel.current,
        kernel.current.rank_root,
    )
    vp_kernel = np.asarray(fit.scale * (rank_root @ rank_root.T), dtype=np.float64)
    vp_expected = np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64)
    expected_fit3 = _run_mgcv_fit3_fixed_sp(data, case.formula, "gamma", sp)
    vp_fit3 = np.asarray(expected_fit3["rV"], dtype=np.float64)
    vp_fit3 = np.asarray(
        float(expected_fit3["scale_est"]) * (vp_fit3 @ vp_fit3.T), dtype=np.float64
    )

    print("=== gamma_log ===")
    print("sp", sp)
    print("sp expected", sp_expected)
    print("log sp diff", np.log(sp) - np.log(sp_expected))
    print("criterion actual", float(gam.smoothing_score_))
    print("criterion expected", float(expected["fit"]["criterion_value"]))
    print(
        "coef max abs err vs fit3",
        float(
            np.max(
                np.abs(
                    np.asarray(sol.coef_full, dtype=np.float64)
                    - np.asarray(expected_fit3["coefficients"], dtype=np.float64)
                )
            )
        ),
    )
    print(
        "eta max abs err vs fit3",
        float(
            np.max(
                np.abs(
                    np.asarray(sol.eta, dtype=np.float64)
                    - np.asarray(expected_fit3["linear_predictors"], dtype=np.float64)
                )
            )
        ),
    )
    print(
        "mu max abs err vs fit3",
        float(
            np.max(
                np.abs(
                    np.asarray(sol.mu, dtype=np.float64)
                    - np.asarray(expected_fit3["fitted_values"], dtype=np.float64)
                )
            )
        ),
    )
    print(
        "fisher weights max abs err vs fit3",
        float(
            np.max(
                np.abs(
                    np.asarray(sol.fisher_weights, dtype=np.float64)
                    - np.asarray(expected_fit3["weights"], dtype=np.float64)
                )
            )
        ),
    )
    print(
        "working weights max abs err vs fit3",
        float(
            np.max(
                np.abs(
                    np.asarray(sol.working_weights, dtype=np.float64)
                    - np.asarray(expected_fit3["working_weights"], dtype=np.float64)
                )
            )
        ),
    )
    print(
        "working response max abs err vs fit3",
        float(
            np.max(
                np.abs(
                    np.asarray(sol.working_response, dtype=np.float64)
                    - np.asarray(expected_fit3["working_response"], dtype=np.float64)
                )
            )
        ),
    )
    print("scale actual", float(fit.scale))
    print("scale expected", float(expected["fit"]["scale"]))
    print("scale fit3", float(expected_fit3["scale_est"]))
    print("diag actual", np.diag(np.asarray(fit.cov_bayes, dtype=np.float64)))
    print("diag kernel", np.diag(vp_kernel))
    print("diag fit3", np.diag(vp_fit3))
    print("diag expected", np.diag(vp_expected))
    print(
        "max abs diag error current",
        float(
            np.max(
                np.abs(
                    np.diag(np.asarray(fit.cov_bayes, dtype=np.float64))
                    - np.diag(vp_expected)
                )
            )
        ),
    )
    print(
        "max abs diag error kernel",
        float(np.max(np.abs(np.diag(vp_kernel) - np.diag(vp_expected)))),
    )


def inspect_mrf() -> None:
    data = _make_mrf_data()
    formula = (
        'y ~ s(region, bs="mrf", k=3, '
        'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
    )
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    expected_outer = _run_mgcv_outer_trace(data, formula, "gaussian", "REML", "newton")
    saved_sigma = getattr(gam, "_gaussian_reml_sigma2_opt_", None)
    gam._gaussian_reml_sigma2_opt_ = None
    sp_actual = np.atleast_1d(np.asarray(gam.smoothing_params, dtype=np.float64))
    sp_expected = np.atleast_1d(
        np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    )
    expected_fixed_actual = _run_mgcv_magic_fixed_sp(data, formula, sp_actual)
    expected_fixed_expected = _run_mgcv_magic_fixed_sp(data, formula, sp_expected)
    fit3_fixed_actual = _run_mgcv_fit3_fixed_sp(data, formula, "gaussian", sp_actual)
    fit3_fixed_expected = _run_mgcv_fit3_fixed_sp(data, formula, "gaussian", sp_expected)
    log_sp_actual = np.log(sp_actual)
    log_sp_expected = np.log(sp_expected)
    sol_actual = solve_gaussian_given_smoothing(gam, gam.y_, sp_actual)
    sol_expected = solve_gaussian_given_smoothing(gam, gam.y_, sp_expected)
    stacked_actual = solve_gaussian_penalized_ls_stacked_qr(
        np.asarray(sol_actual["X"], dtype=np.float64),
        np.asarray(gam.y_, dtype=np.float64),
        np.asarray(sol_actual["working_weights"], dtype=np.float64),
        np.asarray(sol_actual["penalty_matrix"], dtype=np.float64),
        penalty_blocks=tuple(_penalty_blocks_seq(gam)),
        fit_intercept=bool(_fit_intercept(gam)),
        n_coef=int(_n_coef(gam)),
    )
    stacked_expected = solve_gaussian_penalized_ls_stacked_qr(
        np.asarray(sol_expected["X"], dtype=np.float64),
        np.asarray(gam.y_, dtype=np.float64),
        np.asarray(sol_expected["working_weights"], dtype=np.float64),
        np.asarray(sol_expected["penalty_matrix"], dtype=np.float64),
        penalty_blocks=tuple(_penalty_blocks_seq(gam)),
        fit_intercept=bool(_fit_intercept(gam)),
        n_coef=int(_n_coef(gam)),
    )
    canon_actual = build_penalty_reparameterization_state(
        gam,
        np.asarray(sol_actual["X"], dtype=np.float64),
        sp_actual,
        deriv=0,
    )
    X_canon_actual = np.asarray(
        np.asarray(sol_actual["X"], dtype=np.float64)
        @ np.asarray(canon_actual.T, dtype=np.float64),
        dtype=np.float64,
    )
    pls_actual = _stacked_penalized_ls_nonneg_solution(
        X_canon_actual,
        np.asarray(gam.y_, dtype=np.float64),
        np.asarray(sol_actual["working_weights"], dtype=np.float64),
        penalty_sqrt=np.asarray(canon_actual.Sr, dtype=np.float64),
        penalty_rank_rows=np.asarray(canon_actual.Eb, dtype=np.float64),
        rank_tol=np.finfo(np.float64).eps * 100.0,
    )
    canon_expected = build_penalty_reparameterization_state(
        gam,
        np.asarray(sol_expected["X"], dtype=np.float64),
        sp_expected,
        deriv=0,
    )
    X_canon_expected = np.asarray(
        np.asarray(sol_expected["X"], dtype=np.float64)
        @ np.asarray(canon_expected.T, dtype=np.float64),
        dtype=np.float64,
    )
    pls_expected = _stacked_penalized_ls_nonneg_solution(
        X_canon_expected,
        np.asarray(gam.y_, dtype=np.float64),
        np.asarray(sol_expected["working_weights"], dtype=np.float64),
        penalty_sqrt=np.asarray(canon_expected.Sr, dtype=np.float64),
        penalty_rank_rows=np.asarray(canon_expected.Eb, dtype=np.float64),
        rank_tol=np.finfo(np.float64).eps * 100.0,
    )
    kernel_actual = _gdi1_kernel(gam, gam.y_, sol_actual, sp_actual, method="REML")
    kernel_expected = _gdi1_kernel(gam, gam.y_, sol_expected, sp_expected, method="REML")
    dbeta_actual = np.column_stack(
        [
            _restore_pirls_dbeta_to_original_parameterization(
                kernel_actual.current,
                kernel_actual.ift.dbeta[j],
            )
            for j in range(len(kernel_actual.ift.dbeta))
        ]
    )
    dbeta_expected = np.column_stack(
        [
            _restore_pirls_dbeta_to_original_parameterization(
                kernel_expected.current,
                kernel_expected.ift.dbeta[j],
            )
            for j in range(len(kernel_expected.ift.dbeta))
        ]
    )
    coef_expected_snapshot = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)

    quad_actual = quadratic_form_penalty(
        np.asarray(sol_actual["coef_full"], dtype=np.float64),
        np.asarray(sol_actual["penalty_matrix"], dtype=np.float64),
    )
    quad_expected = quadratic_form_penalty(
        np.asarray(sol_expected["coef_full"], dtype=np.float64),
        np.asarray(sol_expected["penalty_matrix"], dtype=np.float64),
    )
    mgcv_quad_actual = _gaussian_penalty_quadratic_mgcv_style(gam, sol_actual, sp_actual)
    mgcv_quad_expected = _gaussian_penalty_quadratic_mgcv_style(
        gam, sol_expected, sp_expected
    )
    crit_actual = criterion_ml_reml_gaussian_dynamic_joint(
        gam,
        gam.y_,
        log_sp_actual,
        float(np.log(sol_actual["scale"])),
        method="REML",
    )
    crit_expected = criterion_ml_reml_gaussian_dynamic_joint(
        gam,
        gam.y_,
        log_sp_expected,
        float(np.log(sol_expected["scale"])),
        method="REML",
    )
    crit_expected_snapshot_scale = criterion_ml_reml_gaussian_dynamic_joint(
        gam,
        gam.y_,
        log_sp_expected,
        float(np.log(float(expected["fit"]["scale"]))),
        method="REML",
    )
    prof_actual = criterion_ml_reml_exact_dynamic(gam, gam.y_, log_sp_actual, "REML")
    prof_expected = criterion_ml_reml_exact_dynamic(
        gam, gam.y_, log_sp_expected, "REML"
    )
    exact_actual = criterion_ml_reml_exact(gam, gam.y_, log_sp_actual, "REML")
    exact_expected = criterion_ml_reml_exact(gam, gam.y_, log_sp_expected, "REML")
    grad_actual = criterion_gradient_ml_reml_gaussian_dynamic_joint(
        gam,
        gam.y_,
        log_sp_actual,
        float(np.log(sol_actual["scale"])),
        method="REML",
    )
    grad_expected_snapshot_scale = criterion_gradient_ml_reml_gaussian_dynamic_joint(
        gam,
        gam.y_,
        log_sp_expected,
        float(np.log(float(expected["fit"]["scale"]))),
        method="REML",
    )
    hess_actual = criterion_hessian_ml_reml_gaussian_dynamic_joint(
        gam,
        gam.y_,
        log_sp_actual,
        float(np.log(sol_actual["scale"])),
        method="REML",
    )
    gam._gaussian_reml_sigma2_opt_ = saved_sigma
    logdet_s_actual, logdet_s1_actual, logdet_s2_actual = (
        _stable_penalty_logdet_derivatives(gam, sp_actual, order=2)
    )
    logdet_s_expected, logdet_s1_expected, logdet_s2_expected = (
        _stable_penalty_logdet_derivatives(gam, sp_expected, order=2)
    )

    print("=== mrf_exact_reml ===")
    print("sp actual", sp_actual, log_sp_actual)
    print("sp expected", sp_expected, log_sp_expected)
    print("optimizer expected", expected["fit"].get("optimizer"))
    print("outer trace fit", expected_outer["fit"])
    print("outer trace rows", expected_outer["trace"])
    print("scale actual", float(sol_actual["scale"]))
    print("scale expected", float(sol_expected["scale"]))
    print("snapshot scale expected", float(expected["fit"]["scale"]))
    print("trace_H actual", float(sol_actual["trace_H"]))
    print("trace_H expected-sp", float(sol_expected["trace_H"]))
    print("mgcv fit3 trA actual", float(fit3_fixed_actual["trA"]))
    print("mgcv fit3 trA expected-sp", float(fit3_fixed_expected["trA"]))
    print("criterion actual", float(gam.smoothing_score_))
    print("criterion expected", float(expected["fit"]["criterion_value"]))
    print("mgcv fixed-sp REML @ actual sp", float(expected_fixed_actual["reml"]))
    print("mgcv fixed-sp REML @ expected sp", float(expected_fixed_expected["reml"]))
    print("mgcv magic deviance actual", float(expected_fixed_actual["deviance"]))
    print("mgcv magic deviance expected", float(expected_fixed_expected["deviance"]))
    print("mgcv fit3 REML @ actual sp", float(fit3_fixed_actual["REML"]))
    print("mgcv fit3 REML @ expected sp", float(fit3_fixed_expected["REML"]))
    print("mgcv fit3 deviance actual", float(fit3_fixed_actual["deviance"]))
    print("mgcv fit3 deviance expected", float(fit3_fixed_expected["deviance"]))
    print("mgcv fit3 scale_est actual", float(fit3_fixed_actual["scale_est"]))
    print("mgcv fit3 scale_est expected", float(fit3_fixed_expected["scale_est"]))
    print(
        "mgcv fit3 reml.scale actual",
        None
        if fit3_fixed_actual["reml_scale"] is None
        else float(fit3_fixed_actual["reml_scale"]),
    )
    print(
        "mgcv fit3 reml.scale expected",
        None
        if fit3_fixed_expected["reml_scale"] is None
        else float(fit3_fixed_expected["reml_scale"]),
    )
    print("exact criterion actual", float(exact_actual))
    print("exact criterion expected-sp", float(exact_expected))
    print("profiled criterion actual", float(prof_actual))
    print("profiled criterion expected-sp", float(prof_expected))
    print("criterion joint@actualscale", float(crit_actual))
    print("criterion joint@expectedscale", float(crit_expected))
    print(
        "criterion joint@expected snapshot scale",
        float(crit_expected_snapshot_scale),
    )
    print("grad actual", np.asarray(grad_actual, dtype=np.float64))
    print(
        "grad expected snapshot scale",
        np.asarray(grad_expected_snapshot_scale, dtype=np.float64),
    )
    print("hess actual", np.asarray(hess_actual, dtype=np.float64))
    print("deviance actual", float(sol_actual["deviance"]))
    print("deviance expected-sp", float(sol_expected["deviance"]))
    print(
        "stacked raw eta dev actual",
        float(np.sum((np.asarray(gam.y_, dtype=np.float64) - stacked_actual["eta"]) ** 2)),
    )
    print(
        "stacked raw eta dev expected-sp",
        float(
            np.sum((np.asarray(gam.y_, dtype=np.float64) - stacked_expected["eta"]) ** 2)
        ),
    )
    print(
        "stacked coef rss actual",
        float(
            np.sum(
                (
                    np.asarray(gam.y_, dtype=np.float64)
                    - np.asarray(sol_actual["X"], dtype=np.float64)
                    @ np.asarray(stacked_actual["coef_full"], dtype=np.float64)
                )
                ** 2
            )
        ),
    )
    print(
        "stacked coef rss expected-sp",
        float(
            np.sum(
                (
                    np.asarray(gam.y_, dtype=np.float64)
                    - np.asarray(sol_expected["X"], dtype=np.float64)
                    @ np.asarray(stacked_expected["coef_full"], dtype=np.float64)
                )
                ** 2
            )
        ),
    )
    print(
        "canonical pls rss actual",
        float(
            np.sum(
                (
                    np.asarray(gam.y_, dtype=np.float64)
                    - X_canon_actual @ np.asarray(pls_actual.coef_full, dtype=np.float64)
                )
                ** 2
            )
        ),
    )
    print(
        "canonical pls rss expected-sp",
        float(
            np.sum(
                (
                    np.asarray(gam.y_, dtype=np.float64)
                    - X_canon_expected
                    @ np.asarray(pls_expected.coef_full, dtype=np.float64)
                )
                ** 2
            )
        ),
    )
    print("kernel D1 actual", np.asarray(kernel_actual.D1, dtype=np.float64))
    print("kernel D1 expected-sp", np.asarray(kernel_expected.D1, dtype=np.float64))
    print("kernel bSb1 actual", np.asarray(kernel_actual.bSb1, dtype=np.float64))
    print(
        "kernel bSb1 expected-sp", np.asarray(kernel_expected.bSb1, dtype=np.float64)
    )
    print(
        "kernel Dp1 actual",
        np.asarray(kernel_actual.D1 + kernel_actual.bSb1, dtype=np.float64),
    )
    print(
        "kernel Dp1 expected-sp",
        np.asarray(kernel_expected.D1 + kernel_expected.bSb1, dtype=np.float64),
    )
    print(
        "mgcv fit3 D1 actual",
        np.asarray(fit3_fixed_actual["D1"], dtype=np.float64),
    )
    print(
        "mgcv fit3 D1 expected-sp",
        np.asarray(fit3_fixed_expected["D1"], dtype=np.float64),
    )
    print(
        "kernel K1 actual",
        np.asarray(kernel_actual.K1, dtype=np.float64),
    )
    print(
        "kernel K1 expected-sp",
        np.asarray(kernel_expected.K1, dtype=np.float64),
    )
    print(
        "mgcv fit3 REML1 actual",
        np.asarray(fit3_fixed_actual["REML1"], dtype=np.float64),
    )
    print(
        "mgcv fit3 REML1 expected-sp",
        np.asarray(fit3_fixed_expected["REML1"], dtype=np.float64),
    )
    print(
        "kernel D2 actual",
        np.asarray(kernel_actual.D2, dtype=np.float64),
    )
    print(
        "kernel D2 expected-sp",
        np.asarray(kernel_expected.D2, dtype=np.float64),
    )
    print(
        "kernel bSb2 actual",
        np.asarray(kernel_actual.bSb2, dtype=np.float64),
    )
    print(
        "kernel bSb2 expected-sp",
        np.asarray(kernel_expected.bSb2, dtype=np.float64),
    )
    print(
        "kernel Dp2 actual",
        np.asarray(kernel_actual.D2 + kernel_actual.bSb2, dtype=np.float64),
    )
    print(
        "kernel Dp2 expected-sp",
        np.asarray(kernel_expected.D2 + kernel_expected.bSb2, dtype=np.float64),
    )
    print(
        "mgcv fit3 D2 actual",
        np.asarray(fit3_fixed_actual["D2"], dtype=np.float64),
    )
    print(
        "mgcv fit3 D2 expected-sp",
        np.asarray(fit3_fixed_expected["D2"], dtype=np.float64),
    )
    print("dbeta actual", np.asarray(dbeta_actual, dtype=np.float64))
    print("dbeta expected-sp", np.asarray(dbeta_expected, dtype=np.float64))
    print(
        "mgcv fit3 db_drho actual",
        np.asarray(fit3_fixed_actual["db_drho"], dtype=np.float64),
    )
    print(
        "mgcv fit3 db_drho expected-sp",
        np.asarray(fit3_fixed_expected["db_drho"], dtype=np.float64),
    )
    print(
        "coef max abs err vs snapshot at actual sp",
        float(
            np.max(
                np.abs(
                    np.asarray(sol_actual["coef_full"], dtype=np.float64)
                    - coef_expected_snapshot
                )
            )
        ),
    )
    print(
        "coef max abs err vs snapshot at expected sp",
        float(
            np.max(
                np.abs(
                    np.asarray(sol_expected["coef_full"], dtype=np.float64)
                    - coef_expected_snapshot
                )
            )
        ),
    )
    print("rss(mu) actual", float(np.sum((gam.y_ - sol_actual["mu"]) ** 2)))
    print("rss(mu) expected-sp", float(np.sum((gam.y_ - sol_expected["mu"]) ** 2)))
    print("kernel bSb actual", float(kernel_actual.bSb))
    print("kernel bSb expected-sp", float(kernel_expected.bSb))
    print("quad actual", float(quad_actual))
    print("quad expected-sp", float(quad_expected))
    print("mgcv quad actual", float(mgcv_quad_actual))
    print("mgcv quad expected-sp", float(mgcv_quad_expected))
    print("ldet actual", float(kernel_actual.ldet_XWX_plus_S))
    print("ldet expected-sp", float(kernel_expected.ldet_XWX_plus_S))
    print("logdet S actual", float(logdet_s_actual), logdet_s1_actual, logdet_s2_actual)
    print(
        "logdet S expected",
        float(logdet_s_expected),
        logdet_s1_expected,
        logdet_s2_expected,
    )


if __name__ == "__main__":
    inspect_gamma()
    inspect_mrf()
