from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.fit.solvers.general_family_solver import (
    run_general_family_fixed_smoothing,
    sl_initial_repara,
)
from nampy.gam.smooths.tensor.t2 import TensorANOVASplineTerm

from nampy.gam.fit.selection.criteria import (
    criterion_gradient_numerical,
    criterion_hessian_numerical,
)
from nampy.gam.smooths.algebra import t2_marginal_reparameterization
from nampy.gam.smooths.tensor.marginals import tensor_marginal_fit_matrices
from nampy.gam.smooths.univariate.cr import CubicSplineTerm
from nampy.gam.smooths.univariate.tp import ThinPlateSplineTerm
from tests.families.test_general_family_mgcv_parity import (
    GENERAL_SE_CASES,
    _general_newdata,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model_fixed_sp,
    _run_mgcv_natparam_type3,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_smoothcon_predict_matrix,
    _run_mgcv_snapshot,
)
from tests.optimization.test_mgcv_fixed_inner_fit_parity import (
    _run_mgcv_fit5_fixed_sp,
)
from tests.optimization.test_mgcv_general_family_preoptimization_parity import (
    _run_mgcv_general_preoptimization,
)


def _fit5_linear_predictors(setup, fit, offset_list):
    coef = np.asarray(fit["coef"], dtype=np.float64)
    eta_cols = []
    for k, jj in enumerate(setup.jj):
        eta_k = setup.X_initial[:, jj] @ coef[jj]
        if offset_list is not None and k < len(offset_list):
            off_k = offset_list[k]
            if off_k is not None:
                eta_k = eta_k + np.asarray(off_k, dtype=np.float64)
        eta_cols.append(np.asarray(eta_k, dtype=np.float64))
    return np.column_stack(eta_cols)


def _case_table():
    return {case[0]: case for case in GENERAL_SE_CASES}


def _max_abs_diff(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"Shape mismatch: {aa.shape} != {bb.shape}")
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def _try_max_abs_diff(a, b):
    try:
        return _max_abs_diff(a, b)
    except Exception as exc:  # debug helper
        return {"error": str(exc)}


def _as_float_or_none(x) -> float | None:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _column_sign_alignment_report(actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape:
        raise ValueError(f"Shape mismatch: {a.shape} != {e.shape}")
    if a.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {a.ndim}D.")

    signs = np.ones(a.shape[1], dtype=np.float64)
    for j in range(a.shape[1]):
        if np.linalg.norm(a[:, j] - e[:, j]) > np.linalg.norm(-a[:, j] - e[:, j]):
            signs[j] = -1.0
    aligned = a * signs[np.newaxis, :]
    return {
        "best_signs": signs.tolist(),
        "signed_max_abs_diff": _max_abs_diff(aligned, e),
        "raw_col_max_abs_diff": np.max(np.abs(a - e), axis=0).tolist(),
        "signed_col_max_abs_diff": np.max(np.abs(aligned - e), axis=0).tolist(),
    }


def _column_signature(vec):
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size == 0:
        return {}
    idx = int(np.argmax(np.abs(v)))
    return {
        "sum": float(np.sum(v)),
        "first": float(v[0]),
        "last": float(v[-1]),
        "argmax_abs_index": idx,
        "argmax_abs_value": float(v[idx]),
    }


def _hp_diagnostics(fit: dict) -> dict[str, float | int | None]:
    lbb = np.asarray(fit["lbb"], dtype=np.float64)
    St = np.asarray(fit["St_full"], dtype=np.float64)
    Hp = -lbb + St
    sign, logabsdet = np.linalg.slogdet(Hp)
    evals = np.linalg.eigvalsh(0.5 * (Hp + Hp.T))
    abs_evals = np.abs(evals)
    pos = evals[evals > 0.0]
    return {
        "hp_logdet_direct": None if sign <= 0 else float(logabsdet),
        "hp_min_eval": float(np.min(evals)) if evals.size else None,
        "hp_max_eval": float(np.max(evals)) if evals.size else None,
        "hp_rank_pos": int(np.count_nonzero(pos > 0.0)),
        "hp_cond_abs": (
            None
            if abs_evals.size == 0 or np.min(abs_evals) == 0.0
            else float(np.max(abs_evals) / np.min(abs_evals))
        ),
    }


def _maybe_t2_lpmatrix_diagnostics(case_id, family, formula, data, gam):
    formula_text = " ".join(str(f) for f in formula)
    if "t2(" not in formula_text or not {"x0", "x1"}.issubset(set(data.columns)):
        return {}

    full = "full=True" in formula_text
    newdata = _general_newdata(data)
    expected_lp = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family=family,
        method="ML",
        type="lpmatrix",
        return_se=False,
        select=("select_true" in case_id),
    )
    expected_lp = np.asarray(expected_lp["pred"], dtype=np.float64)
    actual_lp = np.asarray(gam.predict(newdata, type="lpmatrix"), dtype=np.float64)

    runtime_term = TensorANOVASplineTerm(
        feature=["x0", "x1"],
        k=[6, 6],
        basis=["tp", "cr"],
        full=full,
    )
    X_fit = data[["x0", "x1"]].to_numpy(dtype=np.float64)
    runtime_term.fit(X_fit, ["x0", "x1"])
    X_new = newdata[["x0", "x1"]].to_numpy(dtype=np.float64)
    runtime_basis = np.asarray(runtime_term.transform_new(X_new), dtype=np.float64)
    pred_basis_map = dict(runtime_term.metadata or {}).get("prediction_basis_map", None)
    if pred_basis_map is not None:
        runtime_basis = runtime_basis @ np.asarray(pred_basis_map, dtype=np.float64)

    expected_term = _run_mgcv_smoothcon_predict_matrix(
        data[["x0", "x1"]],
        newdata[["x0", "x1"]],
        (
            't2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)'
            if full
            else 't2(x0, x1, bs=["tp", "cr"], k=[6, 6])'
        ),
        absorb_cons=True,
        scale_penalty=True,
    )
    expected_term = np.asarray(expected_term["X"], dtype=np.float64)

    predictor0 = gam.gam_result_.compiled_model.predictors[0]
    compiled_term = predictor0.compiled_terms[0]
    compiled_term_basis = np.asarray(
        compiled_term.prediction_parameterization_matrix(X_new),
        dtype=np.float64,
    )
    expected_block = np.asarray(expected_lp[:, 1:], dtype=np.float64)

    tp_term = ThinPlateSplineTerm(feature="x0", k=6, basis="tp")
    tp_term.fit(data[["x0"]].to_numpy(dtype=np.float64), ["x0"])
    tp_raw_X, tp_raw_S, _ = tensor_marginal_fit_matrices(tp_term, centered=False)
    tp_actual = t2_marginal_reparameterization(tp_raw_X, tp_raw_S, basis_name="tp")
    tp_expected = _run_mgcv_natparam_type3(data[["x0"]], 's(x0, bs="tp", k=6)')
    tp_actual_X = np.column_stack([tp_actual["B_range"], tp_actual["B_null"]])
    tp_actual_P = np.column_stack([tp_actual["T_range"], tp_actual["T_null"]])
    tp_expected_X = np.asarray(tp_expected["X"], dtype=np.float64)
    tp_expected_P = np.asarray(tp_expected["P"], dtype=np.float64)

    cr_term = CubicSplineTerm(feature="x1", k=6, basis="cr")
    cr_term.fit(data[["x1"]].to_numpy(dtype=np.float64), ["x1"])
    cr_raw_X, cr_raw_S, _ = tensor_marginal_fit_matrices(cr_term, centered=False)
    cr_actual = t2_marginal_reparameterization(cr_raw_X, cr_raw_S, basis_name="cr")
    cr_expected = _run_mgcv_natparam_type3(data[["x1"]], 's(x1, bs="cr", k=6)')
    cr_actual_X = np.column_stack([cr_actual["B_range"], cr_actual["B_null"]])
    cr_actual_P = np.column_stack([cr_actual["T_range"], cr_actual["T_null"]])
    cr_expected_X = np.asarray(cr_expected["X"], dtype=np.float64)
    cr_expected_P = np.asarray(cr_expected["P"], dtype=np.float64)

    return {
        "runtime_t2_predict_max_abs_diff_vs_mgcv": _try_max_abs_diff(
            runtime_basis,
            expected_term,
        ),
        "compiled_term_predict_max_abs_diff_vs_mgcv_block": _try_max_abs_diff(
            compiled_term_basis,
            expected_block,
        ),
        "full_lpmatrix_max_abs_diff_vs_mgcv": _try_max_abs_diff(
            actual_lp,
            expected_lp,
        ),
        "compiled_term_vs_runtime_predict_max_abs_diff": _try_max_abs_diff(
            compiled_term_basis,
            runtime_basis,
        ),
        "tp_natparam_X_max_abs_diff": _try_max_abs_diff(
            tp_actual_X,
            tp_expected_X,
        ),
        "tp_natparam_P_max_abs_diff": _try_max_abs_diff(
            tp_actual_P,
            tp_expected_P,
        ),
        "cr_natparam_X_max_abs_diff": _try_max_abs_diff(
            cr_actual_X,
            cr_expected_X,
        ),
        "cr_natparam_P_max_abs_diff": _try_max_abs_diff(
            cr_actual_P,
            cr_expected_P,
        ),
        "cr_natparam_X_sign_report": _column_sign_alignment_report(
            cr_actual_X,
            cr_expected_X,
        ),
        "cr_natparam_P_sign_report": _column_sign_alignment_report(
            cr_actual_P,
            cr_expected_P,
        ),
        "cr_natparam_last_X_actual_signature": _column_signature(cr_actual_X[:, -1]),
        "cr_natparam_last_X_expected_signature": _column_signature(cr_expected_X[:, -1]),
        "cr_natparam_last_P_actual_signature": _column_signature(cr_actual_P[:, -1]),
        "cr_natparam_last_P_expected_signature": _column_signature(cr_expected_P[:, -1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument(
        "--with-numerical-outer",
        action="store_true",
        help="Also compute numerical outer gradient/Hessian diagnostics.",
    )
    parser.add_argument(
        "--with-preopt",
        action="store_true",
        help="Also compare exact preoptimization Sl block factors to mgcv.",
    )
    args = parser.parse_args()

    cases = _case_table()
    if args.case_id not in cases:
        raise SystemExit(
            f"Unknown case_id {args.case_id!r}. Available keys: {sorted(cases)}"
        )

    (
        case_id,
        family,
        formula,
        data_factory,
        method,
        _pred_atol,
        _se_atol,
        _check_response_se,
    ) = cases[args.case_id]
    del _pred_atol, _se_atol, _check_response_se

    select = "select_true" in case_id
    data = data_factory()
    snapshot = _run_mgcv_snapshot(data, formula, family, method, select=select)
    sp = np.asarray(snapshot["fit"]["smoothing_params"], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp, select=select)
    y = gam.family.validate_y(gam.y_)
    log_sp = np.log(sp)
    run = run_general_family_fixed_smoothing(
        gam,
        y,
        sp,
        weights=gam.prior_weights_,
        deriv=2,
        score_type=method,
    )
    fit = run["fit"]
    setup = run["setup"]

    coef_full = np.asarray(
        sl_initial_repara(
            setup.Sl,
            np.asarray(fit["coef"], dtype=np.float64),
            inverse=True,
            both_sides=False,
            cov=False,
        ),
        dtype=np.float64,
    )
    eta = _fit5_linear_predictors(setup, fit, run["offset_list"])
    fitted = np.asarray(gam.family.predict(eta=eta), dtype=np.float64)
    db_drho_full = np.column_stack(
        [
            np.asarray(
                sl_initial_repara(
                    setup.Sl,
                    np.asarray(fit["db_drho"], dtype=np.float64)[:, i],
                    inverse=True,
                    both_sides=False,
                    cov=False,
                ),
                dtype=np.float64,
            )
            for i in range(np.asarray(fit["db_drho"], dtype=np.float64).shape[1])
        ]
    )
    mgcv_fit5 = _run_mgcv_fit5_fixed_sp(
        data,
        formula,
        family,
        sp,
        score_type=method,
    )

    mgcv_fit5_score = _as_float_or_none(mgcv_fit5.get("score"))
    mgcv_fit5_loglik = _as_float_or_none(mgcv_fit5.get("loglik"))
    mgcv_fit5_ldetHp = _as_float_or_none(mgcv_fit5.get("ldetHp"))
    hp_diag = _hp_diagnostics(fit)
    family_state = {}
    if family == "gammals":
        th = np.asarray(gam.family.linfo[1].linkinv(eta[:, 1]), dtype=np.float64)
        family_state = {
            "th_min": float(np.min(th)),
            "th_max": float(np.max(th)),
            "th_margin_to_b_min": float(np.min(th - gam.family.b)),
            "etat_min": float(np.min(eta[:, 1])),
            "etat_max": float(np.max(eta[:, 1])),
        }
    numerical_outer = {}
    if args.with_numerical_outer:
        num_grad = np.asarray(
            criterion_gradient_numerical(gam, y, log_sp, method=method.lower()),
            dtype=np.float64,
        )
        num_hess = np.asarray(
            criterion_hessian_numerical(gam, y, log_sp, method=method.lower()),
            dtype=np.float64,
        )
        numerical_outer = {
            "num_score1_max_abs_diff_vs_snapshot": _try_max_abs_diff(
                num_grad,
                snapshot["fit"]["outer_grad"],
            ),
            "num_score2_max_abs_diff_vs_snapshot": _try_max_abs_diff(
                num_hess,
                snapshot["fit"]["outer_hess"],
            ),
        }
    preopt_diag = {}
    if args.with_preopt:
        preopt = _run_mgcv_general_preoptimization(
            data, formula, family, method, select=select
        )
        actual_blocks = list(setup.sl)
        expected_blocks = list(preopt["Sl"]["blocks"])
        block_reports = []
        for idx, (ab, eb) in enumerate(zip(actual_blocks, expected_blocks)):
            block_reports.append(
                {
                    "index": idx,
                    "start": int(ab.start),
                    "stop": int(ab.stop),
                    "repara": bool(ab.repara),
                    "linear": bool(ab.linear),
                    "D_max_abs_diff": _try_max_abs_diff(
                        np.asarray(ab.D, dtype=np.float64)
                        if ab.D is not None
                        else np.array([], dtype=np.float64),
                        np.asarray(eb.get("D", []), dtype=np.float64),
                    ),
                    "Di_max_abs_diff": _try_max_abs_diff(
                        np.asarray(ab.Di, dtype=np.float64)
                        if ab.Di is not None
                        else np.array([], dtype=np.float64),
                        np.asarray(eb.get("Di", []), dtype=np.float64),
                    ),
                }
            )
        preopt_diag = {"sl_block_reports": block_reports}
        preopt_diag["X_full_max_abs_diff"] = _try_max_abs_diff(
            np.asarray(setup.X_full, dtype=np.float64),
            np.asarray(preopt["X_full"], dtype=np.float64),
        )
        preopt_diag["X_initial_max_abs_diff"] = _try_max_abs_diff(
            np.asarray(setup.X_initial, dtype=np.float64),
            np.asarray(preopt["X_initial"], dtype=np.float64),
        )
    lpmatrix_diag = _maybe_t2_lpmatrix_diagnostics(case_id, family, formula, data, gam)

    report = {
        "case_id": case_id,
        "family": family,
        "method": method,
        "fit_rank": int(fit["rank"]),
        "fit_iter": int(fit["iter"]),
        "fit_warn": list(fit.get("warn", []) or []),
        "fit_coef_len": int(np.asarray(fit["coef"], dtype=np.float64).size),
        "full_coef_len": int(coef_full.size),
        "n_smoothing_params": int(sp.size),
        "n_bdrop": int(np.sum(np.asarray(fit["bdrop"], dtype=bool))),
        "fit_score": float(fit["score"]),
        "snapshot_score": float(snapshot["fit"]["criterion_value"]),
        "mgcv_fit5_score": mgcv_fit5_score,
        "fit_score_diff_vs_snapshot": float(
            fit["score"] - float(snapshot["fit"]["criterion_value"])
        ),
        "fit_score_diff_vs_mgcv_fit5": (
            None
            if mgcv_fit5_score is None
            else float(fit["score"] - mgcv_fit5_score)
        ),
        "ll_diff_vs_mgcv_fit5": (
            None if mgcv_fit5_loglik is None else float(fit["l"] - mgcv_fit5_loglik)
        ),
        "ldetHp_diff_vs_mgcv_fit5": (
            None if mgcv_fit5_ldetHp is None else float(fit["ldetHp"] - mgcv_fit5_ldetHp)
        ),
        "mgcv_fit5_score_raw": mgcv_fit5.get("score"),
        "mgcv_fit5_loglik_raw": mgcv_fit5.get("loglik"),
        "mgcv_fit5_ldetHp_raw": mgcv_fit5.get("ldetHp"),
        "hp_logdet_direct": hp_diag["hp_logdet_direct"],
        "hp_logdet_diff_vs_factor": (
            None
            if hp_diag["hp_logdet_direct"] is None
            else float(hp_diag["hp_logdet_direct"] - fit["ldetHp"])
        ),
        "hp_min_eval": hp_diag["hp_min_eval"],
        "hp_max_eval": hp_diag["hp_max_eval"],
        "hp_rank_pos": hp_diag["hp_rank_pos"],
        "hp_cond_abs": hp_diag["hp_cond_abs"],
        **family_state,
        "coef_full_max_abs_diff": _max_abs_diff(
            coef_full,
            mgcv_fit5["coefficients_full"],
        ),
        "lbb_max_abs_diff": _try_max_abs_diff(
            fit["lbb"],
            mgcv_fit5.get("lbb", []),
        ),
        "hp_max_abs_diff": _try_max_abs_diff(
            -np.asarray(fit["lbb"], dtype=np.float64)
            + np.asarray(fit["St_full"], dtype=np.float64),
            -np.asarray(mgcv_fit5.get("lbb", []), dtype=np.float64)
            + np.asarray(fit["St_full"], dtype=np.float64),
        ),
        "eta_max_abs_diff": _max_abs_diff(eta, mgcv_fit5["linear_predictors"]),
        "fitted_max_abs_diff": _max_abs_diff(fitted, mgcv_fit5["fitted_values"]),
        "db_drho_full_max_abs_diff": _max_abs_diff(
            db_drho_full,
            mgcv_fit5["db_drho_full"],
        ),
        "score1_max_abs_diff_vs_snapshot": _max_abs_diff(
            fit["score1"],
            snapshot["fit"]["outer_grad"],
        ),
        "score1_max_abs_diff_vs_mgcv_fit5": _try_max_abs_diff(
            fit["score1"],
            mgcv_fit5["score1"],
        ),
        "score2_max_abs_diff_vs_snapshot": _max_abs_diff(
            fit["score2"],
            snapshot["fit"]["outer_hess"],
        ),
        "score2_max_abs_diff_vs_mgcv_fit5": _try_max_abs_diff(
            fit["score2"],
            mgcv_fit5["score2"],
        ),
        "score1_actual": np.asarray(fit["score1"], dtype=np.float64).tolist(),
        "score1_expected": np.asarray(mgcv_fit5["score1"], dtype=np.float64).tolist(),
        "score2_actual": np.asarray(fit["score2"], dtype=np.float64).tolist(),
        "score2_expected": np.asarray(mgcv_fit5["score2"], dtype=np.float64).tolist(),
        **numerical_outer,
        **preopt_diag,
        **lpmatrix_diag,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
