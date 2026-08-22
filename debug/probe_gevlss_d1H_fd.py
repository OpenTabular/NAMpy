from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.fit.solvers.general_family_solver import (
    run_general_family_fixed_smoothing,
)
from nampy.gam.fit.solvers.general_newton_solver import (
    _sl_ldetS,
    _sl_mult,
    _sl_second_mult,
    chol_solve_pivoted,
)

from tests.families.test_general_family_mgcv_parity import GENERAL_SE_CASES
from tests.mgcv_parity_utils import _fit_nampy_model_fixed_sp, _run_mgcv_snapshot


def _case_table():
    return {case[0]: case for case in GENERAL_SE_CASES}


def _max_abs_diff(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"Shape mismatch: {aa.shape} != {bb.shape}")
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def main() -> None:
    case_id = "gevlss_t2_full_true"
    (
        _case_id,
        family_name,
        formula,
        data_factory,
        method,
        _pred_atol,
        _se_atol,
        _check_response_se,
    ) = _case_table()[case_id]
    del _pred_atol, _se_atol, _check_response_se

    data = data_factory()
    snapshot = _run_mgcv_snapshot(data, formula, family_name, method, select=False)
    sp = np.asarray(snapshot["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, family_name, sp, select=False)
    y = gam.family.validate_y(gam.y_)

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

    X = np.asarray(setup.X_initial, dtype=np.float64)
    jj = [np.asarray(j, dtype=int) for j in setup.jj]
    coef = np.asarray(fit["coef_fit_space"], dtype=np.float64)
    d1b = np.asarray(fit["db_drho_fit_space"], dtype=np.float64)
    weights = np.asarray(gam.prior_weights_, dtype=np.float64)
    offset = run["offset_list"]

    ll_d3 = gam.family.ll(
        y,
        X,
        jj,
        coef,
        weights,
        offset=offset,
        deriv=3,
        d1b=d1b,
    )
    d1H = [np.asarray(mat, dtype=np.float64) for mat in ll_d3["d1H"]]
    L = np.asarray(fit["L"], dtype=np.float64)
    D = np.asarray(fit["D"], dtype=np.float64)
    piv = np.asarray(fit["piv"], dtype=int)
    ipiv = np.asarray(fit["ipiv"], dtype=int)
    keep = ~np.asarray(fit["bdrop"], dtype=bool)
    Hp_inv = D[:, None] * chol_solve_pivoted(L, np.diag(D), piv=piv, ipiv=ipiv)

    log_sp = np.log(sp)
    rp_state = _sl_ldetS(
        setup.Sl,
        rho=log_sp,
        fixed=np.zeros_like(log_sp, dtype=bool),
        np_=coef.size,
        root=True,
        Stot=True,
        deriv=2,
    )
    sl_current = rp_state["Sl"]

    analytic_d2b = np.zeros((d1b.shape[0], d1b.shape[1] * (d1b.shape[1] + 1) // 2))
    pair_to_kk = {}
    kk = 0
    for i in range(d1b.shape[1]):
        for j in range(i, d1b.shape[1]):
            pair_to_kk[(i, j)] = kk
            dH_i_v = np.asarray(d1H[i], dtype=np.float64)[
                : d1b.shape[0], : d1b.shape[0]
            ] @ d1b[:, j]
            d2s_beta = np.asarray(
                _sl_second_mult(sl_current, coef, i + 1, j + 1, full=True),
                dtype=np.float64,
            )[keep]
            v = (
                -dH_i_v
                + np.asarray(_sl_mult(sl_current, d1b[:, j], i + 1), dtype=np.float64)[
                    keep
                ]
                + np.asarray(_sl_mult(sl_current, d1b[:, i], j + 1), dtype=np.float64)[
                    keep
                ]
                + d2s_beta
            )
            analytic_d2b[:, kk] = -D * chol_solve_pivoted(L, D * v, piv=piv, ipiv=ipiv)
            kk += 1

    ll_r = gam.family.ll(
        y,
        X,
        jj,
        coef,
        weights,
        offset=offset,
        deriv=4,
        d1b=d1b,
        d2b=analytic_d2b,
        fh=L,
        D=D,
    )
    analytic_trHid2H = np.asarray(ll_r["trHid2H"], dtype=np.float64)

    d2b_reports = []
    kk = 0
    eps_rho = 1e-6
    for i in range(d1b.shape[1]):
        for j in range(i, d1b.shape[1]):
            log_plus = log_sp.copy()
            log_minus = log_sp.copy()
            log_plus[j] += eps_rho
            log_minus[j] -= eps_rho
            run_plus = run_general_family_fixed_smoothing(
                gam,
                y,
                np.exp(log_plus),
                weights=gam.prior_weights_,
                deriv=1,
                score_type=method,
            )
            run_minus = run_general_family_fixed_smoothing(
                gam,
                y,
                np.exp(log_minus),
                weights=gam.prior_weights_,
                deriv=1,
                score_type=method,
            )
            db_plus = np.asarray(run_plus["fit"]["db_drho_fit_space"], dtype=np.float64)
            db_minus = np.asarray(run_minus["fit"]["db_drho_fit_space"], dtype=np.float64)
            fd_col = (db_plus[:, i] - db_minus[:, i]) / (2.0 * eps_rho)
            d2b_reports.append(
                {
                    "pair": [i, j],
                    "d2b_max_abs_diff_vs_fd": _max_abs_diff(analytic_d2b[:, kk], fd_col),
                }
            )
            kk += 1

    tr_reports = []
    kk = 0
    for i in range(d1b.shape[1]):
        for j in range(i, d1b.shape[1]):
            coef_plus = coef + eps_rho * d1b[:, j]
            coef_minus = coef - eps_rho * d1b[:, j]
            d1b_plus = d1b.copy()
            d1b_minus = d1b.copy()
            for q in range(d1b.shape[1]):
                qj = (q, j) if q <= j else (j, q)
                d1b_plus[:, q] += eps_rho * analytic_d2b[:, pair_to_kk[qj]]
                d1b_minus[:, q] -= eps_rho * analytic_d2b[:, pair_to_kk[qj]]
            d1H_plus = gam.family.ll(
                y,
                X,
                jj,
                coef_plus,
                weights,
                offset=offset,
                deriv=3,
                d1b=d1b_plus,
            )["d1H"]
            d1H_minus = gam.family.ll(
                y,
                X,
                jj,
                coef_minus,
                weights,
                offset=offset,
                deriv=3,
                d1b=d1b_minus,
            )["d1H"]
            trace_plus = float(
                np.trace(Hp_inv @ np.asarray(d1H_plus[i], dtype=np.float64))
            )
            trace_minus = float(
                np.trace(Hp_inv @ np.asarray(d1H_minus[i], dtype=np.float64))
            )
            fd_trace = (trace_plus - trace_minus) / (2.0 * eps_rho)
            tr_reports.append(
                {
                    "pair": [i, j],
                    "trHid2H_abs_diff_vs_fd": float(
                        abs(analytic_trHid2H[kk] - fd_trace)
                    ),
                    "analytic_trHid2H": float(analytic_trHid2H[kk]),
                    "fd_trHid2H": float(fd_trace),
                }
            )
            kk += 1

    eps = 1e-6
    reports = []
    for i in range(d1b.shape[1]):
        coef_plus = coef + eps * d1b[:, i]
        coef_minus = coef - eps * d1b[:, i]
        lbb_plus = np.asarray(
            gam.family.ll(
                y,
                X,
                jj,
                coef_plus,
                weights,
                offset=offset,
                deriv=1,
            )["lbb"],
            dtype=np.float64,
        )
        lbb_minus = np.asarray(
            gam.family.ll(
                y,
                X,
                jj,
                coef_minus,
                weights,
                offset=offset,
                deriv=1,
            )["lbb"],
            dtype=np.float64,
        )
        fd = (lbb_plus - lbb_minus) / (2.0 * eps)
        reports.append(
            {
                "sp_index": i,
                "d1H_max_abs_diff_vs_fd": _max_abs_diff(d1H[i], fd),
                "d1H_trace": float(np.trace(d1H[i])),
                "fd_trace": float(np.trace(fd)),
            }
        )

    print(
        json.dumps(
            {
                "case_id": case_id,
                "d1H_reports": reports,
                "d2b_reports": d2b_reports,
                "trHid2H_reports": tr_reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
