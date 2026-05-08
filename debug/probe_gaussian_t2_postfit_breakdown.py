"""Compare Gaussian t2(full=False) gam.fit3 postfit components with mgcv."""

from __future__ import annotations

# ruff: noqa: E402, I001

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from nampy.gam.fit.parameterization import prediction_parameterization_map
from nampy.gam.fit.backends import solve_gaussian_given_smoothing
from nampy.gam.fit.postprocess.unconditional_covariance import (
    _restore_pirls_dbeta_to_original_parameterization,
)
from nampy.gam.smoothing_selection.criteria.pirls.derivatives import _gdi1_kernel
from nampy.gam.smoothing_selection.reparam import build_estimate_gam_setup_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.optimization.test_mgcv_postprocessing_final_fit_parity import (
    ORDINARY_CASES,
    _fit_requested_case,
    _nampy_optimizer_name,
    _run_mgcv_snapshot,
)
from tests.mgcv_parity_utils import (
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_predict_matrix,
)


def main() -> None:
    case = next(c for c in ORDINARY_CASES if c.case_id == "gaussian_t2_full_false")
    data = case.data_factory()
    expected = _run_mgcv_snapshot(
        data=data,
        formula=case.formula,
        family=case.family,
        method="REML",
        select=case.select,
        weights_column=case.weights_column,
    )
    optimizer = _nampy_optimizer_name(expected)
    _, gam, _warnings = _fit_requested_case(case, method="REML", optimizer=optimizer)
    term = gam.compiled_model_.compiled_terms[0]
    expected_smooth = _run_mgcv_smoothcon_predict_matrix(
        data[["x1", "x2"]],
        data[["x1", "x2"]],
        't2(x1, x2, bs=["cr", "cr"], k=[8, 8], full=False)',
        absorb_cons=True,
        scale_penalty=True,
    )
    expected_fit_smooth = _run_mgcv_smoothcon_matrix(
        data[["x1", "x2"]],
        't2(x1, x2, bs=c("cr", "cr"), k=c(8, 8), full=FALSE)',
    )
    actual_basis = np.asarray(term.basis_train, dtype=np.float64)
    expected_basis = np.asarray(expected_smooth["X"], dtype=np.float64)
    expected_fit_basis = np.asarray(expected_fit_smooth["X"], dtype=np.float64)
    actual_prediction_basis = term.prediction_parameterization_matrix(
        data[["x1", "x2"]].to_numpy(dtype=np.float64)
    )
    print("basis shapes", actual_basis.shape, expected_basis.shape)
    print("fit basis max abs", np.max(np.abs(actual_basis - expected_fit_basis)))
    print(
        "fit basis projector max abs",
        np.max(
            np.abs(
                actual_basis @ np.linalg.pinv(actual_basis)
                - expected_fit_basis @ np.linalg.pinv(expected_fit_basis)
            )
        ),
    )
    print(
        "prediction basis max abs",
        np.max(np.abs(actual_prediction_basis - expected_basis)),
    )
    print(
        "prediction basis projector max abs",
        np.max(
            np.abs(
                actual_prediction_basis @ np.linalg.pinv(actual_prediction_basis)
                - expected_basis @ np.linalg.pinv(expected_basis)
            )
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        csv_path = tmp / "data.csv"
        json_path = tmp / "mgcv.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                "Rscript",
                str(ROOT / "debug" / "mgcv_t2_vc_breakdown.R"),
                str(csv_path),
                str(json_path),
                case.formula,
            ],
            check=True,
        )
        mgcv = json.loads(json_path.read_text())

    fit_result = gam.fit_core_solution_.fit_result
    actual_vc = np.asarray(fit_result.cov_unconditional, dtype=np.float64)
    p_map = prediction_parameterization_map(gam)
    transformed_actual_vc = (
        None if p_map is None else np.asarray(p_map @ actual_vc @ p_map.T, dtype=np.float64)
    )
    actual_edf2 = np.asarray(fit_result.edf2, dtype=np.float64)
    expected_vc = np.asarray(expected["fit"]["cov_unconditional"], dtype=np.float64)
    expected_edf2 = np.asarray(expected["fit"]["edf2"], dtype=np.float64)
    actual_hess = np.asarray(gam._optim_result.outer_info["hess"], dtype=np.float64)
    expected_hess = np.asarray(expected["fit"]["outer_info"]["hess"], dtype=np.float64)
    outer_info = dict(getattr(gam._optim_result, "outer_info", {}) or {})
    debug_vc = np.asarray(mgcv["Vc"], dtype=np.float64)
    debug_final_vc = np.asarray(mgcv["final_Vc"], dtype=np.float64)
    mgcv_p_map = np.asarray(mgcv["G_P"], dtype=np.float64)
    kernel = _gdi1_kernel(
        gam,
        gam.y_,
        solve_gaussian_given_smoothing(gam, gam.y_, gam.smoothing_params),
        gam.smoothing_params,
        method="REML",
    )
    sol_for_probe = solve_gaussian_given_smoothing(gam, gam.y_, gam.smoothing_params)
    setup_probe = build_estimate_gam_setup_state(gam)
    print(
        "sol X vs setup X max abs",
        np.max(
            np.abs(
                np.asarray(sol_for_probe["X"], dtype=np.float64)
                - np.asarray(setup_probe.X, dtype=np.float64)
            )
        ),
    )
    actual_db = np.column_stack(
        [
            _restore_pirls_dbeta_to_original_parameterization(
                kernel.current,
                kernel.ift.dbeta[j],
            )
            for j in range(len(kernel.ift.dbeta))
        ]
    )
    pivot1 = np.asarray(kernel.current.pivot1, dtype=np.int64)
    dropped = np.asarray(kernel.current.dropped_column_indices, dtype=np.int64)
    T = np.asarray(kernel.current.canonical.T, dtype=np.float64)
    from nampy.gam.fit.linalg.matrix_reindexing import (
        permute_rows,
        restore_dropped_rows,
    )

    raw_cols = []
    for col in kernel.ift.dbeta:
        unpermuted = permute_rows(np.asarray(col, dtype=np.float64)[:, None], pivot1, reverse=True)
        raw_cols.append(restore_dropped_rows(unpermuted, int(T.shape[1]), dropped).ravel())
    raw_db = np.column_stack(raw_cols)
    variants = {
        "T_raw": T @ raw_db,
        "Tt_raw": T.T @ raw_db,
    }
    try:
        variants["solve_T_raw"] = np.linalg.solve(T, raw_db)
    except Exception:
        pass
    try:
        variants["solve_Tt_raw"] = np.linalg.solve(T.T, raw_db)
    except Exception:
        pass
    setup = build_estimate_gam_setup_state(gam)
    beta = np.asarray(gam.fit_core_solution_.fit_result.coef_full, dtype=np.float64)
    p_full = int(beta.size)
    S_blocks_full = []
    for S_local, off_i in zip(list(setup.S), np.asarray(setup.off, dtype=np.int64)):
        S_local = np.asarray(S_local, dtype=np.float64)
        S_full = np.zeros((p_full, p_full), dtype=np.float64)
        start = int(off_i) - 1
        stop = start + int(S_local.shape[0])
        S_full[start:stop, start:stop] = S_local
        S_blocks_full.append(S_full)
    sp = np.asarray(gam.smoothing_params, dtype=np.float64)
    rho = np.log(sp)
    if setup.L is None:
        P_derivs = [sp[i] * S_blocks_full[i] for i in range(len(sp))]
    else:
        L = np.asarray(setup.L, dtype=np.float64)
        lam = np.exp(L @ rho + np.asarray(setup.lsp0, dtype=np.float64)[: L.shape[0]])
        P_derivs = []
        for j in range(L.shape[1]):
            Pj = np.zeros((p_full, p_full), dtype=np.float64)
            for i, Si in enumerate(S_blocks_full):
                Pj += float(lam[i]) * float(L[i, j]) * Si
            P_derivs.append(Pj)
    A_inv = np.asarray(gam.fit_core_solution_.fit_state.A_inv, dtype=np.float64)
    db_direct = np.column_stack([-A_inv @ (Pj @ beta) for Pj in P_derivs])
    debug_db = np.asarray(mgcv["db_drho"], dtype=np.float64)
    fit3_coef = np.asarray(mgcv["fit3_coef"], dtype=np.float64)
    final_coef = np.asarray(mgcv["final_coef"], dtype=np.float64)
    actual_coef = np.asarray(gam.fit_core_solution_.fit_result.coef_full, dtype=np.float64)

    print("sp max abs", np.max(np.abs(np.asarray(gam.smoothing_params) - mgcv["sp"])))
    print("sp actual", np.asarray(gam.smoothing_params, dtype=np.float64))
    print("sp mgcv", np.asarray(mgcv["sp"], dtype=np.float64))
    print("Vc actual-vs-snapshot max abs", np.max(np.abs(actual_vc - expected_vc)))
    if transformed_actual_vc is not None:
        print(
            "Vc P-actual-Pt-vs-snapshot max abs",
            np.max(np.abs(transformed_actual_vc - expected_vc)),
        )
    print("Vc debug-vs-snapshot max abs", np.max(np.abs(debug_vc - expected_vc)))
    print(
        "Vc P-debug-Pt-vs-snapshot max abs",
        np.max(np.abs(p_map @ debug_vc @ p_map.T - expected_vc)),
    )
    print("Vc debug-final-vs-snapshot max abs", np.max(np.abs(debug_final_vc - expected_vc)))
    print("P map max abs", np.max(np.abs(p_map - mgcv_p_map)))
    print("edf2 sum actual", float(np.sum(actual_edf2)))
    print("edf2 sum expected", float(np.sum(expected_edf2)))
    print("edf2 max abs", np.max(np.abs(actual_edf2 - expected_edf2)))
    print("outer hess max abs", np.max(np.abs(actual_hess - expected_hess)))
    print("outer has hess1", outer_info.get("hess1", None) is not None)
    print("outer has db_drho1", outer_info.get("db_drho1", None) is not None)
    print("outer has lsp1", outer_info.get("lsp1", None) is not None)
    print("coef actual-fit3 max abs", np.max(np.abs(actual_coef - fit3_coef)))
    print("coef actual-final max abs", np.max(np.abs(actual_coef - final_coef)))
    print("rank", int(kernel.current.penalized_system_rank))
    print("dropped", np.asarray(kernel.current.dropped_column_indices, dtype=np.int64))
    print("pivot1", np.asarray(kernel.current.pivot1, dtype=np.int64))
    print("db_drho shape actual", actual_db.shape, "mgcv", debug_db.shape)
    print("db_drho max abs", np.max(np.abs(actual_db - debug_db)))
    for name, val in variants.items():
        print(f"db variant {name} max abs", np.max(np.abs(val - debug_db)))
    print("db_direct max abs", np.max(np.abs(db_direct - debug_db)))
    diag_diff = np.diag(actual_vc) - np.diag(expected_vc)
    idx = int(np.argmax(np.abs(diag_diff)))
    print("worst Vc diag index", idx)
    print("worst Vc diag actual", float(np.diag(actual_vc)[idx]))
    print("worst Vc diag expected", float(np.diag(expected_vc)[idx]))


if __name__ == "__main__":
    main()
