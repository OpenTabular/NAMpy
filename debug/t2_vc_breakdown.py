"""Compare NAMpy and mgcv t2(full=FALSE) unconditional covariance pieces."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nampy.gam.fit.backends import solve_gaussian_given_smoothing
from nampy.gam.fit.postprocess.unconditional_covariance import (
    _mgcv_vcorr,
    _restore_pirls_dbeta_to_original_parameterization,
    _restore_pirls_rank_root_to_original_parameterization,
)
from nampy.gam.linalg import symmetrize_matrix
from nampy.gam.smoothing_selection.criteria.gaussian_dyn import (
    criterion_hessian_ml_reml_gaussian_dynamic_joint,
)
from nampy.gam.smoothing_selection.criteria.pirls.derivatives import _gdi1_kernel
from tests.parity.test_mgcv_prediction_inference_diagnostics_parity import (
    CASE_BY_ID,
    _case_outer_bundle,
    _newdata_for_case,
)
from tests.mgcv_parity_utils import _run_mgcv_predict_on_newdata

ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = ROOT / "debug" / "mgcv_t2_vc_breakdown.R"


def _max(name: str, actual, expected) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    print(
        f"{name:34s}", actual.shape, expected.shape, np.max(np.abs(actual - expected))
    )


def _best_column_signs(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    signs = np.ones(actual.shape[1], dtype=np.float64)
    for j in range(actual.shape[1]):
        if np.linalg.norm(-actual[:, j] - expected[:, j]) < np.linalg.norm(
            actual[:, j] - expected[:, j]
        ):
            signs[j] = -1.0
    return signs


def _run_r(data, formula: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "data.csv"
        json_path = Path(tmp) / "out.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            ["Rscript", str(R_SCRIPT), str(csv_path), str(json_path), formula],
            check=True,
            cwd=str(ROOT),
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _nampy_vc1(gam, sp, hess):
    sol = solve_gaussian_given_smoothing(gam, gam.y_, sp)
    kernel = _gdi1_kernel(gam, gam.y_, sol, sp, method="REML")
    db_drho = np.column_stack(
        [
            _restore_pirls_dbeta_to_original_parameterization(kernel.current, db)
            for db in kernel.ift.dbeta
        ]
    )
    evals, evecs = np.linalg.eigh(np.asarray(hess, dtype=np.float64))
    pos = evals > 0.0
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[pos] = 1.0 / np.sqrt(evals[pos])
    rV = (inv_sqrt[:, None] * evecs.T)[:, : db_drho.shape[1]]
    Vc1 = (rV @ db_drho.T).T @ (rV @ db_drho.T)
    return db_drho, Vc1


def _nampy_vb(gam):
    fit_result = gam.fit_core_solution_.fit_result
    fit_state = gam.fit_core_solution_.fit_state
    scale = float(fit_result.scale)
    rank_root = getattr(fit_state, "rank_root", None)
    if rank_root is None:
        return None
    root = _restore_pirls_rank_root_to_original_parameterization(
        fit_state, np.asarray(rank_root, dtype=np.float64)
    )
    return scale * (root @ root.T)


def main() -> None:
    case = CASE_BY_ID["gaussian_t2_full_false"]
    data, expected, gam = _case_outer_bundle(case.case_id)
    r = _run_r(data, case.formula)
    sp = np.asarray(r["sp"], dtype=np.float64)
    P_sign = _best_column_signs(
        np.asarray(gam.predict(X=None, type="lpmatrix"), dtype=np.float64),
        np.asarray(r["X"], dtype=np.float64),
    )
    D = np.diag(P_sign)

    print("sign flips vs mgcv fit X", np.flatnonzero(P_sign < 0).tolist())
    public_train = np.asarray(gam.predict(X=None, type="lpmatrix"), dtype=np.float64)
    r_public_train = np.asarray(r["final_lpmatrix_train"], dtype=np.float64)
    public_sign = _best_column_signs(public_train, r_public_train)
    print("public train flips vs mgcv public", np.flatnonzero(public_sign < 0).tolist())
    _max("public train raw", public_train, r_public_train)
    _max("public train sign-aligned", public_train * public_sign, r_public_train)
    _max("mgcv public train vs fit X", r_public_train, r["X"])
    mgcv_public_fit_sign = _best_column_signs(r_public_train, np.asarray(r["X"]))
    print(
        "mgcv public train flips vs fit X",
        np.flatnonzero(mgcv_public_fit_sign < 0).tolist(),
    )
    _max("stored Vc raw", gam.vcov(unconditional=True), r["final_Vc"])
    _max("stored Vc sign-aligned", D @ gam.vcov(unconditional=True) @ D, r["final_Vc"])
    _max("R X root", gam.fit_core_solution_.fit_state.X, r["X"])
    _max("R X root sign-aligned", gam.fit_core_solution_.fit_state.X * P_sign, r["X"])

    joint_log_sigma2 = getattr(gam._optim_result, "joint_log_sigma2", None)
    hess = criterion_hessian_ml_reml_gaussian_dynamic_joint(
        gam,
        gam.y_,
        np.log(sp),
        float(joint_log_sigma2),
        method="REML",
    )
    _max("outer hess", hess, r["hess"])

    db_drho, Vc1 = _nampy_vc1(gam, sp, hess)
    _max("db.drho raw", db_drho, r["db_drho"])
    _max("db.drho sign-aligned", db_drho * P_sign[:, None], r["db_drho"])
    _max("Vc1 raw", Vc1, r["Vc1"])
    _max("Vc1 sign-aligned", D @ Vc1 @ D, r["Vc1"])

    Vb = _nampy_vb(gam)
    if Vb is not None:
        _max("Vb raw", Vb, r["Vb"])
        _max("Vb sign-aligned", D @ Vb @ D, r["Vb"])

    stored = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
    vc2_est = stored - np.asarray(gam.fit_core_solution_.fit_result.cov_bayes)
    print("stored Vc eigen min", np.linalg.eigvalsh(stored).min())
    newdata = _newdata_for_case(case.case_id)
    Xp = np.asarray(gam.predict(X=newdata, type="lpmatrix"), dtype=np.float64)
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xp, stored, Xp))
    r_link = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        case.formula,
        family=case.family,
        method="REML",
        type="link",
        return_se=True,
        unconditional=True,
    )
    r_lpmat = np.asarray(
        _run_mgcv_predict_on_newdata(
            data,
            newdata,
            case.formula,
            family=case.family,
            method="REML",
            type="lpmatrix",
        )["pred"],
        dtype=np.float64,
    )
    _max("newdata public X", Xp, r_lpmat)
    _max("newdata se direct", se, r_link["se"])
    pred, pred_se = gam.predict(X=newdata, type="link", return_se=True, cov=stored)
    _max("newdata se via predict", pred_se, r_link["se"])
    print("nampy stored SE range", float(se.min()), float(se.max()))


if __name__ == "__main__":
    main()
