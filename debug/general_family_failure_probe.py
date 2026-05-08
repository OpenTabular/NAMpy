from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.linalg import qr as scipy_qr

from nampy.gam._model_state import _coef_full, _term_blocks_seq
from nampy.gam.diagnostics.concurvity import _concurvity_measures, _qr_R
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gaulss_data,
    _gaulss_tensor_data,
    _general_newdata,
    _gevlss_two_smooth_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
)


def _print_lpmatrix_probe() -> None:
    data = _gaulss_tensor_data()
    newdata = _general_newdata(data)
    formula = ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1"]
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaulss",
        method="ML",
        type="lpmatrix",
    )
    actual = np.asarray(gam.predict(newdata, type="lpmatrix"), dtype=np.float64)
    target = np.asarray(expected["pred"], dtype=np.float64)
    col_norm = np.linalg.norm(actual - target, axis=0)
    sign_norm = np.linalg.norm(actual + target, axis=0)
    sign_flip_cols = np.flatnonzero(sign_norm < col_norm)
    print("t2 lpmatrix max_abs", float(np.max(np.abs(actual - target))))
    print("t2 lpmatrix sign_flip_cols", sign_flip_cols.tolist())
    print(
        "t2 lpmatrix signed_max_abs",
        float(
            np.max(np.abs(actual * np.where(sign_norm < col_norm, -1.0, 1.0) - target))
        ),
    )


def _print_unconditional_probe() -> None:
    data = _gaulss_data()
    newdata = _general_newdata(data)
    gam = _fit_nampy_model(
        data,
        GAULSS_FORMULA,
        "gaulss",
        "ML",
        select=True,
    )
    snap = _run_mgcv_snapshot(
        data,
        GAULSS_FORMULA,
        "gaulss",
        "ML",
        select=True,
    )
    fixed = _fit_nampy_model_fixed_sp(
        data,
        GAULSS_FORMULA,
        "gaulss",
        np.asarray(snap["fit"]["smoothing_params"], dtype=np.float64),
        select=True,
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        GAULSS_FORMULA,
        family="gaulss",
        method="ML",
        type="link",
        return_se=True,
        unconditional=True,
        select=True,
    )
    pred, se = gam.predict(
        newdata,
        type="link",
        return_se=True,
        cov=gam.vcov(unconditional=True),
    )
    print(
        "select Vc link pred max_abs",
        float(np.max(np.abs(np.asarray(pred) - np.asarray(expected["pred"])))),
    )
    print(
        "select Vc link se max_abs",
        float(np.max(np.abs(np.asarray(se) - np.asarray(expected["se"])))),
    )
    print("select log_sp", np.log(np.asarray(gam.smoothing_params)).tolist())
    print(
        "select mgcv log_sp",
        np.asarray(snap["fit"]["log_smoothing_params"], dtype=np.float64).tolist(),
    )
    print(
        "select Vc max_abs",
        float(
            np.max(
                np.abs(
                    gam.vcov(unconditional=True)
                    - np.asarray(snap["fit"]["cov_unconditional"], dtype=np.float64)
                )
            )
        ),
    )
    _, fixed_se = fixed.predict(
        newdata,
        type="link",
        return_se=True,
        cov=fixed.vcov(unconditional=True),
    )
    print(
        "select fixed-at-mgcv-sp se max_abs",
        float(np.max(np.abs(np.asarray(fixed_se) - np.asarray(expected["se"])))),
    )


def _print_concurvity_probe() -> None:
    data = _gevlss_two_smooth_data()
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1", "~ 1"]
    gam = _fit_nampy_model(data, formula, "gevlss", "ML")
    expected = _run_mgcv_snapshot(data, formula, "gevlss", "ML")
    csv_path = Path("/tmp/gevlss_two_cr_concurvity_probe.csv")
    data.to_csv(csv_path, index=False)
    r_out = subprocess.run(
        [
            "Rscript",
            "debug/general_family_concurvity_probe.R",
            str(csv_path),
            "gevlss",
            json.dumps(formula),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    print(r_out.stdout)
    print("gevlss_two_cr log_sp", np.log(np.asarray(gam.smoothing_params)).tolist())
    print(
        "gevlss_two_cr mgcv log_sp",
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64).tolist(),
    )
    print(
        "gevlss_two_cr score diff",
        float(gam.smoothing_score_ - float(expected["fit"]["criterion_value"])),
    )
    print(
        "gevlss_two_cr link max_abs",
        float(
            np.max(
                np.abs(
                    np.asarray(gam.predict(data, type="link"), dtype=np.float64).ravel(
                        order="F"
                    )
                    - np.asarray(expected["predictions"]["link"], dtype=np.float64)
                )
            )
        ),
    )
    print("term blocks")
    for tb in _term_blocks_seq(gam):
        print(str(tb.label), tb.coef_slice, str(tb.term_type))
    print("actual concurvity full")
    print(np.asarray(gam.concurvity(full=True)["values"], dtype=np.float64))
    print("expected concurvity full")
    print(
        np.asarray(
            expected["parity"]["diagnostics"]["concurvity_full"], dtype=np.float64
        )
    )
    print("actual labels", gam.concurvity(full=True)["labels"])
    print("expected labels", expected["parity"]["diagnostics"]["concurvity_labels"])
    X = np.asarray(gam.predict(type="lpmatrix"), dtype=np.float64)
    X_expected = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)
    print("training lpmatrix max_abs", float(np.max(np.abs(X - X_expected))))
    coef = np.asarray(_coef_full(gam), dtype=np.float64)
    order = np.array([0, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
    Xr = _qr_R(X[:, order])
    br = coef[order]
    blocks = [
        ("para", np.arange(0, 3, dtype=int)),
        ("s(x)", np.arange(3, 8, dtype=int)),
        ("s(z)", np.arange(8, 13, dtype=int)),
    ]
    out = np.zeros((3, 3), dtype=np.float64)
    for i, (_label, idx) in enumerate(blocks):
        keep = np.ones(Xr.shape[1], dtype=bool)
        keep[idx] = False
        out[:, i] = _concurvity_measures(Xr[:, keep], Xr[:, idx], br[idx])
    print("reordered concurvity full")
    print(out)
    X1 = _qr_R(X[:, :11])
    b1 = coef[:11]
    blocks1 = [
        ("para", np.arange(0, 1, dtype=int)),
        ("s(x)", np.arange(1, 6, dtype=int)),
        ("s(z)", np.arange(6, 11, dtype=int)),
    ]
    out1 = np.zeros((3, 3), dtype=np.float64)
    for i, (_label, idx) in enumerate(blocks1):
        keep = np.ones(X1.shape[1], dtype=bool)
        keep[idx] = False
        out1[:, i] = _concurvity_measures(X1[:, keep], X1[:, idx], b1[idx])
    print("first-predictor-only concurvity full")
    print(out1)

    def scipy_R(A):
        return scipy_qr(
            np.asarray(A, dtype=np.float64),
            mode="economic",
            pivoting=False,
            check_finite=False,
        )[1]

    Xs = scipy_R(X)
    outs = np.zeros((3, 3), dtype=np.float64)
    blocks0 = [
        ("para", np.arange(0, 1, dtype=int)),
        ("s(x)", np.arange(1, 6, dtype=int)),
        ("s(z)", np.arange(6, 11, dtype=int)),
    ]
    for i, (_label, idx) in enumerate(blocks0):
        keep = np.ones(Xs.shape[1], dtype=bool)
        keep[idx] = False
        Xi = Xs[:, keep]
        Xj = Xs[:, idx]
        r = Xi.shape[1]
        R = scipy_R(np.column_stack([Xi, Xj]))[:, r:]
        Rt = scipy_R(R)
        leading = np.asarray(R[:r, :], dtype=np.float64)
        beta = coef[idx]
        outs[:, i] = (
            np.linalg.svd(
                np.linalg.solve(Rt.T, leading.T),
                compute_uv=False,
            )[0]
            ** 2,
            np.sum((leading @ beta) ** 2) / np.sum((Rt @ beta) ** 2),
            np.sum(leading**2) / np.sum(R**2),
        )
    print("scipy concurvity full")
    print(outs)


if __name__ == "__main__":
    _print_lpmatrix_probe()
    _print_unconditional_probe()
    _print_concurvity_probe()
