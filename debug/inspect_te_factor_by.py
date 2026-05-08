from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.linalg import qr

from nampy.gam._model_state import (
    _cov_bayes,
    _cov_freq,
    _cov_unconditional,
    _edf2,
    _fit_scale,
    _fit_state,
    _H_coef,
    _summary_R,
    _term_blocks_seq,
    _penalty_blocks_seq,
)
from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data
from tests.mgcv_parity_utils import _run_mgcv_gam_setup_assembly, _run_mgcv_snapshot
from nampy.gam.fit.parameterization import prediction_parameterization_map
from nampy.gam.smoothing_selection.reparam import build_estimate_gam_setup_state


def _arr(value):
    return np.asarray(value, dtype=np.float64)


def main() -> None:
    case = MatrixCase(
        case_id="debug_te_factor_by_fixed",
        formula='y ~ f + te(x0, x1, by=f, bs=["cr","cr"], k=[5,5], sp=[1.0,1.2])',
        family="gaussian",
        method="fixed",
        data_kind="gaussian",
    )
    data = make_data(case.data_kind)
    gam = fit_model(case, data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.method,
        allow_live_run=True,
    )

    print("formula", case.formula)
    print("levels", sorted(set(data["f"])))
    for key in ("loglik", "aic", "edf_total", "deviance", "scale", "rss"):
        print(key, actual["fit"].get(key), expected["fit"].get(key))
    print("public loglik/aic", gam.loglik(), gam.aic())
    internal_edf2 = _edf2(gam)
    print(
        "internal edf2 sum",
        None if internal_edf2 is None else float(np.sum(_arr(internal_edf2))),
    )
    state = _fit_state(gam)
    scale = float(_fit_scale(gam))
    H = _H_coef(gam)
    if H is not None:
        H = _arr(H)
        edf1 = 2.0 * np.diag(H) - np.sum(H * H.T, axis=1)
        print("edf1 sum", float(np.sum(edf1)))
    R_summary = _summary_R(gam)
    if R_summary is not None:
        RTR = _arr(R_summary).T @ _arr(R_summary)
        for name, cov in (
            ("bayes", _cov_bayes(gam)),
            ("freq", _cov_freq(gam)),
            ("uncond", _cov_unconditional(gam)),
        ):
            if cov is None:
                continue
            edf2_alt = np.sum(_arr(cov) * RTR, axis=1) / scale
            print("edf2 via summary_R", name, float(np.sum(edf2_alt)))
            if H is not None and float(np.sum(edf2_alt)) > float(np.sum(edf1)):
                print("edf2 via summary_R capped", name, float(np.sum(edf1)))
        P = prediction_parameterization_map(gam)
        if P is not None and _cov_bayes(gam) is not None:
            P = _arr(P)
            cov_fit = np.linalg.solve(P, np.linalg.solve(P, _arr(_cov_bayes(gam)).T).T)
            edf2_fit_alt = np.sum(cov_fit * RTR, axis=1) / scale
            print("edf2 via summary_R fit cov", float(np.sum(edf2_fit_alt)))
    if state is not None and state.X is not None:
        X = _arr(state.X)
        w = state.fisher_weights
        if w is None:
            w = state.working_weights
        w = np.ones(X.shape[0], dtype=np.float64) if w is None else _arr(w).ravel()
        if w.size == 1:
            w = np.full(X.shape[0], float(w[0]), dtype=np.float64)
        _, R_unpiv = qr(np.sqrt(np.clip(w, 0.0, None))[:, None] * X, mode="economic")
        RTR_unpiv = _arr(R_unpiv).T @ _arr(R_unpiv)
        for name, cov in (("bayes", _cov_bayes(gam)), ("freq", _cov_freq(gam))):
            if cov is None:
                continue
            edf2_alt = np.sum(_arr(cov) * RTR_unpiv, axis=1) / scale
            print("edf2 via unpivoted qr", name, float(np.sum(edf2_alt)))
    setup = build_estimate_gam_setup_state(gam)
    setup_X = _arr(setup.X)
    expected_setup = _run_mgcv_gam_setup_assembly(
        data,
        case.formula,
        case.family,
        case.method,
        allow_live_run=True,
    )
    expected_X = _arr(expected_setup["X"])
    print(
        "setup X",
        setup_X.shape,
        expected_X.shape,
        "max_abs",
        float(np.max(np.abs(setup_X - expected_X))),
        "sum_abs",
        float(np.sum(np.abs(setup_X - expected_X))),
    )
    setup_blocks = [(0, 4), (4, 28), (28, 52), (52, 76), (76, 100)]
    for bi, (start, stop) in enumerate(setup_blocks):
        print(
            "setup X block",
            bi,
            float(np.max(np.abs(setup_X[:, start:stop] - expected_X[:, start:stop]))),
            float(np.sum(np.abs(setup_X[:, start:stop] - expected_X[:, start:stop]))),
        )
        row_l1 = np.sum(
            np.abs(setup_X[:, start:stop] - expected_X[:, start:stop]), axis=1
        )
        print("  row l1 head", row_l1[:12])
        print("  row l1 by f", data["f"].to_numpy()[:12])
    w_setup = state.fisher_weights if state is not None else None
    if w_setup is None and state is not None:
        w_setup = state.working_weights
    w_setup = (
        np.ones(setup_X.shape[0], dtype=np.float64)
        if w_setup is None
        else _arr(w_setup).ravel()
    )
    if w_setup.size == 1:
        w_setup = np.full(setup_X.shape[0], float(w_setup[0]), dtype=np.float64)
    _, R_setup_piv, pivot_setup = qr(
        np.sqrt(np.clip(w_setup, 0.0, None))[:, None] * setup_X,
        mode="economic",
        pivoting=True,
    )
    n_rows, n_cols = R_setup_piv.shape
    flat = np.concatenate(
        [
            np.asarray(R_setup_piv, dtype=np.float64, order="F").ravel(order="F"),
            np.zeros(n_cols * n_cols, dtype=np.float64),
        ]
    )
    R_setup_sq = np.zeros((n_cols, n_cols), dtype=np.float64)
    for j in range(n_cols):
        for i in range(min(n_cols, j + 1)):
            R_setup_sq[i, j] = flat[i + n_rows * j]
    R_setup_nat = np.zeros_like(R_setup_sq)
    R_setup_nat[:, np.asarray(pivot_setup, dtype=np.intp)] = R_setup_sq
    if _cov_bayes(gam) is not None:
        edf2_setup = np.sum(_arr(_cov_bayes(gam)) * (R_setup_nat.T @ R_setup_nat), axis=1) / scale
        print("edf2 via setup_X summary_R bayes", float(np.sum(edf2_setup)))
        x_path = Path("/tmp/nampy_te_factor_by_X.csv")
        cov_path = Path("/tmp/nampy_te_factor_by_cov.csv")
        data_path = Path("/tmp/nampy_te_factor_by_data.csv")
        r_x_bin = Path("/tmp/mgcv_te_factor_by_prefit_X.bin")
        r_r_bin = Path("/tmp/mgcv_te_factor_by_fit_R.bin")
        np.savetxt(x_path, setup_X, delimiter=",")
        np.savetxt(cov_path, _arr(_cov_bayes(gam)), delimiter=",")
        data.to_csv(data_path, index=False)
        r_script = Path(__file__).with_name("inspect_pqr_from_matrix.R")
        r_out = subprocess.check_output(
            [
                "Rscript",
                str(r_script),
                str(x_path),
                str(cov_path),
                str(scale),
                str(data_path),
                case.formula,
                str(r_x_bin),
                str(r_r_bin),
            ],
            text=True,
        )
        print("R pqr on local setup_X")
        print(r_out)
        if r_x_bin.exists():
            r_x = np.fromfile(r_x_bin, dtype="<f8").reshape(setup_X.shape, order="F")
            diff = setup_X - r_x
            idx_flat = int(np.argmax(np.abs(diff)))
            idx = np.unravel_index(idx_flat, diff.shape)
            print(
                "local setup_X vs R internal bin",
                "max_abs",
                float(np.max(np.abs(diff))),
                "sum_abs",
                float(np.sum(np.abs(diff))),
                "idx",
                idx,
                "local",
                float(setup_X[idx]),
                "R",
                float(r_x[idx]),
            )
            for bi, (start, stop) in enumerate(setup_blocks):
                block_diff = diff[:, start:stop]
                print(
                    "internal X block",
                    bi,
                    float(np.max(np.abs(block_diff))),
                    float(np.sum(np.abs(block_diff))),
                )
        if r_r_bin.exists():
            r_R = np.fromfile(r_r_bin, dtype="<f8").reshape((setup_X.shape[1], setup_X.shape[1]), order="F")
            local_R = R_setup_nat
            print(
                "local R vs R internal bin",
                float(np.max(np.abs(local_R - r_R))),
                float(np.sum(np.abs(local_R - r_R))),
            )
    for key in ("edf2",):
        av = actual["fit"].get(key)
        ev = expected["fit"].get(key)
        print(key, None if av is None else float(np.sum(_arr(av))), None if ev is None else float(np.sum(_arr(ev))))
        if ev is not None and R_summary is not None and _cov_bayes(gam) is not None:
            edf2_alt = np.sum(_arr(_cov_bayes(gam)) * (_arr(R_summary).T @ _arr(R_summary)), axis=1) / scale
            ee = _arr(ev)
            print("edf2 summary diff sum", float(np.sum(edf2_alt - ee)))
            idx = np.argsort(np.abs(edf2_alt - ee))[-12:]
            print("edf2 largest diffs", [(int(i), float(edf2_alt[i]), float(ee[i]), float(edf2_alt[i] - ee[i])) for i in idx])
    for key in ("cov_bayes", "cov_freq", "hat"):
        av = actual["fit"].get(key)
        ev = expected["fit"].get(key)
        if av is None or ev is None:
            print(key, av is None, ev is None)
            continue
        aa = _arr(av)
        ee = _arr(ev)
        if aa.shape != ee.shape and aa.size == ee.size:
            aa = aa.reshape(ee.shape, order="F")
        print(
            key,
            "shape",
            aa.shape,
            ee.shape,
            "max_abs",
            float(np.max(np.abs(aa - ee))),
            "sum_abs",
            float(np.sum(np.abs(aa - ee))),
        )
    for key in ("lpmatrix", "terms"):
        av = actual["predictions"].get(key)
        ev = expected["predictions"].get(key)
        if av is None or ev is None:
            continue
        aa = _arr(av)
        ee = _arr(ev)
        if aa.shape != ee.shape and aa.size == ee.size:
            aa = aa.reshape(ee.shape)
        print(
            key,
            aa.shape,
            ee.shape,
            "max_abs",
            float(np.max(np.abs(aa - ee))),
            "sum_abs",
            float(np.sum(np.abs(aa - ee))),
        )
        if key == "lpmatrix":
            blocks = [(0, 4), (4, 28), (28, 52), (52, 76), (76, 100)]
            for bi, (start, stop) in enumerate(blocks):
                print(
                    "lpmatrix block",
                    bi,
                    float(np.max(np.abs(aa[:, start:stop] - ee[:, start:stop]))),
                    float(np.sum(np.abs(aa[:, start:stop] - ee[:, start:stop]))),
                )
    asti = actual["parity"]["diagnostics"].get("smooth_test_inputs", None)
    esti = expected["parity"]["diagnostics"].get("smooth_test_inputs", None)
    if asti is not None and esti is not None:
        for i, (ar, er) in enumerate(zip(asti["r_blocks"], esti["r_blocks"])):
            aa = _arr(ar)
            ee = _arr(er)
            print(
                "r_block",
                i,
                aa.shape,
                ee.shape,
                "max_abs",
                float(np.max(np.abs(aa - ee))),
                "sum_abs",
                float(np.sum(np.abs(aa - ee))),
            )
    print("sp", actual["fit"].get("smoothing_params"), expected["fit"].get("smoothing_params"))
    print(
        "pred max abs",
        np.max(np.abs(_arr(actual["predictions"]["link"]) - _arr(expected["predictions"]["link"]))),
    )
    print("pred head actual", _arr(actual["predictions"]["link"])[:8])
    print("pred head expected", _arr(expected["predictions"]["link"])[:8])
    print("term_results expected", expected["fit"].get("term_results"))
    print("term_results actual", actual["fit"].get("term_results"))
    print("side reports", actual["fit"].get("side_condition_reports"))

    print("term blocks")
    for tb in _term_blocks_seq(gam):
        print(
            tb.label,
            tb.term_type,
            tb.basis_train.shape,
            tb.coef_slice,
            getattr(tb, "basis_name", None),
            "edf",
            getattr(tb, "edf", None),
        )
        print("  metadata", getattr(tb, "metadata", None))

    print("penalty blocks")
    for pb in _penalty_blocks_seq(gam):
        print(
            pb.smoothing_index,
            pb.coef_slice,
            pb.matrix.shape,
            float(np.linalg.norm(pb.matrix, ord=1)),
            getattr(pb, "metadata", None),
        )


if __name__ == "__main__":
    main()
