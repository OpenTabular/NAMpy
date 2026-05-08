from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nampy.gam._model_state import _compiled_model, _fit_state, _term_blocks_seq
from nampy.gam.smooths.categorical.fs import FSmoothInteractionTerm
from tests.gam_cartesian_matrix import (
    SPECIAL_TERMS,
    MatrixCase,
    family_entry_spec,
    formula_for_method,
    make_data,
)
from tests.mgcv_parity_utils import _run_mgcv_snapshot
from tests.mgcv_parity_utils import (
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_penalties,
)
from tests.gam_cartesian_matrix import fit_model
from tests.mgcv_invariant_policy import canonicalize_raw_representation_state
from tests.smooths.test_mgcv_raw_constructor_parity import (
    _build_runtime_term,
    _serialize_term_raw,
)
from tests.mgcv_parity_utils import _run_mgcv_raw_constructor


def _fs_default_case() -> MatrixCase:
    term = next(item for item in SPECIAL_TERMS if item[0] == "fs_default")
    family = ("gaussian_identity", "gaussian", "gaussian")
    name, rhs, sp_text = term
    return MatrixCase(
        case_id=f"diagnostic_{name}_{family[0]}_fixed",
        formula=formula_for_method(rhs, "fixed", sp_text),
        family=family_entry_spec(family),
        method="fixed",
        data_kind=family[2],
    )


def _max_abs(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.ndim == 2 and bb.ndim == 1 and bb.size == aa.size:
        bb = bb.reshape(aa.shape, order="F")
    if bb.ndim == 2 and aa.ndim == 1 and aa.size == bb.size:
        aa = aa.reshape(bb.shape, order="F")
    return float(np.max(np.abs(aa - bb)))


def _nat_type1_variant(B0, S0, rank, *, qr_kind: str, eig_driver: str, triangle: str):
    if qr_kind == "numpy":
        Q, R = np.linalg.qr(B0, mode="reduced")
    elif qr_kind == "scipy":
        Q, R = scipy_qr(B0, mode="economic", pivoting=False)
    elif qr_kind in {"linpack_fwd", "linpack_rev"}:
        Q, R = _linpack_qr_no_pivot(B0, reverse_q=(qr_kind == "linpack_rev"))
    else:
        raise ValueError(qr_kind)
    tmp = solve_triangular(R.T, S0.T, lower=True, check_finite=False)
    RSR = solve_triangular(R.T, tmp.T, lower=True, check_finite=False)
    if triangle == "sym":
        RSR = 0.5 * (RSR + RSR.T)
        lower = True
    elif triangle == "lower":
        lower = True
    elif triangle == "upper":
        lower = False
    else:
        raise ValueError(triangle)
    evals, U = scipy_eigh(RSR, driver=eig_driver, lower=lower, check_finite=False)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    U = U[:, order]
    D = evals[:rank].copy()
    Xn = Q @ U
    P = solve_triangular(R, U, lower=False, check_finite=False)
    E = np.ones(Xn.shape[1], dtype=np.float64)
    E[:rank] = np.sqrt(D)
    Xn = Xn / E[np.newaxis, :]
    P = P / E[np.newaxis, :]
    scale = 1.0 / np.sqrt(np.mean(Xn[:, :rank] ** 2))
    Xn[:, :rank] *= scale
    P[:, :rank] *= scale
    if rank < Xn.shape[1]:
        scalef = 1.0 / np.sqrt(np.mean(Xn[:, rank:] ** 2))
        Xn[:, rank:] *= scalef
        P[:, rank:] *= scalef
    return Xn, P


def _linpack_qr_no_pivot(A, *, reverse_q: bool):
    x = np.array(A, dtype=np.float64, order="F", copy=True)
    n, p = x.shape
    k = min(n, p)
    qraux = np.zeros(k, dtype=np.float64)
    for l in range(k):
        nrmxl = float(np.linalg.norm(x[l:, l]))
        if nrmxl == 0.0:
            continue
        if x[l, l] != 0.0:
            nrmxl = float(np.copysign(nrmxl, x[l, l]))
        x[l:, l] /= nrmxl
        x[l, l] = 1.0 + x[l, l]
        for j in range(l + 1, p):
            t = -float(np.dot(x[l:, l], x[l:, j])) / float(x[l, l])
            x[l:, j] += t * x[l:, l]
        qraux[l] = x[l, l]
        x[l, l] = -nrmxl

    R = np.triu(x[:k, :p])
    q_full = np.eye(n, dtype=np.float64, order="F")
    indices = range(k - 1, -1, -1) if reverse_q else range(k)
    for l in indices:
        if qraux[l] == 0.0:
            continue
        temp = float(x[l, l])
        x[l, l] = qraux[l]
        for col in range(n):
            t = -float(np.dot(x[l:, l], q_full[l:, col])) / float(x[l, l])
            q_full[l:, col] += t * x[l:, l]
        x[l, l] = temp
    return np.asarray(q_full[:, :p], dtype=np.float64), np.asarray(R[:p, :], dtype=np.float64)


def main() -> None:
    case = _fs_default_case()
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
    original_align = FSmoothInteractionTerm._align_multivariate_base_reparameterization

    def _identity_align(self, X, X_reparam, P_coef, *, range_rank, null_dim):
        return X_reparam, P_coef

    FSmoothInteractionTerm._align_multivariate_base_reparameterization = _identity_align
    try:
        gam_no_align = fit_model(case, data)
        no_align = gam_no_align.parity_snapshot(X=data, include_covariances=True)
        print("no-align public loglik", gam_no_align.loglik())
        print(
            "no-align pred max_abs response",
            _max_abs(no_align["predictions"]["response"], expected["predictions"]["response"]),
        )
        print(
            "no-align edf_total",
            no_align["fit"].get("edf_total"),
            "expected",
            expected["fit"].get("edf_total"),
        )
    finally:
        FSmoothInteractionTerm._align_multivariate_base_reparameterization = original_align

    print("formula", case.formula)
    print("public loglik", gam.loglik())
    print("public aic", gam.aic())
    for key in (
        "loglik",
        "aic",
        "edf_total",
        "edf2",
        "scale",
        "rss",
        "deviance",
        "penalty_quadratic",
        "smoothing_params",
    ):
        print("fit", key, "actual", actual["fit"].get(key), "expected", expected["fit"].get(key))

    for key in ("response", "link", "lpmatrix"):
        print("pred max_abs", key, _max_abs(actual["predictions"][key], expected["predictions"][key]))
    lpa = np.asarray(actual["predictions"]["lpmatrix"], dtype=np.float64)
    lpe_flat = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)
    if lpe_flat.ndim == 1:
        print(
            "lpmatrix max_abs orderC",
            float(np.max(np.abs(lpa - lpe_flat.reshape(lpa.shape, order="C")))),
        )
        print(
            "lpmatrix max_abs orderF",
            float(np.max(np.abs(lpa - lpe_flat.reshape(lpa.shape, order="F")))),
        )

    coef_a = np.asarray(actual["fit"]["coef_full"], dtype=np.float64)
    coef_e = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)
    print("coef shape", coef_a.shape, coef_e.shape, "max_abs", _max_abs(coef_a, coef_e))

    compiled = _compiled_model(gam)
    fit_state = _fit_state(gam)
    print("compiled n_coef", getattr(compiled, "n_coef", None))
    print("fit X shape", None if fit_state is None else np.asarray(fit_state.X).shape)
    for i, tb in enumerate(_term_blocks_seq(gam)):
        print(
            "term",
            i,
            getattr(tb, "label", None),
            getattr(tb, "basis_name", None),
            getattr(tb, "coef_slice", None),
            np.asarray(getattr(tb, "basis_train", np.empty((0, 0)))).shape,
        )
    smooth_expr = 's(f, x0, bs="fs", k=6, sp=c(1.0,1.2,1.4))'
    sm_x = np.asarray(_run_mgcv_smoothcon_matrix(data, smooth_expr)["X"], dtype=np.float64)
    term = next(tb for tb in _term_blocks_seq(gam) if getattr(tb, "basis_name", "") == "fs")
    bx = np.asarray(term.basis_train, dtype=np.float64)
    print("smoothCon X shape", sm_x.shape, "local", bx.shape)
    print("smoothCon X max_abs direct", _max_abs(bx, sm_x))
    pen_expected = _run_mgcv_smoothcon_penalties(
        data,
        smooth_expr,
        absorb_cons=True,
        scale_penalty=True,
    )["S"]
    compiled = _compiled_model(gam)
    local_pens = [
        np.asarray(p.matrix, dtype=np.float64)
        for p in getattr(compiled, "compiled_penalties", ())
        if getattr(p, "label", None) == term.label
    ]
    for idx, (pa, pe) in enumerate(zip(local_pens, pen_expected)):
        print("penalty", idx, "shape", pa.shape, np.asarray(pe).shape, "max_abs", _max_abs(pa, pe))

    raw_term, raw_x, _ = _build_runtime_term(data, case.formula)
    raw_actual_exact = _serialize_term_raw(raw_term, raw_x)
    raw_expected_exact = _run_mgcv_raw_constructor(data, smooth_expr)
    print("raw exact flev actual", raw_actual_exact["extra"]["flev"])
    print("raw exact flev expected", raw_expected_exact["extra"]["flev"])
    for key in ("X",):
        print("raw exact", key, "max_abs", _max_abs(raw_actual_exact[key], raw_expected_exact[key]))
    for key in ("Xb", "P"):
        print(
            "raw exact extra",
            key,
            "max_abs",
            _max_abs(raw_actual_exact["extra"][key], raw_expected_exact["extra"][key]),
        )
    for idx, (pa, pe) in enumerate(zip(raw_actual_exact["S"], raw_expected_exact["S"])):
        print("raw exact penalty", idx, "max_abs", _max_abs(pa, pe))
    xa = np.asarray(raw_actual_exact["extra"]["Xb"], dtype=np.float64)
    xe = np.asarray(raw_expected_exact["extra"]["Xb"], dtype=np.float64)
    print("raw exact Xb corr")
    print(np.round(np.corrcoef(xa.T, xe.T)[: xa.shape[1], xa.shape[1] :], 6))
    pa = np.asarray(raw_actual_exact["extra"]["P"], dtype=np.float64)
    pe = np.asarray(raw_expected_exact["extra"]["P"], dtype=np.float64)
    transform = np.linalg.lstsq(pa, pe, rcond=None)[0]
    print("P actual-to-expected transform")
    print(np.round(transform, 6))
    base_term = raw_term._base_term
    B0 = np.asarray(raw_term._base_constructor_predict_matrix(raw_x), dtype=np.float64)
    S0 = np.asarray(base_term.penalties[0], dtype=np.float64)
    rank = int(raw_term._range_rank)
    for qr_kind in ("numpy", "scipy", "linpack_fwd", "linpack_rev"):
        for eig_driver in ("evr", "evd", "evx", "ev"):
            for triangle in ("sym", "lower", "upper"):
                try:
                    xv, pv = _nat_type1_variant(
                        B0,
                        S0,
                        rank,
                        qr_kind=qr_kind,
                        eig_driver=eig_driver,
                        triangle=triangle,
                    )
                except Exception as exc:
                    print("variant", qr_kind, eig_driver, triangle, "error", repr(exc))
                    continue
                print(
                    "variant",
                    qr_kind,
                    eig_driver,
                    triangle,
                    "Xb max_abs",
                    _max_abs(xv, xe),
                    "P max_abs",
                    _max_abs(pv, pe),
                )
        if False:
            try:
                xv, pv = _nat_type1_variant(
                    B0,
                    S0,
                    rank,
                    qr_kind=qr_kind,
                    eig_driver=eig_driver,
                    triangle="sym",
                )
            except Exception as exc:
                print("variant", qr_kind, eig_driver, "error", repr(exc))
                continue
            print(
                "variant",
                qr_kind,
                eig_driver,
                "Xb max_abs",
                _max_abs(xv, xe),
                "P max_abs",
                _max_abs(pv, pe),
            )
    x_metric = np.asarray(data["x0"], dtype=np.float64)
    target = np.column_stack(
        [x_metric - np.mean(x_metric), np.ones_like(x_metric)]
    )
    print("target-to-expected-null")
    print(np.round(np.linalg.lstsq(target, xe[:, rank:], rcond=None)[0], 6))
    print("target-to-actual-null")
    print(np.round(np.linalg.lstsq(target, xa[:, rank:], rcond=None)[0], 6))

    raw_actual = canonicalize_raw_representation_state(raw_actual_exact)
    raw_expected = canonicalize_raw_representation_state(raw_expected_exact)
    print("raw invariant flev actual", raw_actual["extra"]["flev"])
    print("raw invariant flev expected", raw_expected["extra"]["flev"])
    for key in ("X",):
        print("raw", key, "max_abs", _max_abs(raw_actual[key], raw_expected[key]))
    for key in ("Xb", "P"):
        print(
            "raw extra",
            key,
            "max_abs",
            _max_abs(raw_actual["extra"][key], raw_expected["extra"][key]),
        )
    for idx, (pa, pe) in enumerate(zip(raw_actual["S"], raw_expected["S"])):
        print("raw penalty", idx, "max_abs", _max_abs(pa, pe))


if __name__ == "__main__":
    main()
