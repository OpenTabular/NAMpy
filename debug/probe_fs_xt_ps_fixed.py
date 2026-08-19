from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.diagnostics.test_mgcv_diagnostics_cartesian_matrix import _make_case
from tests.gam_cartesian_matrix import FAMILIES, SPECIAL_TERMS, fit_model, make_data

from tests.mgcv_parity_utils import (
    _run_mgcv_raw_constructor,
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_penalties,
    _run_mgcv_snapshot,
)
from tests.smooths.test_mgcv_raw_constructor_parity import (
    _build_runtime_term,
    _serialize_term_raw,
)


def _max_abs(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.ndim == 2 and bb.ndim == 1 and aa.size == bb.size:
        bb = bb.reshape(aa.shape, order="F")
    if bb.ndim == 2 and aa.ndim == 1 and aa.size == bb.size:
        aa = aa.reshape(bb.shape, order="F")
    return float(np.max(np.abs(aa - bb)))


def main() -> None:
    term = next(item for item in SPECIAL_TERMS if item[0] == "fs_xt_ps")
    family = next(item for item in FAMILIES if item[0] == "gaussian_identity")
    case = _make_case(term, family, "fixed")
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
    print("loglik", gam.loglik(), expected["fit"]["loglik"])
    print("edf_total", actual["fit"]["edf_total"], expected["fit"]["edf_total"])
    print("deviance", actual["fit"]["deviance"], expected["fit"]["deviance"])
    print(
        "pred response max_abs",
        _max_abs(actual["predictions"]["response"], expected["predictions"]["response"]),
    )

    raw_term, raw_x, _ = _build_runtime_term(data, case.formula)
    raw_actual = _serialize_term_raw(raw_term, raw_x)
    smooth_expr = (
        's(f, x0, bs="fs", k=7, m=2, xt=list(bs="ps"), '
        "sp=c(1.0,1.2,1.4))"
    )
    raw_expected = _run_mgcv_raw_constructor(data, smooth_expr)
    print("raw class", raw_actual["class_name"], raw_expected["class_name"])
    print("raw flev", raw_actual["extra"]["flev"], raw_expected["extra"]["flev"])
    print("raw X max_abs", _max_abs(raw_actual["X"], raw_expected["X"]))
    print("raw Xb max_abs", _max_abs(raw_actual["extra"]["Xb"], raw_expected["extra"]["Xb"]))
    print("raw P max_abs", _max_abs(raw_actual["extra"]["P"], raw_expected["extra"]["P"]))
    for idx, (pa, pe) in enumerate(zip(raw_actual["S"], raw_expected["S"])):
        print("raw S", idx, _max_abs(pa, pe))

    sm_x = np.asarray(_run_mgcv_smoothcon_matrix(data, smooth_expr)["X"], dtype=np.float64)
    term_block = next(tb for tb in gam.gam_result_.compiled_model.compiled_terms if tb.basis_name == "fs")
    print("smoothCon X max_abs", _max_abs(term_block.basis_train, sm_x))
    sm_s = _run_mgcv_smoothcon_penalties(
        data,
        smooth_expr,
        absorb_cons=True,
        scale_penalty=True,
    )["S"]
    local_s = [
        p.matrix
        for p in gam.gam_result_.compiled_model.compiled_penalties
        if p.label == term_block.label
    ]
    for idx, (pa, pe) in enumerate(zip(local_s, sm_s)):
        print("smoothCon S", idx, _max_abs(pa, pe))


if __name__ == "__main__":
    main()
