from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data

from nampy.gam.diagnostics.concurvity import _term_indices_for_concurvity
from nampy.gam.model_state import _coef_full, _term_blocks_seq
from nampy.gam.predict.linear_predictor_matrix import build_lpmatrix
from tests.mgcv_parity_utils import _run_mgcv_snapshot


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

    actual_pair = actual["parity"]["diagnostics"]["concurvity_pairwise"]
    expected_pair = expected["parity"]["diagnostics"]["concurvity_pairwise"]
    print("actual labels", actual_pair["labels"])
    print("expected labels", expected_pair["labels"])
    for name in ("worst", "observed", "estimate"):
        aa = _arr(actual_pair[name])
        ee = _arr(expected_pair[name])
        print(name, "max_abs", float(np.max(np.abs(aa - ee))))
        print("actual")
        print(aa)
        print("expected")
        print(ee)

    X = _arr(build_lpmatrix(gam))
    X = X[np.sum(np.isnan(X), axis=1) == 0, :]
    expected_X = _arr(expected["predictions"]["lpmatrix"])
    print(
        "lpmatrix vs mgcv",
        float(np.max(np.abs(X - expected_X))),
        float(np.sum(np.abs(X - expected_X))),
    )
    print("lpmatrix shape", X.shape)
    print("coef len", len(_arr(_coef_full(gam))))
    print("blocks")
    for label, idx in _term_indices_for_concurvity(gam, X.shape[1]):
        print(label, int(idx[0]), int(idx[-1]), len(idx))
    print("term blocks")
    for tb in _term_blocks_seq(gam):
        print(tb.label, tb.term_type, tb.coef_slice)

    def run_r_concurvity(matrix: np.ndarray, label: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            x_path = tmp_path / "X.csv"
            coef_path = tmp_path / "coef.csv"
            starts_path = tmp_path / "starts.csv"
            stops_path = tmp_path / "stops.csv"
            out_path = tmp_path / "out.json"
            np.savetxt(x_path, matrix, delimiter=",")
            np.savetxt(coef_path, _arr(_coef_full(gam)), delimiter=",")
            starts = []
            stops = []
            for _label, idx in _term_indices_for_concurvity(gam, X.shape[1]):
                starts.append(int(idx[0]) + 1)
                stops.append(int(idx[-1]) + 1)
            np.savetxt(
                starts_path, np.asarray(starts, dtype=int), fmt="%d", delimiter=","
            )
            np.savetxt(
                stops_path, np.asarray(stops, dtype=int), fmt="%d", delimiter=","
            )
            subprocess.run(
                [
                    "Rscript",
                    str(Path(__file__).with_name("probe_concurvity_from_matrix.R")),
                    str(x_path),
                    str(coef_path),
                    str(starts_path),
                    str(stops_path),
                    str(out_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            r_pair = json.loads(out_path.read_text(encoding="utf-8"))
            for name in ("worst", "observed", "estimate"):
                rr = _arr(r_pair[name])
                ee = _arr(expected_pair[name])
                print(label, name, "max_abs", float(np.max(np.abs(rr - ee))))
                print(rr)

    run_r_concurvity(X, "R on local X")
    run_r_concurvity(expected_X, "R on expected X")

    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        x_path = tmp_path / "X.csv"
        coef_path = tmp_path / "coef.csv"
        starts_path = tmp_path / "starts.csv"
        stops_path = tmp_path / "stops.csv"
        out_path = tmp_path / "out.json"
        np.savetxt(x_path, X, delimiter=",")
        np.savetxt(coef_path, _arr(_coef_full(gam)), delimiter=",")
        starts = []
        stops = []
        for _label, idx in _term_indices_for_concurvity(gam, X.shape[1]):
            starts.append(int(idx[0]) + 1)
            stops.append(int(idx[-1]) + 1)
        np.savetxt(starts_path, np.asarray(starts, dtype=int), fmt="%d", delimiter=",")
        np.savetxt(stops_path, np.asarray(stops, dtype=int), fmt="%d", delimiter=",")
        subprocess.run(
            [
                "Rscript",
                str(Path(__file__).with_name("probe_concurvity_from_matrix.R")),
                str(x_path),
                str(coef_path),
                str(starts_path),
                str(stops_path),
                str(out_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        r_pair = json.loads(out_path.read_text(encoding="utf-8"))
        for name in ("worst", "observed", "estimate"):
            rr = _arr(r_pair[name])
            ee = _arr(expected_pair[name])
            print("R on local X", name, "max_abs", float(np.max(np.abs(rr - ee))))
            print(rr)
    """


if __name__ == "__main__":
    main()
