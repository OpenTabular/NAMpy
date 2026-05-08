"""Probe gaulss outer Newton trace against mgcv."""

from __future__ import annotations

# ruff: noqa: E402, I001

import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg.lapack import get_lapack_funcs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nampy.gam.smoothing_selection.optimize.basics import (
    _initial_smoothing_params_mgcv_style,
)
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gaulss_data,
)
from tests.mgcv_parity_utils import _run_mgcv_snapshot
from tests.optimization.test_mgcv_general_family_preoptimization_parity import (
    _run_mgcv_general_preoptimization,
)
from tests.optimization.test_mgcv_outer_optimization_parity import _run_mgcv_outer_trace
from tests.optimization.test_mgcv_postprocessing_final_fit_parity import (
    _fit_general_case,
    _nampy_optimizer_name,
)
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
)
from nampy.gam.smoothing_selection.reparam import build_estimate_gam_setup_state
from nampy.gam.linalg.norms import r_matrix_norm_max_abs


def main() -> None:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "gaulss_cr"
    select = case_id == "gaulss_select_true_cr"
    case = (case_id, "gaulss", GAULSS_FORMULA, _gaulss_data, "ML", 5e-6, 5e-6, True)
    data = _gaulss_data()
    expected = _run_mgcv_snapshot(
        data=data,
        formula=GAULSS_FORMULA,
        family="gaulss",
        method="ML",
        select=select,
    )
    preopt = _run_mgcv_general_preoptimization(
        data,
        GAULSS_FORMULA,
        "gaulss",
        "ML",
        select=select,
    )
    expected_trace = _run_mgcv_outer_trace(
        data,
        str(GAULSS_FORMULA),
        "gaulss",
        "ML",
        "newton",
        select=select,
    )
    optimizer = _nampy_optimizer_name(expected)
    _data, gam, warnings = _fit_general_case(case, optimizer=optimizer)
    init = _initial_smoothing_params_mgcv_style(gam, gam.y_)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        data.to_csv(csv_path, index=False)
        r_init = subprocess.run(
            [
                "Rscript",
                "debug/mgcv_general_initial_spg.R",
                str(csv_path),
                str(GAULSS_FORMULA),
                "gaulss",
                "ML",
                "true" if select else "false",
            ],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()

    print("warnings", warnings)
    print("optimizer", optimizer)
    print("our init sp", np.asarray(init, dtype=np.float64))
    print("mgcv initial.spg", r_init)
    print("preopt sp", np.asarray(preopt["smoothing_params"], dtype=np.float64))
    print("actual final sp", np.asarray(gam.smoothing_params, dtype=np.float64))
    print("expected final sp", np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64))
    print("actual outer", gam._optim_result.outer_info)
    print("expected outer", expected["fit"]["outer_info"])
    print("expected trace")
    for row in expected_trace["trace"]:
        print(row)
    print("actual trace")
    for row in getattr(gam._optim_result, "optim_trace", []) or []:
        print(row)

    fit5_setup = build_general_family_setup_state(gam, np.ones_like(np.asarray(init)))
    exact_setup = build_estimate_gam_setup_state(gam)
    X = np.asarray(fit5_setup.X_initial, dtype=np.float64)
    E = np.asarray(exact_setup.Eb, dtype=np.float64)
    start = gam.family.initialize(
        gam.y_,
        X,
        fit5_setup.jj,
        offset=fit5_setup.offset_list,
        weights=np.ones_like(gam.y_),
        E=E,
    )
    lbb = np.asarray(
        gam.family.ll(
            gam.y_,
            X,
            fit5_setup.jj,
            start,
            np.ones_like(gam.y_),
            offset=fit5_setup.offset_list,
            deriv=1,
        )["lbb"],
        dtype=np.float64,
    )
    print("lambda variants")
    for i, S_i in enumerate(exact_setup.S):
        S_i = np.asarray(S_i, dtype=np.float64)
        start_i = int(exact_setup.off[i]) - 1
        stop_i = start_i + S_i.shape[0]
        block_lbb = np.asarray(lbb[start_i:stop_i, start_i:stop_i], dtype=np.float64)
        rank_i = int(exact_setup.rank[i])
        pivots = {}
        pstrf = get_lapack_funcs("pstrf", dtype=np.float64)
        _R, piv, _rank_p, _info = pstrf(S_i.copy(), lower=0)
        pivots["pstrf"] = np.asarray(piv, dtype=int).ravel() - 1
        _Q, _R_qr, piv_qr = scipy_qr(S_i, pivoting=True, check_finite=False)
        pivots["qr"] = np.asarray(piv_qr, dtype=int)
        pivots["natural"] = np.arange(S_i.shape[0], dtype=int)
        for name, pivv in pivots.items():
            Z = np.asarray(S_i[:, pivv[:rank_i]], dtype=np.float64)
            zn = float(np.max(np.sum(np.abs(Z), axis=0)))
            Z = Z / zn
            lam = 0.3 * r_matrix_norm_max_abs(-Z.T @ block_lbb @ Z) / r_matrix_norm_max_abs(
                Z.T @ S_i @ Z
            )
            print(i, name, lam, pivv[:rank_i])


if __name__ == "__main__":
    main()
