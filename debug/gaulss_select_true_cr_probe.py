from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nampy.gam.fit.selection.optimize.basics import (
    _initial_smoothing_params_from_design,
)
from nampy.gam.fit.selection.reparam import build_estimate_gam_setup_state
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
    criterion_gradient_ml_reml_general_family,
    criterion_hessian_ml_reml_general_family,
    criterion_ml_reml_general_family,
    run_general_family_fixed_smoothing,
)
from nampy.gam.linalg.cholesky import safe_pivoted_cholesky
from tests.families.test_general_family_mgcv_parity import (
    GENERAL_SE_CASES,
    _gaulss_data,
)
from tests.mgcv_parity_utils import _run_mgcv_snapshot
from tests.optimization.test_mgcv_outer_optimization_parity import _run_mgcv_outer_trace
from tests.optimization.test_mgcv_postprocessing_final_fit_parity import (
    _fit_general_case,
    _nampy_optimizer_name,
)


def main() -> None:
    data = _gaulss_data()
    formula = ['y ~ s(x, bs="cr", k=6)', '~ 1']
    case = next(case for case in GENERAL_SE_CASES if case[0] == "gaulss_select_true_cr")
    script_path = Path("debug/mgcv_fixed_sp_fit5_select_true.R")
    expected = _run_mgcv_snapshot(
        data=data,
        formula=formula,
        family="gaulss",
        method="ML",
        select=True,
    )
    expected_trace = _run_mgcv_outer_trace(
        data=data,
        formula=str(formula),
        family="gaulss",
        method="ML",
        optimizer="newton",
        select=True,
    )
    expected_initial = None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "initial.json"
        data.to_csv(csv_path, index=False)
        try:
            subprocess.run(
                [
                    "Rscript",
                    "tests/parity/mgcv_initial_spg.R",
                    str(csv_path),
                    str(json_path),
                    str(formula),
                    "gaulss",
                    "ML",
                    "true",
                ],
                check=True,
                cwd=Path(".").resolve(),
                capture_output=True,
                text=True,
            )
            expected_initial = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("mgcv initial.spg probe failed:", repr(exc))
            if getattr(exc, "stdout", None):
                print("mgcv initial.spg stdout:", exc.stdout)
            if getattr(exc, "stderr", None):
                print("mgcv initial.spg stderr:", exc.stderr)
    optimizer = _nampy_optimizer_name(expected)
    _data, gam, fit_warnings = _fit_general_case(case, optimizer=optimizer)
    expected_sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    _fixed_data, fixed_gam, fixed_warnings = _fit_general_case(
        case, fixed_sp=expected_sp
    )
    expected_fit5 = None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "out.json"
        data.to_csv(csv_path, index=False)
        try:
            subprocess.run(
                [
                    "Rscript",
                    str(script_path),
                    str(csv_path),
                    str(json_path),
                    str(formula),
                    "gaulss",
                    json.dumps(np.asarray(gam.smoothing_params, dtype=np.float64).tolist()),
                    "ML",
                ],
                check=True,
                cwd=Path(".").resolve(),
                capture_output=True,
                text=True,
            )
            expected_fit5 = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("select fit5 probe failed:", repr(exc))
            if getattr(exc, "stdout", None):
                print("select fit5 stdout:", exc.stdout)
            if getattr(exc, "stderr", None):
                print("select fit5 stderr:", exc.stderr)
    exact = build_estimate_gam_setup_state(gam)
    fit = gam.gam_result_.fit_core_solution.fit_result
    optim = gam._optim_result
    run = run_general_family_fixed_smoothing(
        gam,
        gam.y_,
        np.asarray(gam.smoothing_params, dtype=np.float64),
        weights=None,
    )
    init_sp = np.asarray(_initial_smoothing_params_from_design(gam, gam.y_), dtype=np.float64)
    init_setup = build_general_family_setup_state(
        gam,
        np.ones_like(init_sp, dtype=np.float64),
        score_type="REML",
    )
    init_E = np.asarray(exact.Eb, dtype=np.float64)
    init_start = np.asarray(
        gam.family.initialize(
            gam.y_,
            init_setup.X_initial,
            init_setup.jj,
            offset=init_setup.offset_list,
            weights=np.ones_like(gam.y_, dtype=np.float64),
            E=init_E,
        ),
        dtype=np.float64,
    )
    init_lbb = np.asarray(
        gam.family.ll(
            gam.y_,
            init_setup.X_initial,
            init_setup.jj,
            init_start,
            np.ones_like(gam.y_, dtype=np.float64),
            offset=init_setup.offset_list,
            deriv=1,
        )["lbb"],
        dtype=np.float64,
    )
    init_fit = None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "out.json"
        data.to_csv(csv_path, index=False)
        try:
            subprocess.run(
                [
                    "Rscript",
                    str(script_path),
                    str(csv_path),
                    str(json_path),
                    str(formula),
                    "gaulss",
                    json.dumps(init_sp.tolist()),
                    "ML",
                ],
                check=True,
                cwd=Path(".").resolve(),
                capture_output=True,
                text=True,
            )
            init_fit = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("select init-sp probe failed:", repr(exc))
            if getattr(exc, "stdout", None):
                print("select init-sp stdout:", exc.stdout)
            if getattr(exc, "stderr", None):
                print("select init-sp stderr:", exc.stderr)

    print("L shape:", None if exact.L is None else exact.L.shape)
    print("lsp0 shape:", exact.lsp0.shape)
    print("lsp0:", np.asarray(exact.lsp0, dtype=np.float64))
    print(
        "helper initial sp:",
        init_sp,
    )
    print(
        "mgcv initial sp:",
        None
        if expected_initial is None
        else np.asarray(expected_initial["initial_sp"], dtype=np.float64),
    )
    if expected_initial is not None:
        print("mgcv init Sl blocks:", expected_initial.get("Sl_blocks", None))
        if expected_initial.get("Sl_blocks", None):
            exp_D = np.asarray(expected_initial["Sl_blocks"][0]["D"], dtype=np.float64)
            act_D = np.asarray(init_setup.Sl[0].D, dtype=np.float64)
            print("initial D actual:", act_D)
            print("initial D expected:", exp_D)
            print("initial D cross:", act_D.T @ exp_D)
            print("initial D max diff:", float(np.max(np.abs(act_D - exp_D))))
        print(
            "initial X max diff:",
            float(
                np.max(
                    np.abs(
                        np.asarray(init_setup.X_initial, dtype=np.float64)
                        - np.asarray(expected_initial["X_initial"], dtype=np.float64)
                    )
                )
            ),
        )
        print(
            "initial Eb gram max diff:",
            float(
                np.max(
                    np.abs(
                        init_E.T @ init_E
                        - np.asarray(expected_initial["Eb"], dtype=np.float64).T
                        @ np.asarray(expected_initial["Eb"], dtype=np.float64)
                    )
                )
            ),
        )
        print(
            "initial start max diff:",
            float(
                np.max(
                    np.abs(
                        init_start
                        - np.asarray(expected_initial["start"], dtype=np.float64)
                    )
                )
            ),
        )
        print(
            "initial lbb max diff:",
            float(
                np.max(
                    np.abs(
                        init_lbb - np.asarray(expected_initial["lbb"], dtype=np.float64)
                    )
                )
            ),
        )
    print("Sl len:", len(getattr(exact, "Sl", [])))
    print("init Sl len:", len(getattr(init_setup, "Sl", [])))
    for idx, block in enumerate(getattr(init_setup, "Sl", [])):
        print(
            "init Sl block",
            idx,
            "linear=",
            bool(getattr(block, "linear", True)),
            "repara=",
            bool(getattr(block, "repara", True)),
            "start=",
            getattr(block, "start", None),
            "stop=",
            getattr(block, "stop", None),
            "rank=",
            getattr(block, "rank", None),
            "nS=",
            len(getattr(block, "S", []) or []),
        )
    for idx, block in enumerate(getattr(exact, "Sl", [])):
        print(
            "Sl block",
            idx,
            "linear=",
            bool(getattr(block, "linear", True)),
            "start=",
            getattr(block, "start", None),
            "stop=",
            getattr(block, "stop", None),
            "n_sp=",
            getattr(block, "n_sp", None),
        )
    print("db_drho shape:", None if getattr(optim, "db_drho1", None) is None else np.asarray(optim.db_drho1).shape)
    print("hess:", None if getattr(optim, "hess", None) is None else np.asarray(optim.hess, dtype=np.float64))
    print("outer hess shape:", None if getattr(optim, "outer_info", None) is None else np.asarray(optim.outer_info.get("hess", None)).shape)
    print("outer_info hess:", None if getattr(optim, "outer_info", None) is None else np.asarray(optim.outer_info.get("hess", None), dtype=np.float64))
    print("optimizer outer_info keys:", sorted((getattr(optim, "outer_info", {}) or {}).keys()))
    print("snapshot outer_hess:", None if expected["fit"].get("outer_hess", None) is None else np.asarray(expected["fit"]["outer_hess"], dtype=np.float64))
    print("snapshot outer_info hess:", None if expected["fit"].get("outer_info", None) is None else np.asarray(expected["fit"]["outer_info"].get("hess", None), dtype=np.float64))
    print("snapshot outer_grad:", None if expected["fit"].get("outer_grad", None) is None else np.asarray(expected["fit"]["outer_grad"], dtype=np.float64))
    print("snapshot optimizer:", expected["fit"].get("optimizer", None))
    print(
        "mgcv trace rows:",
        [
            {
                "iter": row.get("iter", None),
                "log_sp": np.asarray(row.get("log_sp", []), dtype=np.float64).tolist(),
                "criterion": row.get("criterion", None),
                "gradient": (
                    None
                    if row.get("gradient", None) is None
                    else np.asarray(row.get("gradient"), dtype=np.float64).tolist()
                ),
                "converged_here": (row.get("rank_info", {}) or {}).get(
                    "converged_here", None
                ),
            }
            for row in expected_trace["trace"]
        ],
    )
    print("mgcv trace smoothing_params:", expected_trace["fit"]["smoothing_params"])
    print("mgcv trace outer_info:", expected_trace["fit"]["outer_info"])
    print("actual optimizer:", getattr(gam, "_optim_method", None))
    print(
        "actual trace rows:",
        [
            {
                "x": np.asarray(row.get("x", []), dtype=np.float64).tolist(),
                "fun": row.get("fun", None),
            }
            for row in getattr(gam, "_optim_trace", [])[:4]
        ],
    )
    print(
        "result optim trace:",
        [
            {
                "iter": row.get("iter", None),
                "log_sp": np.asarray(row.get("log_sp", []), dtype=np.float64).tolist(),
                "criterion": row.get("criterion", None),
            }
            for row in getattr(gam._optim_result, "optim_trace", [])[:4]
        ],
    )
    print("fit warnings:", fit_warnings)
    print("actual smoothing_params:", np.asarray(gam.smoothing_params, dtype=np.float64))
    print("snapshot smoothing_params:", None if expected["fit"].get("smoothing_params", None) is None else np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64))
    print(
        "init-sp mgcv REML:",
        None if init_fit is None else init_fit.get("REML", None),
    )
    print(
        "init-sp mgcv REML2:",
        None if init_fit is None else init_fit.get("REML2", None),
    )
    print(
        "our init-sp criterion:",
        float(
            criterion_ml_reml_general_family(
                gam,
                gam.y_,
                np.log(init_sp),
                "ML",
            )
        ),
    )
    print(
        "our init-sp grad:",
        np.asarray(
            criterion_gradient_ml_reml_general_family(
                gam,
                gam.y_,
                np.log(init_sp),
                "ML",
            ),
            dtype=np.float64,
        ),
    )
    print(
        "our init-sp hess:",
        np.asarray(
            criterion_hessian_ml_reml_general_family(
                gam,
                gam.y_,
                np.log(init_sp),
                "ML",
            ),
            dtype=np.float64,
        ),
    )
    init_grad = np.asarray(
        criterion_gradient_ml_reml_general_family(
            gam,
            gam.y_,
            np.log(init_sp),
            "ML",
        ),
        dtype=np.float64,
    )
    init_hess = np.asarray(
        criterion_hessian_ml_reml_general_family(
            gam,
            gam.y_,
            np.log(init_sp),
            "ML",
        ),
        dtype=np.float64,
    )
    print("init-sp raw Newton step:", np.linalg.solve(init_hess, -init_grad))
    print(
        "our grad at actual sp:",
        np.asarray(
            criterion_gradient_ml_reml_general_family(
                gam,
                gam.y_,
                np.log(np.asarray(gam.smoothing_params, dtype=np.float64)),
                "ML",
            ),
            dtype=np.float64,
        ),
    )
    if expected["fit"].get("smoothing_params", None) is not None:
        snap_sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        print(
            "our grad at snapshot sp:",
            np.asarray(
                criterion_gradient_ml_reml_general_family(gam, gam.y_, np.log(snap_sp), "ML"),
                dtype=np.float64,
            ),
        )
        print(
            "our hess at snapshot sp:",
            np.asarray(
                criterion_hessian_ml_reml_general_family(gam, gam.y_, np.log(snap_sp), "ML"),
                dtype=np.float64,
            ),
        )
    if expected_fit5 is not None:
        print(
            "mgcv initial_sp:",
            None
            if expected_fit5.get("initial_sp", None) is None
            else np.asarray(expected_fit5["initial_sp"], dtype=np.float64),
        )
        print(
            "fit5 db_drho shape:",
            None
            if expected_fit5.get("db_drho", None) is None
            else np.asarray(expected_fit5["db_drho"], dtype=np.float64).shape,
        )
        print("fit5 REML2:", expected_fit5.get("REML2", None))
        print(
            "fit5 db_drho:",
            None
            if expected_fit5.get("db_drho", None) is None
            else np.asarray(expected_fit5["db_drho"], dtype=np.float64),
        )
    print("snapshot fit keys:", sorted(expected["fit"].keys()))
    print("snapshot outer keys:", sorted(expected["fit"]["outer_info"].keys()) if expected["fit"].get("outer_info") else None)
    print("run fit keys:", sorted(run["fit"].keys()))
    print("run db_drho shape:", None if run["fit"].get("db_drho", None) is None else np.asarray(run["fit"]["db_drho"]).shape)
    print("run lbb shape:", None if run["fit"].get("lbb", None) is None else np.asarray(run["fit"]["lbb"]).shape)
    lbb = -np.asarray(run["fit"]["lbb"], dtype=np.float64)
    D = np.asarray(run["fit"]["D"], dtype=np.float64)
    p = int(np.sum(~np.asarray(run["fit"]["bdrop"], dtype=bool)))
    lbb_c = D[:p, None] * lbb[:p, :p] * D[:p][None, :]
    chol, piv, ipiv, ok = safe_pivoted_cholesky(lbb_c, np.eye(p, dtype=np.float64) * np.finfo(np.float64).eps ** 0.5)
    print("pivoted chol ok:", ok)
    print("pivot:", piv)
    print("ipiv:", ipiv)
    print("chol diag:", np.diag(chol))
    print("cov_unconditional diag:", np.diag(np.asarray(fit.cov_unconditional, dtype=np.float64)))
    print(
        "fixed-sp cov_unconditional diag:",
        np.diag(
            np.asarray(
                fixed_gam.gam_result_.fit_core_solution.fit_result.cov_unconditional,
                dtype=np.float64,
            )
        ),
    )
    print("fixed-sp warnings:", fixed_warnings)


if __name__ == "__main__":
    main()
