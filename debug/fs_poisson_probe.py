"""Probe the remaining Poisson fs full-fit parity gap.

This is intentionally a preserved investigation script rather than an
ephemeral REPL snippet. It separates construction/PIRLS parity from smoothing
selection by comparing full REML predictions and fixed-sp fits at both mgcv and
NAMpy smoothing parameters.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular

from nampy.gam.compiler.compile_model import (
    _r_linpack_qr_no_pivot,
    _r_linpack_qr_R,
    _r_linpack_qy,
)
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import (
    _build_r_command,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_poisson_data,
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_penalties,
    _run_mgcv_snapshot,
)
from tests.smooths.test_mgcv_raw_constructor_parity import (
    _build_runtime_term,
    _normalize_python_formula_text,
    _run_mgcv_raw_constructor,
    _serialize_term_raw,
)
from tests.smooths.test_mgcv_smoothcon_parity import _compile_formula_design


def _coverage_poisson_factor_data(seed: int = 902, n: int = 180):
    rng = np.random.default_rng(seed)
    data = _make_poisson_data(seed=seed, n=n).copy()
    data["f"] = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    return data


def _max_abs(actual, expected) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(actual, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64)
            )
        )
    )


def _max_abs_up_to_column_sign(actual, expected) -> float:
    actual_arr = np.asarray(actual, dtype=np.float64).copy()
    expected_arr = np.asarray(expected, dtype=np.float64)
    if actual_arr.shape != expected_arr.shape:
        return float("inf")
    for j in range(actual_arr.shape[1]):
        direct = np.linalg.norm(actual_arr[:, j] - expected_arr[:, j])
        flipped = np.linalg.norm(-actual_arr[:, j] - expected_arr[:, j])
        if flipped < direct:
            actual_arr[:, j] *= -1.0
    return _max_abs(actual_arr, expected_arr)


def _column_correlations_with_targets(label: str, Xb, x) -> None:
    Xb_arr = np.asarray(Xb, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    targets = np.column_stack(
        [
            x_arr - np.mean(x_arr),
            np.ones_like(x_arr),
        ]
    )
    rank = 3
    block = Xb_arr[:, rank:]
    corr = np.zeros((block.shape[1], targets.shape[1]), dtype=np.float64)
    for i in range(block.shape[1]):
        for j in range(targets.shape[1]):
            denom = np.linalg.norm(block[:, i]) * np.linalg.norm(targets[:, j])
            corr[i, j] = float(block[:, i] @ targets[:, j] / denom)
    print(f"{label}_null_corr_[centered_x,ones]", corr)


def _nat_param_variant(
    X,
    S,
    rank: int,
    *,
    qr_kind: str,
    driver: str,
    symmetrize: bool = True,
    lower: bool = True,
):
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if qr_kind == "numpy":
        Q, R = np.linalg.qr(X, mode="reduced")
    elif qr_kind == "scipy":
        Q, R = scipy_qr(X, mode="economic", pivoting=False, check_finite=False)
    elif qr_kind == "rlinpack":
        qr_pack, qraux = _r_linpack_qr_no_pivot(X)
        R = _r_linpack_qr_R(qr_pack)
        Q = _r_linpack_qy(
            qr_pack,
            qraux,
            np.eye(X.shape[0], X.shape[1], dtype=np.float64),
        )
    else:
        raise ValueError(qr_kind)

    tmp = solve_triangular(R.T, S.T, lower=True, check_finite=False)
    RSR = solve_triangular(R.T, tmp.T, lower=True, check_finite=False)
    if symmetrize:
        RSR = 0.5 * (RSR + RSR.T)
    actual_driver = "ev" if driver == "r-style" else driver
    evals, U = scipy_eigh(RSR, driver=actual_driver, lower=lower, check_finite=False)
    order = np.argsort(evals)[::-1]
    if driver == "r-style":
        ascending = np.argsort(evals)
        order = np.concatenate(
            [
                ascending[::-1][:rank],
                ascending[: max(0, evals.size - rank)],
            ]
        )
    evals = evals[order]
    U = U[:, order]
    D = evals[:rank].copy()
    Xn = Q @ U
    P = solve_triangular(R, U, lower=False, check_finite=False)
    E = np.ones(Xn.shape[1], dtype=np.float64)
    E[:rank] = np.sqrt(D)
    Xn = Xn / E[np.newaxis, :]
    P = P / E[np.newaxis, :]
    if rank > 0:
        scale = 1.0 / np.sqrt(np.mean(Xn[:, :rank] ** 2))
        Xn[:, :rank] *= scale
        P[:, :rank] *= scale
    if rank < Xn.shape[1]:
        scalef = 1.0 / np.sqrt(np.mean(Xn[:, rank:] ** 2))
        Xn[:, rank:] *= scalef
        P[:, rank:] *= scalef
    return Xn


def _run_r_natparam_dump(data: pd.DataFrame):
    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
spec <- s(x0, f, bs="fs", k=5)
sm <- smooth.construct(spec, d, NULL)
base <- smooth.construct(s(x0, bs="tp", k=5), d, NULL)
qrx <- qr(base$X, tol=.Machine$double.eps^.8)
R <- qr.R(qrx)
RSR <- forwardsolve(t(R), t(forwardsolve(t(R), t(base$S[[1]]))))
er <- eigen(RSR, symmetric=TRUE)
rp <- mgcv:::nat.param(base$X, base$S[[1]], rank=base$rank, type=1)
write_json(
  list(
    R = unname(R),
    Q = unname(qr.Q(qrx, complete=FALSE)),
    RSR = unname(RSR),
    evals = unname(er$values),
    evecs = unname(er$vectors),
    rpX = unname(rp$X),
    rpP = unname(rp$P),
    rpD = unname(rp$D),
    fsXb = unname(sm$Xb)
  ),
  args[[2]],
  auto_unbox = TRUE,
  digits = 17
)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "natparam.json"
        script_path = tmpdir_path / "natparam_dump.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(script_path, str(csv_path), str(json_path)),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    data = _coverage_poisson_factor_data()
    formula = 'y ~ s(x0, f, bs="fs", k=5)'

    actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

    actual_sp = np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
    expected_sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    print("actual_sp", actual_sp)
    print("expected_sp", expected_sp)
    print(
        "full_response_max_abs",
        _max_abs(actual["predictions"]["response"], expected["predictions"]["response"]),
    )
    print(
        "full_link_max_abs",
        _max_abs(actual["predictions"]["link"], expected["predictions"]["link"]),
    )
    print("actual_deviance", actual["fit"]["deviance"])
    print("expected_deviance", expected["fit"]["deviance"])

    smooth_expr_r = 's(x0, f, bs="fs", k=5)'
    for py_formula, r_expr in [
        (formula, smooth_expr_r),
        ('y ~ s(f, x0, bs="fs", k=5)', 's(f, x0, bs="fs", k=5)'),
    ]:
        design = _compile_formula_design(data, py_formula)
        expected_X = _run_mgcv_smoothcon_matrix(data, r_expr)["X"]
        print(
            f"smoothcon_X_max_abs_up_to_sign[{r_expr}]",
            _max_abs_up_to_column_sign(design.design_matrix, expected_X),
        )
    design = _compile_formula_design(data, formula)
    term, X_raw_data, _feature_names = _build_runtime_term(data, formula)
    actual_raw = _serialize_term_raw(term, X_raw_data)
    expected_raw = _run_mgcv_raw_constructor(
        data,
        _normalize_python_formula_text(formula.split("~", 1)[1].strip()),
    )
    print(
        "raw_X_max_abs_up_to_sign",
        _max_abs_up_to_column_sign(actual_raw["X"], expected_raw["X"]),
    )
    _column_correlations_with_targets(
        "actual_raw_Xb",
        actual_raw["extra"]["Xb"],
        data["x0"],
    )
    _column_correlations_with_targets(
        "expected_raw_Xb",
        expected_raw["extra"]["Xb"],
        data["x0"],
    )
    for idx, (got, want) in enumerate(
        zip(actual_raw["S"], expected_raw["S"], strict=True)
    ):
        print(f"raw_S{idx}_max_abs", _max_abs(got, want))
    B0, S0, _ = term._base_constructor_fit_matrices()
    expected_base_raw = _run_mgcv_raw_constructor(data, 's(x0, bs="tp", k=5)')
    print(
        "base_raw_X_max_abs_up_to_sign",
        _max_abs_up_to_column_sign(B0, expected_base_raw["X"]),
    )
    print("base_raw_S_max_abs", _max_abs(S0, expected_base_raw["S"][0]))
    _column_correlations_with_targets("actual_base_B0", B0, data["x0"])
    _column_correlations_with_targets("expected_base_B0", expected_base_raw["X"], data["x0"])
    expected_Xb = np.asarray(expected_raw["extra"]["Xb"], dtype=np.float64)
    r_nat = _run_r_natparam_dump(data)
    print("r_nat_R_vs_numpy_R", _max_abs(r_nat["R"], np.linalg.qr(B0, mode="reduced")[1]))
    scipy_Q, scipy_R = scipy_qr(B0, mode="economic", pivoting=False, check_finite=False)
    print("r_nat_R_vs_scipy_R", _max_abs(r_nat["R"], scipy_R))
    qr_pack, qraux = _r_linpack_qr_no_pivot(B0)
    rlin_R = _r_linpack_qr_R(qr_pack)
    rlin_Q = _r_linpack_qy(
        qr_pack,
        qraux,
        np.eye(B0.shape[0], B0.shape[1], dtype=np.float64),
    )
    print("r_nat_R_vs_rlin_R", _max_abs(r_nat["R"], rlin_R))
    print("r_nat_Q_vs_rlin_Q", _max_abs(r_nat["Q"], rlin_Q))
    print("r_nat_rpX_vs_expected_Xb", _max_abs(r_nat["rpX"], expected_Xb))
    print("r_nat_fsXb_vs_expected_Xb", _max_abs(r_nat["fsXb"], expected_Xb))
    print(
        "r_nat_RSR_asymmetry",
        _max_abs(np.asarray(r_nat["RSR"]), np.asarray(r_nat["RSR"]).T),
    )
    for qr_kind in ("numpy", "scipy", "rlinpack"):
        for driver in ("ev", "evd", "evr", "evx"):
            try:
                variant = _nat_param_variant(B0, S0, 3, qr_kind=qr_kind, driver=driver)
            except Exception as exc:
                print(f"variant[{qr_kind},{driver}] error {exc}")
                continue
            print(
                f"variant[{qr_kind},{driver}]_Xb_max_abs_up_to_sign",
                _max_abs_up_to_column_sign(variant, expected_Xb),
            )
            _column_correlations_with_targets(
                f"variant[{qr_kind},{driver}]",
                variant,
                data["x0"],
            )
    for qr_kind in ("numpy", "scipy", "rlinpack"):
        for lower in (True, False):
            try:
                variant = _nat_param_variant(
                    B0,
                    S0,
                    3,
                    qr_kind=qr_kind,
                    driver="r-style",
                    symmetrize=False,
                    lower=lower,
                )
            except Exception as exc:
                print(f"variant_unsym[{qr_kind},{lower}] error {exc}")
                continue
            print(
                f"variant_unsym[{qr_kind},lower={lower}]_Xb_max_abs_up_to_sign",
                _max_abs_up_to_column_sign(variant, expected_Xb),
            )
            _column_correlations_with_targets(
                f"variant_unsym[{qr_kind},lower={lower}]",
                variant,
                data["x0"],
            )
    expected_S = _run_mgcv_smoothcon_penalties(
        data,
        smooth_expr_r,
        absorb_cons=True,
        scale_penalty=True,
    )["S"]
    actual_S = [np.asarray(pb.matrix, dtype=np.float64) for pb in design.compiled_penalties]
    print("n_penalties", len(actual_S), len(expected_S))
    for idx, (got, want) in enumerate(zip(actual_S, expected_S, strict=True)):
        want_arr = np.asarray(want, dtype=np.float64)
        print(f"smoothcon_S{idx}_max_abs", _max_abs(got, want_arr))

    fixed_mgcv = _fit_nampy_model_fixed_sp(data, formula, "poisson", expected_sp)
    fixed_mgcv_snapshot = fixed_mgcv.parity_snapshot(X=data, include_covariances=True)
    print(
        "fixed_at_mgcv_sp_response_max_abs",
        _max_abs(
            fixed_mgcv_snapshot["predictions"]["response"],
            expected["predictions"]["response"],
        ),
    )
    print(
        "fixed_at_mgcv_sp_link_max_abs",
        _max_abs(
            fixed_mgcv_snapshot["predictions"]["link"],
            expected["predictions"]["link"],
        ),
    )
    print("fixed_at_mgcv_sp_deviance", fixed_mgcv_snapshot["fit"]["deviance"])

    swapped_expected_sp = expected_sp.copy()
    if swapped_expected_sp.size == 3:
        swapped_expected_sp[[1, 2]] = swapped_expected_sp[[2, 1]]
    fixed_swapped_mgcv = _fit_nampy_model_fixed_sp(
        data,
        formula,
        "poisson",
        swapped_expected_sp,
    )
    fixed_swapped_snapshot = fixed_swapped_mgcv.parity_snapshot(
        X=data,
        include_covariances=True,
    )
    print("swapped_expected_sp_for_nampy", swapped_expected_sp)
    print(
        "fixed_at_swapped_mgcv_sp_response_max_abs",
        _max_abs(
            fixed_swapped_snapshot["predictions"]["response"],
            expected["predictions"]["response"],
        ),
    )
    print(
        "fixed_at_swapped_mgcv_sp_link_max_abs",
        _max_abs(
            fixed_swapped_snapshot["predictions"]["link"],
            expected["predictions"]["link"],
        ),
    )
    print("fixed_at_swapped_mgcv_sp_deviance", fixed_swapped_snapshot["fit"]["deviance"])

    expected_at_actual = _run_mgcv_snapshot(
        data,
        formula.replace('k=5)', f"k=5, sp={actual_sp.tolist()})"),
        "poisson",
        "fixed",
    )
    print(
        "mgcv_fixed_at_actual_sp_response_vs_nampy_max_abs",
        _max_abs(
            expected_at_actual["predictions"]["response"],
            actual["predictions"]["response"],
        ),
    )
    print(
        "mgcv_fixed_at_actual_sp_link_vs_nampy_max_abs",
        _max_abs(expected_at_actual["predictions"]["link"], actual["predictions"]["link"]),
    )
    print("mgcv_fixed_at_actual_sp_deviance", expected_at_actual["fit"]["deviance"])


if __name__ == "__main__":
    main()
