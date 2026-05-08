from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from copy import copy
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg.lapack import get_lapack_funcs

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.fit.solvers.general_family.fixed_smoothing import (  # noqa: E402
    GeneralPenaltySetup,
    build_general_family_setup_state,
    sl_initial_repara,
)
from nampy.gam.linalg.norms import r_matrix_norm_max_abs  # noqa: E402
from nampy.gam.smoothing_selection.optimize.basics import (  # noqa: E402
    _initial_smoothing_params_mgcv_style,
)
from nampy.gam.smoothing_selection.reparam import (  # noqa: E402
    build_estimate_gam_setup_state,
)
from tests.families.test_general_family_mgcv_parity import (  # noqa: E402
    _gaulss_two_smooth_data,
    _gevlss_data,
)
from tests.optimization.test_mgcv_outer_optimization_parity import (  # noqa: E402
    _compile_optimization_state,
    _run_mgcv_initial_spg,
)
from tests.mgcv_parity_utils import _normalize_python_formula_text  # noqa: E402


CASES = {
    "gaulss_two_cr": (
        "gaulss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        lambda: _gaulss_two_smooth_data(seed=33, n=140),
        "ML",
    ),
    "gevlss_cr": (
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        "ML",
    ),
}

R_SCRIPT = shutil.which("Rscript")
MGCV_SL_STATE_SCRIPT = REPO_ROOT / "debug" / "probe_mgcv_sl_initial_state.R"


def _r_default_matrix_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.size == 0:
        return float(np.max(np.abs(x))) if x.size else 0.0
    return float(np.max(np.sum(np.abs(x), axis=0)))


def _rank_from_values(values: np.ndarray) -> int:
    vmax = float(np.max(values)) if values.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        return 0
    return int(np.sum(values > np.finfo(np.float64).eps**0.8 * vmax))


def _eigh_variant(S: np.ndarray, *, driver: str, lower: bool):
    values, vectors = scipy_eigh(
        np.asarray(S, dtype=np.float64),
        lower=lower,
        check_finite=False,
        driver=driver,
    )
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=np.float64), np.asarray(
        vectors[:, order], dtype=np.float64
    )


def _variant_setup(base_setup, *, driver: str, lower: bool):
    blocks = []
    for block in base_setup.Sl:
        block_copy = copy(block)
        S_list = [np.asarray(Si, dtype=np.float64).copy() for Si in block.S]
        block_copy.S = S_list
        if len(S_list) == 1:
            S0 = S_list[0]
            ut = np.triu_indices_from(S0, k=1)
            if float(np.sum(np.abs(S0[ut]))) != 0.0:
                values, vectors = _eigh_variant(S0, driver=driver, lower=lower)
                rank = int(block.rank) if block.rank is not None else _rank_from_values(values)
                ind = np.zeros(values.shape[0], dtype=bool)
                ind[: min(rank, ind.size)] = True
                dvals = values.copy()
                dvals[ind] = 1.0 / np.sqrt(dvals[ind])
                dvals[~ind] = 1.0
                block_copy.D = np.asarray(vectors * dvals[np.newaxis, :], dtype=np.float64)
                block_copy.Di = np.asarray(vectors.T / dvals[:, np.newaxis], dtype=np.float64)
                block_copy.ind = ind
        elif len(S_list) > 1:
            St = np.zeros_like(S_list[0], dtype=np.float64)
            for Si in S_list:
                St += Si
            values, vectors = _eigh_variant(St, driver=driver, lower=lower)
            block_copy.D = np.asarray(vectors, dtype=np.float64)
        blocks.append(block_copy)
    return GeneralPenaltySetup(
        blocks=blocks,
        E=np.asarray(base_setup.Sl.E, dtype=np.float64).copy(),
        S=np.asarray(base_setup.Sl.S, dtype=np.float64).copy(),
        lambda_=np.asarray(base_setup.Sl.lambda_, dtype=np.float64).copy(),
        cholesky=bool(base_setup.Sl.cholesky),
    )


def _initial_lambda_from_start(gam, setup, exact_setup, y, start):
    weights = (
        np.ones_like(y, dtype=np.float64)
        if gam.prior_weights_ is None
        else np.asarray(gam.prior_weights_, dtype=np.float64)
    )
    lbb = np.asarray(
        gam.family.ll(
            y,
            setup.X_initial,
            setup.jj,
            start,
            weights,
            offset=setup.offset_list,
            deriv=1,
        )["lbb"],
        dtype=np.float64,
    )
    pstrf = get_lapack_funcs("pstrf", dtype=np.float64)
    lam = np.zeros(len(exact_setup.S), dtype=np.float64)
    for i, S_i in enumerate(exact_setup.S):
        S_i = np.asarray(S_i, dtype=np.float64)
        off = int(exact_setup.off[i]) - 1
        stop = off + int(S_i.shape[1])
        block_lbb = np.asarray(lbb[off:stop, off:stop], dtype=np.float64)
        rank_i = int(exact_setup.rank[i])
        if rank_i < S_i.shape[1]:
            _R, piv, _rank_p, _info = pstrf(S_i.copy(), lower=0)
            piv = np.asarray(piv, dtype=int).ravel() - 1
            Z = np.asarray(S_i[:, piv[:rank_i]], dtype=np.float64)
            Z /= _r_default_matrix_norm(Z)
            ZHZ = -np.asarray(Z.T @ block_lbb @ Z, dtype=np.float64)
            ZSZ = np.asarray(Z.T @ S_i @ Z, dtype=np.float64)
        else:
            ZHZ = -block_lbb
            ZSZ = S_i
        lam[i] = 0.3 * r_matrix_norm_max_abs(ZHZ) / r_matrix_norm_max_abs(ZSZ)
    return lam, lbb


def _column_correlations(X: np.ndarray, X_ref: np.ndarray) -> list[float]:
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]
        r = X_ref[:, j]
        den = float(np.linalg.norm(x) * np.linalg.norm(r))
        out.append(float(x @ r / den) if den else 0.0)
    return out


def _run_mgcv_sl_state(data, formula, family: str, method: str) -> dict:
    if R_SCRIPT is None:
        raise RuntimeError("Rscript is required")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "sl_state.json"
        data.to_csv(csv_path, index=False)
        proc = subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_SL_STATE_SCRIPT),
                str(csv_path),
                _normalize_python_formula_text(formula),
                family,
                method,
                str(json_path),
            ],
            check=False,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        return json.loads(json_path.read_text(encoding="utf-8"))


def _matrix_report(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    out = {
        "shape_actual": list(actual.shape),
        "shape_expected": list(expected.shape),
    }
    if actual.shape != expected.shape:
        return out
    corr = _column_correlations(actual, expected) if actual.ndim == 2 else []
    expected_pivot_sign = []
    actual_pivot_sign = []
    if actual.ndim == 2:
        for j in range(actual.shape[1]):
            pivot = int(np.argmax(np.abs(expected[:, j])))
            expected_pivot_sign.append(float(np.sign(expected[pivot, j])))
            actual_pivot_sign.append(float(np.sign(actual[pivot, j])))
    sign = np.sign(np.asarray(corr, dtype=np.float64))
    sign[sign == 0.0] = 1.0
    out.update(
        {
            "max_abs": float(np.max(np.abs(actual - expected))) if actual.size else 0.0,
            "signed_max_abs": float(np.max(np.abs(actual * sign[np.newaxis, :] - expected)))
            if actual.ndim == 2 and actual.size
            else 0.0,
            "corr": corr,
            "actual_sign_at_expected_pivot": actual_pivot_sign,
            "expected_sign_at_pivot": expected_pivot_sign,
        }
    )
    return out


def main() -> None:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "gaulss_two_cr"
    family, formula, data_factory, method = CASES[case_id]
    data = data_factory()
    expected = _run_mgcv_initial_spg(data, formula, family, method)
    sl_expected = _run_mgcv_sl_state(data, formula, family, method)
    gam = _compile_optimization_state(data, formula, family, method)
    y = np.asarray(gam.y_, dtype=np.float64)
    n_sp = int(np.asarray(gam.smoothing_params, dtype=np.float64).size)
    base = build_general_family_setup_state(
        gam,
        np.ones(n_sp, dtype=np.float64),
        score_type=method,
    )
    exact = build_estimate_gam_setup_state(gam)
    weights = (
        np.ones_like(y, dtype=np.float64)
        if gam.prior_weights_ is None
        else np.asarray(gam.prior_weights_, dtype=np.float64)
    )
    X_ref = np.asarray(expected["X_initial"], dtype=np.float64)
    start_ref = np.asarray(expected["start"], dtype=np.float64)
    lbb_ref = np.asarray(expected["lbb"], dtype=np.float64)

    reports = []
    for driver in ("evr", "evd", "ev"):
        for lower in (True, False):
            Sl = _variant_setup(base, driver=driver, lower=lower)
            X = sl_initial_repara(Sl, base.X_full, both_sides=False)
            setup = copy(base)
            setup.sl = Sl
            setup.X_initial = X
            start = np.asarray(
                gam.family.initialize(
                    y,
                    X,
                    setup.jj,
                    offset=setup.offset_list,
                    weights=weights,
                    E=np.asarray(exact.Eb, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            lam, lbb = _initial_lambda_from_start(gam, setup, exact, y, start)
            reports.append(
                {
                    "driver": driver,
                    "lower": lower,
                    "x_max_abs": float(np.max(np.abs(X - X_ref))),
                    "x_signed_max_abs": float(
                        np.max(
                            np.abs(
                                X * np.sign(np.asarray(_column_correlations(X, X_ref)))[
                                    np.newaxis, :
                                ]
                                - X_ref
                            )
                        )
                    ),
                    "x_corr": _column_correlations(X, X_ref),
                    "start_max_abs": float(np.max(np.abs(start - start_ref))),
                    "lbb_max_abs": float(np.max(np.abs(lbb - lbb_ref))),
                    "lambda": lam.tolist(),
                    "lambda_expected": np.asarray(expected["initial_sp"], dtype=np.float64).tolist(),
                    "lambda_max_abs": float(
                        np.max(
                            np.abs(
                                lam
                                - np.asarray(expected["initial_sp"], dtype=np.float64)
                            )
                        )
                    ),
                }
            )
    reports.sort(key=lambda item: (item["x_max_abs"], item["lambda_max_abs"]))
    current = _initial_smoothing_params_mgcv_style(gam, y)
    block_reports = []
    for i, block in enumerate(base.Sl):
        if i >= len(sl_expected["blocks"]):
            continue
        exp_block = sl_expected["blocks"][i]
        block_reports.append(
            {
                "i": i,
                "start_stop_actual": [int(block.start), int(block.stop)],
                "start_stop_expected": [
                    int(exp_block["start"]),
                    int(exp_block["stop"]),
                ],
                "D": _matrix_report(
                    np.asarray(block.D, dtype=np.float64),
                    np.asarray(exp_block["D"], dtype=np.float64),
                ),
                "S0": _matrix_report(
                    np.asarray(block.S[0], dtype=np.float64),
                    np.asarray(exp_block["S"][0], dtype=np.float64),
                ),
            }
        )
    print(
        json.dumps(
            {
                "case_id": case_id,
                "X_before": _matrix_report(base.X_full, sl_expected["X_before"]),
                "X_after_current": _matrix_report(
                    base.X_initial, sl_expected["X_after"]
                ),
                "Sl_blocks_current": block_reports,
                "current_initial_sp": None
                if current is None
                else np.asarray(current, dtype=np.float64).tolist(),
                "expected_initial_sp": np.asarray(
                    expected["initial_sp"], dtype=np.float64
                ).tolist(),
                "Eb_gram_max_abs": float(
                    np.max(
                        np.abs(
                            np.asarray(exact.Eb, dtype=np.float64).T
                            @ np.asarray(exact.Eb, dtype=np.float64)
                            - np.asarray(expected["Eb"], dtype=np.float64).T
                            @ np.asarray(expected["Eb"], dtype=np.float64)
                        )
                    )
                ),
                "variants": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
