from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.gam_cartesian_matrix import MatrixCase, fit_model, make_data  # noqa: E402

from nampy.gam._mgcv_constants import LOG_GUARD_MIN  # noqa: E402
from nampy.gam.inference.anova import _smooth_test_stat  # noqa: E402
from nampy.gam.linalg import symmetric_eigh, symmetrize_matrix  # noqa: E402
from tests.mgcv_parity_utils import _run_mgcv_snapshot  # noqa: E402

FORMULA = 'y ~ te(x0, x1, bs=["cc","cr"], k=[8,6], sp=[1.0,1.2])'
FAMILY = {"name": "poisson", "link": "identity"}


def _first_block(diag: dict, key: str) -> np.ndarray:
    value = diag[key]
    if "blocks" in value:
        return np.asarray(value["blocks"][0], dtype=np.float64)
    return np.asarray(value["values"], dtype=np.float64)


def _test_stat_variant(
    p: np.ndarray, R_block: np.ndarray, V: np.ndarray, rank: float, residual_df: float
) -> dict[str, tuple[float, float]]:
    out = {}
    for pivoting in (False, True):
        if pivoting:
            _, R, pivot = qr(
                np.asarray(R_block, dtype=np.float64),
                mode="economic",
                pivoting=True,
            )
        else:
            _, R = qr(
                np.asarray(R_block, dtype=np.float64),
                mode="economic",
                pivoting=False,
            )
            pivot = np.arange(R.shape[1], dtype=np.intp)
        Vt = R @ V[np.ix_(pivot, pivot)] @ R.T
        evals, evecs = symmetric_eigh(symmetrize_matrix(Vt), descending=True)
        if evecs.size > 0:
            signs = np.sign(evecs[0, :])
            signs[signs == 0.0] = 1.0
            evecs = evecs * signs
        k = max(0, int(np.floor(rank)))
        nu = abs(float(rank) - k)
        k1 = k + 1 if nu > 0.0 else k
        tol = (
            max(float(np.max(evals)) if evals.size else 0.0, 0.0)
            * np.finfo(np.float64).eps ** 0.9
        )
        r_est = int(np.sum(evals > tol))
        if r_est < k1:
            k1 = r_est
            k = r_est
            nu = 0.0
            rank = float(r_est)
        vec = evecs[:, :k1].copy()
        if nu > 0.0 and k > 0:
            if k > 1:
                vec[:, : (k - 1)] = vec[:, : (k - 1)] / np.sqrt(
                    np.clip(evals[: (k - 1)], LOG_GUARD_MIN, None)
                )
            b12 = np.sqrt(max(0.5 * nu * (1.0 - nu), 0.0))
            B = np.array([[1.0, b12], [b12, nu]], dtype=np.float64)
            ev = np.diag(
                np.power(np.clip(evals[(k - 1) : k1], LOG_GUARD_MIN, None), -0.5)
            )
            B = ev @ B @ ev
            eb_vals, eb_vecs = np.linalg.eigh(B)
            rB = eb_vecs @ np.diag(np.sqrt(np.clip(eb_vals, 0.0, None))) @ eb_vecs.T
            vec[:, (k - 1) : k1] = (rB @ vec[:, (k - 1) : k1].T).T
        else:
            scale = np.sqrt(np.clip(evals[:k1], LOG_GUARD_MIN, None))
            vec = vec / scale
            if k == 1:
                rank = 1.0
        Rp = R @ p[np.asarray(pivot, dtype=np.intp)]
        out["pivoted" if pivoting else "unpivoted"] = (
            float(np.sum((vec.T @ Rp) ** 2)),
            float(rank),
        )
    return out


def _print_case(label: str, diag: dict) -> None:
    p = np.asarray(diag["smooth_test_inputs"]["coef_blocks"][0], dtype=np.float64)
    R_block = np.asarray(diag["smooth_test_inputs"]["r_blocks"][0], dtype=np.float64)
    V = _first_block(diag, "smooth_cov_bayes")
    edf = float(
        np.atleast_1d(np.asarray(diag["smooth_test_inputs"]["edf"], dtype=np.float64))[0]
    )
    edf1 = float(
        np.atleast_1d(np.asarray(diag["smooth_test_inputs"]["edf1"], dtype=np.float64))[
            0
        ]
    )
    residual_df = float(diag["smooth_test_inputs"]["residual_df"])
    rank_arg = min(float(R_block.shape[1]), max(edf1, 1.0))
    print(label)
    print("  shapes", {"p": p.shape, "R": R_block.shape, "V": V.shape})
    print("  edf/edf1/resid/rank_arg", edf, edf1, residual_df, rank_arg)
    print("  anova", diag["anova_smooth"])
    print("  R rank/default", int(np.linalg.matrix_rank(R_block)))
    print("  R column norms first/last", np.linalg.norm(R_block, axis=0)[:5], np.linalg.norm(R_block, axis=0)[-5:])
    stat = _smooth_test_stat(p, R_block, V, rank=rank_arg, residual_df=-1.0)
    print("  python current testStat", stat)
    variants = _test_stat_variant(p, R_block, V, rank=rank_arg, residual_df=-1.0)
    print("  python QR variants", variants)


def main() -> None:
    data = make_data("count")
    case = MatrixCase(
        case_id="debug_poisson_identity_te_cc_cr",
        formula=FORMULA,
        family=FAMILY,
        method="fixed",
        data_kind="count",
    )
    gam = fit_model(case, data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.method,
        allow_live_run=True,
    )
    for key in ("loglik", "log_likelihood", "deviance", "edf_total"):
        if key in actual["fit"] or key in expected["fit"]:
            print("fit", key, actual["fit"].get(key), expected["fit"].get(key))
    actual_diag = actual["parity"]["diagnostics"]
    expected_diag = expected["parity"]["diagnostics"]
    _print_case("actual", actual_diag)
    _print_case("expected", expected_diag)
    Ra = np.asarray(actual_diag["smooth_test_inputs"]["r_blocks"][0], dtype=np.float64)
    Re = np.asarray(expected_diag["smooth_test_inputs"]["r_blocks"][0], dtype=np.float64)
    print("R diff max/frob", float(np.max(np.abs(Ra - Re))), float(np.linalg.norm(Ra - Re)))
    pa = np.asarray(actual_diag["smooth_test_inputs"]["coef_blocks"][0], dtype=np.float64)
    pe = np.asarray(expected_diag["smooth_test_inputs"]["coef_blocks"][0], dtype=np.float64)
    print("coef diff max", float(np.max(np.abs(pa - pe))))
    Va = _first_block(actual_diag, "smooth_cov_bayes")
    Ve = _first_block(expected_diag, "smooth_cov_bayes")
    print("Vp diff max/frob", float(np.max(np.abs(Va - Ve))), float(np.linalg.norm(Va - Ve)))


if __name__ == "__main__":
    main()
