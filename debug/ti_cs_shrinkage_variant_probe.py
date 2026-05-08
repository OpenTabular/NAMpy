from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.linalg import lapack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nampy.gam.constraints.absorption import full_term_sum_to_zero_constraint
from nampy.gam.linalg import symmetric_spectrum
from nampy.gam.penalties.algebra import scale_penalty
from nampy.splines.univariate.cr import CubicSplines, add_full_rank_shrinkage
from nampy.splines.basis.cr import cr_exact_null_basis_from_knots
from tests.mgcv_parity_utils import _run_mgcv_smoothcon_penalties
from tests.mgcv_parity_utils import _run_mgcv_raw_constructor
from tests.smooths.test_mgcv_raw_constructor_parity import CASES


def _alt_shrink(S, *, lower: bool) -> np.ndarray:
    mat = np.asarray(S, dtype=np.float64)
    if lower:
        mat = np.tril(mat) + np.tril(mat, -1).T
    else:
        mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals[-2] = vals[-3] * 0.1
    vals[-1] = vals[-2] * 0.1
    return (vecs * vals) @ vecs.T


def _driver_shrink(S, driver: str, *, lower_mirror: bool = False) -> np.ndarray:
    raw = np.asarray(S, dtype=np.float64)
    mat = np.tril(raw) + np.tril(raw, -1).T if lower_mirror else 0.5 * (raw + raw.T)
    vals, vecs = eigh(mat, driver=driver, check_finite=False)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals[-2] = vals[-3] * 0.1
    vals[-1] = vals[-2] * 0.1
    return (vecs * vals) @ vecs.T


def _eig_shrink(S) -> np.ndarray:
    mat = 0.5 * (np.asarray(S, dtype=np.float64) + np.asarray(S, dtype=np.float64).T)
    vals, vecs = np.linalg.eig(mat)
    vals = vals.real
    vecs = vecs.real
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals[-2] = vals[-3] * 0.1
    vals[-1] = vals[-2] * 0.1
    return (vecs * vals) @ vecs.T


def _dsyevr_shrink(S, *, lower: int, abstol: float) -> np.ndarray:
    mat = 0.5 * (np.asarray(S, dtype=np.float64) + np.asarray(S, dtype=np.float64).T)
    w, z, _m, _isuppz, info = lapack.dsyevr(
        mat,
        compute_v=1,
        range=b"A",
        lower=lower,
        abstol=abstol,
        overwrite_a=0,
    )
    if info != 0:
        raise RuntimeError(info)
    vals = np.asarray(w, dtype=np.float64)
    vecs = np.asarray(z, dtype=np.float64)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals[-2] = vals[-3] * 0.1
    vals[-1] = vals[-2] * 0.1
    return (vecs * vals) @ vecs.T


def main() -> None:
    for case_id, col, k in [
        ("ti_2d_cs_cs", "x0", 5),
        ("ti_2d_cs_cs", "x1", 6),
        ("ti_2d_cs_ps", "x0", 5),
        ("ti_2d_ps_cs", "x1", 6),
    ]:
        print("\nCASE", case_id, col)
        _one(case_id, col, k)


def _one(case_id: str, col: str, k: int) -> None:
    raw_case = {case.case_id: case for case in CASES}[case_id]
    data = raw_case.data_factory()
    x = data[col].to_numpy(dtype=np.float64)
    spline = CubicSplines(x, k)

    expected = _run_mgcv_smoothcon_penalties(
        data[[col]],
        f's({col}, bs="cs", k={k})',
        absorb_cons=True,
        scale_penalty=True,
    )
    expected_spectrum = symmetric_spectrum(
        np.asarray(expected["S"][0], dtype=np.float64)
    )
    expected_raw = _run_mgcv_raw_constructor(
        data[[col]],
        f's({col}, bs="cs", k={k})',
    )
    expected_cr_raw = _run_mgcv_raw_constructor(
        data[[col]],
        f's({col}, bs="cr", k={k})',
    )
    expected_raw_S = np.asarray(expected_raw["S"][0], dtype=np.float64)
    expected_cr_raw_S = np.asarray(expected_cr_raw["S"][0], dtype=np.float64)

    variants = {
        "current": add_full_rank_shrinkage(spline.raw_penalty_unscaled),
        "mgcv_cr_input_evr": add_full_rank_shrinkage(expected_cr_raw_S),
        "current_null_basis": add_full_rank_shrinkage(
            spline.raw_penalty_unscaled,
            null_basis=cr_exact_null_basis_from_knots(spline.knots),
        ),
        "lower_np": _alt_shrink(spline.raw_penalty_unscaled, lower=True),
        "avg_np": _alt_shrink(spline.raw_penalty_unscaled, lower=False),
        "ev": _driver_shrink(spline.raw_penalty_unscaled, "ev"),
        "evd": _driver_shrink(spline.raw_penalty_unscaled, "evd"),
        "evr": _driver_shrink(spline.raw_penalty_unscaled, "evr"),
        "evx": _driver_shrink(spline.raw_penalty_unscaled, "evx"),
        "eig": _eig_shrink(spline.raw_penalty_unscaled),
        "dsyevr_upper": _dsyevr_shrink(
            spline.raw_penalty_unscaled, lower=0, abstol=0.0
        ),
        "dsyevr_lower": _dsyevr_shrink(
            spline.raw_penalty_unscaled, lower=1, abstol=0.0
        ),
        "dsyevr_upper_safe": _dsyevr_shrink(
            spline.raw_penalty_unscaled, lower=0, abstol=np.finfo(float).tiny
        ),
        "lower_ev": _driver_shrink(
            spline.raw_penalty_unscaled, "ev", lower_mirror=True
        ),
        "lower_evr": _driver_shrink(
            spline.raw_penalty_unscaled, "evr", lower_mirror=True
        ),
    }
    raw_vals = np.linalg.eigvalsh(
        0.5 * (spline.raw_penalty_unscaled + spline.raw_penalty_unscaled.T)
    )
    print("raw positive min", raw_vals[2])
    print("knots", spline.knots)
    print(
        "cr raw S diff", np.max(np.abs(spline.raw_penalty_unscaled - expected_cr_raw_S))
    )
    print("raw S current diff", np.max(np.abs(variants["current"] - expected_raw_S)))
    print(
        "raw S current spectrum diff",
        np.max(
            np.abs(
                symmetric_spectrum(variants["current"])
                - symmetric_spectrum(expected_raw_S)
            )
        ),
    )
    print(
        "scale current",
        np.linalg.norm(variants["current"], ord=1)
        / (np.max(np.sum(np.abs(spline.raw_basis), axis=1)) ** 2),
    )
    vals_exp, vecs_exp = np.linalg.eigh(expected_raw_S)
    order_exp = np.argsort(vals_exp)[::-1]
    vals_exp = vals_exp[order_exp]
    vecs_exp = vecs_exp[:, order_exp]
    N = np.column_stack([spline.knots - spline.knots[0], np.ones_like(spline.knots)])
    Qn, _ = np.linalg.qr(N)
    ccoords = Qn.T @ spline.raw_basis.mean(axis=0)
    print("constraint coords", ccoords)
    print("Qn first/last rows", Qn[[0, -1], :])
    print("expected raw evals", vals_exp)
    print("expected trailing in linear-null qr coords")
    print(Qn.T @ vecs_exp[:, -2:])
    vals0, vecs0 = np.linalg.eigh(
        0.5 * (spline.raw_penalty_unscaled + spline.raw_penalty_unscaled.T)
    )
    order0 = np.argsort(vals0)[::-1]
    print("numpy raw trailing evals", vals0[order0][-2:])
    print("numpy raw trailing coords")
    print(Qn.T @ vecs0[:, order0][:, -2:])
    for name, S in variants.items():
        S = scale_penalty(spline.raw_basis, S)
        _, (Sc,), _ = full_term_sum_to_zero_constraint(spline.raw_basis, [S])
        got = symmetric_spectrum(Sc)
        print(name, np.max(np.abs(got - expected_spectrum)))
        print(got)
        print(expected_spectrum)


if __name__ == "__main__":
    main()
