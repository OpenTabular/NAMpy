import ctypes
import os
import shutil
import subprocess
from ctypes.util import find_library
from functools import lru_cache

import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.linalg import eigh

from .._mgcv_constants import EIG_TOL_POWER
from ..penalties.algebra import penalty_eigendecomposition


def _configure_dsyevr_signature(fn):
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
        ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_int),
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_int),
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    fn.restype = None
    return fn


@lru_cache(maxsize=1)
def _discover_r_lapack_path():
    override = os.environ.get("NAMPY_T2_LAPACK_LIB")
    if override:
        return override

    rscript = shutil.which("Rscript")
    if not rscript:
        return None

    cmd = [
        rscript,
        "-e",
        (
            "si <- sessionInfo(); "
            "cat(if (is.null(si$LAPACK)) '' else si$LAPACK)"
        ),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    path = proc.stdout.strip()
    return path or None


@lru_cache(maxsize=1)
def _load_system_dsyevr():
    candidates = []

    r_lapack = _discover_r_lapack_path()
    if r_lapack:
        candidates.append(r_lapack)

    for libname in ("Rlapack", "lapack"):
        resolved = find_library(libname)
        if resolved:
            candidates.append(resolved)

    seen = set()
    for libname in candidates:
        if libname in seen:
            continue
        seen.add(libname)
        try:
            lib = ctypes.CDLL(libname)
            fn = _configure_dsyevr_signature(lib.dsyevr_)
            return fn
        except Exception:
            continue

    return None


def _system_dsyevr_eigh_lower(matrix):
    fn = _load_system_dsyevr()
    if fn is None:
        return None

    A = np.array(matrix, dtype=np.float64, order="F", copy=True)
    n_int = int(A.shape[0])
    n = ctypes.c_int(n_int)
    lda = ctypes.c_int(max(1, n_int))
    vl = ctypes.c_double(0.0)
    vu = ctypes.c_double(0.0)
    il = ctypes.c_int(0)
    iu = ctypes.c_int(0)
    abstol = ctypes.c_double(0.0)
    m = ctypes.c_int(0)
    w = np.empty(n_int, dtype=np.float64)
    z = np.empty((n_int, n_int), dtype=np.float64, order="F")
    ldz = ctypes.c_int(max(1, n_int))
    isuppz = np.empty(2 * n_int, dtype=np.int32)
    work = np.empty(1, dtype=np.float64)
    lwork = ctypes.c_int(-1)
    iwork = np.empty(1, dtype=np.int32)
    liwork = ctypes.c_int(-1)
    info = ctypes.c_int(0)

    query_args = [
        b"V",
        b"A",
        b"L",
        ctypes.byref(n),
        A,
        ctypes.byref(lda),
        ctypes.byref(vl),
        ctypes.byref(vu),
        ctypes.byref(il),
        ctypes.byref(iu),
        ctypes.byref(abstol),
        ctypes.byref(m),
        w,
        z,
        ctypes.byref(ldz),
        isuppz,
        work,
        ctypes.byref(lwork),
        iwork,
        ctypes.byref(liwork),
        ctypes.byref(info),
        1,
        1,
        1,
    ]

    try:
        fn(*query_args)
        if info.value != 0:
            return None

        lwork = ctypes.c_int(int(work[0]))
        liwork = ctypes.c_int(int(iwork[0]))
        work = np.empty(lwork.value, dtype=np.float64)
        iwork = np.empty(liwork.value, dtype=np.int32)
        m = ctypes.c_int(0)
        info = ctypes.c_int(0)

        run_args = [
            b"V",
            b"A",
            b"L",
            ctypes.byref(n),
            A,
            ctypes.byref(lda),
            ctypes.byref(vl),
            ctypes.byref(vu),
            ctypes.byref(il),
            ctypes.byref(iu),
            ctypes.byref(abstol),
            ctypes.byref(m),
            w,
            z,
            ctypes.byref(ldz),
            isuppz,
            work,
            ctypes.byref(lwork),
            iwork,
            ctypes.byref(liwork),
            ctypes.byref(info),
            1,
            1,
            1,
        ]
        fn(*run_args)
    except Exception:
        return None

    if info.value != 0:
        return None

    evals = w[: m.value].copy()
    evecs = z[:, : m.value].copy(order="F")
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


def _t2_symmetric_eigh(matrix):
    A = 0.5 * (
        np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T
    )
    sys_res = _system_dsyevr_eigh_lower(A)
    if sys_res is not None:
        return sys_res
    evals, evecs = eigh(A, driver="evr")
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


def rowwise_kronecker(matrices):
    mats = [np.asarray(M, dtype=np.float64) for M in matrices]
    if len(mats) == 0:
        raise ValueError("matrices must contain at least one matrix.")
    n = mats[0].shape[0]
    for M in mats:
        if M.ndim != 2 or M.shape[0] != n:
            raise ValueError("All marginal model matrices must be 2D with equal rows.")
    out = mats[0]
    for M in mats[1:]:
        out = np.einsum("ij,ik->ijk", out, M, optimize=True).reshape(
            n, out.shape[1] * M.shape[1]
        )
    return out


def _eigen_split(raw_basis, raw_penalty, tol=None, *, mode="range_null", rank=None):
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)

    if tol is None:
        tol = float(np.finfo(np.float64).eps ** EIG_TOL_POWER)

    if mode == "range_null":
        dec = penalty_eigendecomposition(S, tol=tol)
        U0, U1, d_pos = dec["U0"], dec["U1"], dec["d_pos"]
        if d_pos.size > 0:
            T_r = U1 / np.sqrt(d_pos)[np.newaxis, :]
            B_r = X @ T_r
        else:
            T_r = np.empty((S.shape[0], 0), dtype=np.float64)
            B_r = np.empty((X.shape[0], 0), dtype=np.float64)
        T_n = U0
        B_n = (
            X @ T_n if T_n.shape[1] > 0 else np.empty((X.shape[0], 0), dtype=np.float64)
        )
        return {
            "B_range": B_r,
            "B_null": B_n,
            "T_range": T_r,
            "T_null": T_n,
            "range_dim": B_r.shape[1],
            "null_dim": B_n.shape[1],
            "rank": dec["rank"],
            "null_space_dim": dec["null_space_dim"],
            "tol_eff": dec["tol_eff"],
        }

    if mode != "t2":
        raise ValueError(f"Unknown eigen split mode {mode!r}.")

    p = int(X.shape[1])
    # Match mgcv::nat.param(type=3), which uses R's `eigen(..., symmetric=TRUE)`.
    # On repeated-zero eigenspaces, the exact null-space basis feeds directly
    # into t2 block construction. Prefer the system LAPACK dsyevr path when
    # available, because that matches local R builds much more closely than the
    # bundled SciPy LAPACK on parity-sensitive tensor P-spline cases.
    evals, U = _t2_symmetric_eigh(S)

    max_eval = float(np.max(evals)) if evals.size else 0.0
    tol_eff = float(max_eval * tol)
    if rank is None or int(rank) < 1 or int(rank) > p:
        rank = int(np.sum(evals > tol_eff))
    rank = int(rank)
    null_exists = rank < p

    E = np.ones(p, dtype=np.float64)
    if rank > 0:
        E[:rank] = np.sqrt(np.maximum(evals[:rank], 0.0))

    Xp = X @ U
    col_norm = np.sum(Xp**2, axis=0) / (E**2)
    av_norm = float(np.mean(col_norm[:rank])) if rank > 0 else 1.0

    if null_exists:
        for i in range(rank, p):
            if av_norm > 0.0 and col_norm[i] > 0.0:
                E[i] = np.sqrt(col_norm[i] / av_norm)

    P = U / E[np.newaxis, :]
    Xp = Xp / E[np.newaxis, :]

    if null_exists and rank < p - 1:
        ind = list(range(rank, p))
        rind = list(range(p - 1, rank - 1, -1))
        Xn = Xp[:, ind].copy()
        n = Xn.shape[0]
        one = np.ones(n, dtype=np.float64)
        Xn = Xn - (one[:, None] * (one[None, :] @ Xn)) / n
        _, um_vecs = _t2_symmetric_eigh(Xn.T @ Xn)
        Xp[:, rind] = Xp[:, ind] @ um_vecs
        P[:, rind] = P[:, ind] @ um_vecs

    if rank > 0:
        pen_idx = list(range(rank))
        scale = 1.0 / np.sqrt(float(np.mean(Xp[:, pen_idx] ** 2)))
        Xp[:, pen_idx] *= scale
        P[pen_idx, :] *= scale

    if null_exists:
        null_idx = list(range(rank, p))
        scale_f = 1.0 / np.sqrt(float(np.mean(Xp[:, null_idx] ** 2)))
        Xp[:, null_idx] *= scale_f
        P[null_idx, :] *= scale_f

    B_r = Xp[:, :rank] if rank > 0 else np.empty((X.shape[0], 0), dtype=np.float64)
    B_n = Xp[:, rank:] if null_exists else np.empty((X.shape[0], 0), dtype=np.float64)
    T_r = P[:, :rank] if rank > 0 else np.empty((p, 0), dtype=np.float64)
    T_n = P[:, rank:] if null_exists else np.empty((p, 0), dtype=np.float64)

    return {
        "B_range": B_r,
        "B_null": B_n,
        "T_range": T_r,
        "T_null": T_n,
        "range_dim": int(B_r.shape[1]),
        "null_dim": int(B_n.shape[1]),
        "rank": rank,
        "null_space_dim": int(p - rank),
        "tol_eff": tol_eff,
    }


def marginal_range_null_decomposition(raw_basis, raw_penalty, tol=1e-10):
    return _eigen_split(raw_basis, raw_penalty, tol=tol, mode="range_null")


def _apply_t2_mgcv_column_signs(dec, basis_name):
    basis_name = None if basis_name is None else str(basis_name).lower()
    if basis_name is None:
        return dec

    sign_idx = []
    n_cols = int(dec["range_dim"] + dec["null_dim"])
    if n_cols <= 0:
        return dec

    # mgcv::nat.param(type=3) leaves the reparameterized marginal basis defined
    # only up to per-column signs. For t2() those signs feed directly into the
    # tensor ANOVA block columns, so we mirror mgcv's basis-family conventions
    # explicitly here.
    if basis_name in {"cr", "cs"}:
        sign_idx.append(n_cols - 1)
    elif basis_name == "ps":
        if n_cols > 0:
            sign_idx.append(0)
        if n_cols > 1:
            sign_idx.append(1)
    elif basis_name == "cc" and dec["range_dim"] > 0:
        sign_idx.append(int(dec["range_dim"]) - 1)
    elif basis_name in {"tp", "ts"} and n_cols > 2:
        sign_idx.append(2)

    if not sign_idx:
        return dec

    full_X = np.column_stack([dec["B_range"], dec["B_null"]])
    full_P = np.column_stack([dec["T_range"], dec["T_null"]])
    for j in sorted({int(idx) for idx in sign_idx if 0 <= int(idx) < n_cols}):
        full_X[:, j] *= -1.0
        full_P[:, j] *= -1.0

    n_range = int(dec["range_dim"])
    return {
        **dec,
        "B_range": full_X[:, :n_range],
        "B_null": full_X[:, n_range:],
        "T_range": full_P[:, :n_range],
        "T_null": full_P[:, n_range:],
    }


def t2_marginal_reparameterization(
    raw_basis, raw_penalty, tol=1e-10, *, knots=None, basis_name=None
):
    del knots
    dec = _eigen_split(raw_basis, raw_penalty, tol=tol, mode="t2")
    return _apply_t2_mgcv_column_signs(dec, basis_name)


__all__ = [
    "rowwise_kronecker",
    "marginal_range_null_decomposition",
    "t2_marginal_reparameterization",
]
