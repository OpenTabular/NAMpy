from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache

import numpy as np
import pandas as pd

from .._model_state import (
    _coef,
    _coef_column_offset,
    _edf_by_term,
    _n_coef,
    _require_fitted,
    _term_blocks_seq,
)
from .residuals import residuals_gam

_R_KCHECK_SAMPLE_EXPR = r"""
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
seed <- as.integer(args[[1]])
n_obs <- as.integer(args[[2]])
subsample <- as.integer(args[[3]])
nr <- as.integer(args[[4]])
n_perm <- as.integer(args[[5]])
set.seed(seed)
if (n_obs > subsample) {
  row_idx <- sample.int(n_obs, subsample) - 1L
} else {
  row_idx <- seq_len(n_obs) - 1L
}
perm_flat <- integer(0)
if (n_perm > 0L) {
  perms <- replicate(n_perm, sample.int(nr, nr) - 1L, simplify = "matrix")
  perm_flat <- as.vector(perms)
}
cat(
  toJSON(
    list(row_idx = row_idx, perm_flat = perm_flat, nrow = nr, ncol = n_perm),
    auto_unbox = TRUE
  )
)
"""


def _feature_block_indices(tb) -> list[int] | None:
    feature_info = getattr(tb, "feature_info", None)
    if feature_info is None:
        return None
    metric_idx = [int(v) for v in getattr(feature_info, "metric_feature_indices", ())]
    factor_idx = [int(v) for v in getattr(feature_info, "factor_feature_indices", ())]
    if factor_idx:
        return None
    if metric_idx:
        return metric_idx
    feature_idx = [int(v) for v in getattr(feature_info, "feature_indices", ())]
    return feature_idx or None


def _numeric_feature_block(model, tb, row_idx):
    idx = _feature_block_indices(tb)
    if idx is None or len(idx) == 0:
        return None

    X = np.asarray(model.X_, dtype=object)[row_idx][:, idx]
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.all(np.isfinite(X)):
        return None
    return X


def _nearest_indices(X: np.ndarray, n_neighbors: int = 3) -> np.ndarray:
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    return np.argsort(d2, axis=1)[:, :n_neighbors]


@lru_cache(maxsize=128)
def _r_kcheck_sample_plan(
    seed: int,
    n_obs: int,
    subsample: int,
    nr: int,
    n_perm: int,
) -> tuple[np.ndarray, np.ndarray]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Exact mgcv::k.check() parity requires Rscript for the sample.int() RNG path."
        )
    cmd = [
        rscript,
        "-e",
        _R_KCHECK_SAMPLE_EXPR,
        str(int(seed)),
        str(int(n_obs)),
        str(int(subsample)),
        str(int(nr)),
        str(int(n_perm)),
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    row_idx = np.asarray(payload["row_idx"], dtype=int).ravel()
    perm_flat = np.asarray(payload["perm_flat"], dtype=int).ravel()
    nrow = int(payload["nrow"])
    ncol = int(payload["ncol"])
    if ncol == 0:
        perms = np.empty((nrow, 0), dtype=int)
    else:
        perms = np.asarray(perm_flat, dtype=int).reshape((nrow, ncol), order="F")
    return row_idx, perms


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    # mgcv::k.check() consumes R's global RNG stream. Without an explicit seed
    # there is no stable cross-process parity target, so draw one seed locally
    # and then follow R's exact sample.int() path from that seed.
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def _is_tensor_term(tb) -> bool:
    return str(getattr(tb, "term_type", "")) in {
        "tensor_smooth",
        "tensor_interaction",
        "tensor_anova",
    }


def _tensor_rescale_data(model, tb, row_idx, X_term):
    idx = _feature_block_indices(tb)
    if idx is None or len(idx) <= 1:
        return X_term

    X_rows = np.asarray(model.X_, dtype=object)[row_idx].copy()
    beta_term = np.asarray(_coef(model)[tb.coef_slice], dtype=np.float64)
    f0 = np.asarray(tb.predict_matrix(X_rows), dtype=np.float64) @ beta_term

    out = np.asarray(X_term, dtype=np.float64).copy()
    gr_f = np.zeros(out.shape[1], dtype=np.float64)
    for i, col_idx in enumerate(idx):
        dx = float(np.max(out[:, i]) - np.min(out[:, i])) / 1000.0
        X_perturbed = X_rows.copy()
        X_perturbed[:, col_idx] = (
            np.asarray(X_perturbed[:, col_idx], dtype=np.float64) + dx
        )
        fp = np.asarray(tb.predict_matrix(X_perturbed), dtype=np.float64) @ beta_term
        gr_f[i] = float(np.mean(np.abs(fp - f0)) / dx)

    for i in range(out.shape[1]):
        out[:, i] = out[:, i] - np.min(out[:, i])
        out[:, i] = gr_f[i] * out[:, i] / np.max(out[:, i])
    return out


def k_check(model, subsample: int = 5000, n_rep: int = 400, seed: int | None = None):
    """Port of mgcv::k.check() basis-dimension diagnostic."""
    _require_fitted(model)

    term_blocks = [
        tb
        for tb in _term_blocks_seq(model)
        if str(getattr(tb, "term_type", "")) != "parametric"
    ]
    if len(term_blocks) == 0:
        return None

    rsd = np.asarray(residuals_gam(model), dtype=np.float64).ravel()
    n = int(rsd.shape[0])
    subsample = int(subsample)
    n_rep = int(n_rep)
    if subsample <= 0:
        raise ValueError("subsample must be positive.")
    if n_rep < 0:
        raise ValueError("n_rep must be non-negative.")

    n_numeric_terms = sum(_feature_block_indices(tb) is not None for tb in term_blocks)
    seed_use = _resolve_seed(seed)
    n_perm = n_numeric_terms * n_rep
    row_idx, perms = _r_kcheck_sample_plan(
        seed_use,
        n,
        subsample,
        min(n, subsample),
        n_perm,
    )
    rsd = rsd[row_idx]

    rows = []
    edf_by_term = np.asarray(_edf_by_term(model), dtype=np.float64).ravel()
    perm_col = 0
    for i, tb in enumerate(term_blocks):
        label = str(tb.label)
        X_term = _numeric_feature_block(model, tb, row_idx)
        k_prime = int(tb.coef_slice.stop - tb.coef_slice.start)
        edf = float(edf_by_term[i])
        if X_term is None:
            rows.append((label, k_prime, edf, np.nan, np.nan))
            continue

        if X_term.shape[1] == 1:
            e = np.diff(rsd[np.argsort(X_term[:, 0])])
            v_obs = float(np.mean(e**2) / 2.0)
            ve = np.empty(max(n_rep, 1), dtype=np.float64)
            for rep in range(n_rep):
                e_rep = np.diff(rsd[perms[:, perm_col]])
                ve[rep] = float(np.mean(e_rep**2) / 2.0)
                perm_col += 1
        else:
            if _is_tensor_term(tb):
                X_term = _tensor_rescale_data(model, tb, row_idx, X_term)
            ni = _nearest_indices(X_term, n_neighbors=3)
            e = rsd - rsd[ni[:, 0]]
            for j in range(1, ni.shape[1]):
                e = np.concatenate([e, rsd - rsd[ni[:, j]]])
            v_obs = float(np.mean(e**2) / 2.0)
            ve = np.empty(max(n_rep, 1), dtype=np.float64)
            for rep in range(n_rep):
                rsdr = rsd[perms[:, perm_col]]
                e_rep = rsdr - rsdr[ni[:, 0]]
                for j in range(1, ni.shape[1]):
                    e_rep = np.concatenate([e_rep, rsdr - rsdr[ni[:, j]]])
                ve[rep] = float(np.mean(e_rep**2) / 2.0)
                perm_col += 1

        p_val = float(np.mean(ve[:n_rep] < v_obs)) if n_rep > 0 else np.nan
        k_index = float(v_obs / np.mean(rsd**2))
        rows.append((label, k_prime, edf, k_index, p_val))

    return pd.DataFrame(
        rows,
        columns=["label", "k_prime", "edf", "k_index", "p_value"],
    ).set_index("label")


def gam_check(
    model,
    *,
    type: str = "deviance",
    k_sample: int = 5000,
    k_rep: int = 200,
    seed: int | None = None,
):
    """Build a post-fit diagnostic report analogous to ``mgcv::gam.check``.

    The report intentionally separates two categories of information:

    - ``mgcv_comparable``: pieces with a direct numerical analogue in mgcv,
      currently the selected residual series and the ``k_check`` table.
    - ``nampy_specific``: local optimizer and fitted-state diagnostics with no
      direct mgcv counterpart, such as convergence metadata and the assembled
      model rank.

    Callers should use the explicit nested blocks to distinguish parity-safe
    quantities from nampy-only diagnostics.
    """
    resid = residuals_gam(model, type=type)
    k_table = k_check(model, subsample=k_sample, n_rep=k_rep, seed=seed)

    optim = getattr(model, "_optim_result", None)
    convergence = {
        "method": getattr(model, "_optim_method", None),
        "success": None if optim is None else bool(getattr(optim, "success", False)),
        "message": None if optim is None else str(getattr(optim, "message", "")),
        "nit": None if optim is None else int(getattr(optim, "nit", 0)),
        "used_gradient": bool(getattr(model, "_optim_used_gradient", False)),
        "used_hessian": bool(getattr(model, "_optim_used_hessian", False)),
    }

    mgcv_comparable = {
        "residual_type": str(type),
        "residuals": np.asarray(resid, dtype=np.float64),
        "k_check": k_table,
    }
    nampy_specific = {
        "convergence": convergence,
        "model_rank": int(_n_coef(model) + _coef_column_offset(model)),
    }

    return {
        "mgcv_comparable": mgcv_comparable,
        "nampy_specific": nampy_specific,
    }


__all__ = ["k_check", "gam_check"]
