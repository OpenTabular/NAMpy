from __future__ import annotations

import numpy as np
import pandas as pd

from .._model_state import _coef_column_offset, _require_fitted, _term_blocks_seq
from .residuals import residuals_gam


def _numeric_feature_block(model, tb, row_idx):
    runtime = getattr(tb.smooth, "runtime", None)
    if runtime is None or getattr(runtime, "by", None) is not None:
        return None

    if getattr(runtime, "_feature_index", None) is not None:
        idx = [int(runtime._feature_index)]
    elif getattr(runtime, "_feature_indices", None) is not None:
        idx = [int(v) for v in runtime._feature_indices]
    else:
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


def _mgcv_nn_count(n_obs: int) -> int:
    n_obs = int(n_obs)
    if n_obs < 5000:
        return max(1, int(np.floor(np.sqrt(max(n_obs, 1)))))
    return 3


def _stabilized_k_check_edf(model, tb, edf: float) -> float:
    """Mirror mgcv's near-null-space EDF reporting for heavily penalized smooths."""
    if not np.isfinite(edf):
        return edf
    if str(getattr(tb, "term_type", "")) != "smooth":
        return edf

    lower_df = None
    for pb in getattr(model, "penalty_blocks_", ()) or ():
        if pb.coef_slice != tb.coef_slice:
            continue
        nsd = getattr(pb, "null_space_dim", None)
        if nsd is None:
            continue
        lower_df = float(nsd) if lower_df is None else max(lower_df, float(nsd))
    if lower_df is None:
        return edf

    sp_vals = [
        float(model.smoothing_params[j])
        for j in getattr(tb, "smoothing_indices", ()) or ()
        if np.isfinite(float(model.smoothing_params[j]))
    ]
    if len(sp_vals) == 0 or max(sp_vals) < 1e5:
        return edf

    delta = float(edf - lower_df)
    if 0.0 < delta < 1e-3:
        return float(lower_df + 1e-8)
    return edf


def k_check(model, subsample: int = 5000, n_rep: int = 400, seed: int | None = None):
    """Approximate mgcv::k.check() basis-dimension diagnostic.

    The returned table is intended to be numerically comparable to
    ``mgcv::k.check`` for term types where both implementations define the same
    nearest-neighbour residual differencing diagnostic.
    """
    _require_fitted(model)

    term_blocks = [
        tb
        for tb in _term_blocks_seq(model)
        if str(getattr(tb, "term_type", "")) != "parametric"
    ]
    if len(term_blocks) == 0:
        return None

    rsd = np.asarray(residuals_gam(model, type="deviance"), dtype=np.float64).ravel()
    n = int(rsd.shape[0])
    rng = np.random.default_rng(seed)
    if n > int(subsample):
        row_idx = np.sort(rng.choice(n, size=int(subsample), replace=False))
        rsd = rsd[row_idx]
    else:
        row_idx = np.arange(n, dtype=int)

    rows = []
    ve = np.empty(int(max(n_rep, 1)), dtype=np.float64)
    for i, tb in enumerate(term_blocks):
        label = str(tb.label)
        X_term = _numeric_feature_block(model, tb, row_idx)
        k_prime = int(tb.coef_slice.stop - tb.coef_slice.start)
        edf = float(model.edf_by_term_[i])
        edf = _stabilized_k_check_edf(model, tb, edf)
        if X_term is None:
            rows.append((label, k_prime, edf, np.nan, np.nan))
            continue

        if X_term.shape[1] == 1:
            e = np.diff(rsd[np.argsort(X_term[:, 0])])
            v_obs = float(np.mean(e**2) / 2.0)
            for rep in range(int(n_rep)):
                e_rep = np.diff(rsd[rng.permutation(rsd.shape[0])])
                ve[rep] = float(np.mean(e_rep**2) / 2.0)
        else:
            mp = _mgcv_nn_count(rsd.shape[0])
            ni = _nearest_indices(X_term, n_neighbors=mp)
            e = rsd - rsd[ni[:, 0]]
            for j in range(1, ni.shape[1]):
                e = np.concatenate([e, rsd - rsd[ni[:, j]]])
            v_obs = float(np.mean(e**2) / 2.0)
            for rep in range(int(n_rep)):
                rsdr = rsd[rng.permutation(rsd.shape[0])]
                e_rep = rsdr - rsdr[ni[:, 0]]
                for j in range(1, ni.shape[1]):
                    e_rep = np.concatenate([e_rep, rsdr - rsdr[ni[:, j]]])
                ve[rep] = float(np.mean(e_rep**2) / 2.0)

        if int(n_rep) > 0:
            p_val = float(np.mean(ve[: int(n_rep)] < v_obs))
        else:
            p_val = np.nan
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
        "model_rank": int(model.n_coef_ + _coef_column_offset(model)),
    }

    return {
        "mgcv_comparable": mgcv_comparable,
        "nampy_specific": nampy_specific,
    }


__all__ = ["k_check", "gam_check"]
