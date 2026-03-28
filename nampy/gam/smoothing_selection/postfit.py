from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .criteria.ml_reml import resolve_ml_reml_scoring_backend


def _free_smoothing_mask(model) -> np.ndarray:
    n_sp = int(getattr(model, "n_smoothing_params_", 0) or 0)
    fixed = getattr(model, "smoothing_fixed_mask_", None)
    if fixed is None:
        return np.ones(n_sp, dtype=bool)
    return ~np.asarray(fixed, dtype=bool)


def _free_log_smoothing_params(model) -> np.ndarray:
    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    free_mask = _free_smoothing_mask(model)
    return np.log(np.clip(sp[free_mask], 1e-300, None))


def sp_vcov(model, edge_correct: bool = True, reg: float = 1e-3):
    del edge_correct
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return None

    backend = resolve_ml_reml_scoring_backend(model, method=method)
    result = getattr(model, "_optim_result", None)
    H = None if result is None else getattr(result, "hess", None)
    if backend in {"pirls_laplace", "pirls_laplace_dynamic"}:
        H = None
    if H is None:
        log_sp = _free_log_smoothing_params(model)
        H = np.asarray(model._criterion_hessian(model.y_, log_sp, method=method), dtype=np.float64)
    else:
        H = np.asarray(H, dtype=np.float64)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("Smoothing Hessian must be square.")
    if H.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    return np.linalg.inv(H + float(reg))


def gam_vcomp(model, *, rescale: bool = False, conf_lev: float = 0.95):
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")
    if rescale:
        raise NotImplementedError(
            "gam_vcomp(rescale=True) requires stored penalty rescaling factors, which are not yet retained."
        )

    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    if sp.size == 0:
        return None

    scale = float(model.scale_)
    vc = scale / sp
    sd = np.sqrt(vc)
    names = [f"sp_{i}" for i in range(len(sp))]
    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return {"vc": sd, "names": names}

    result = getattr(model, "_optim_result", None)
    H = None if result is None else getattr(result, "hess", None)
    if H is None:
        log_sp = _free_log_smoothing_params(model)
        H = np.asarray(model._criterion_hessian(model.y_, log_sp, method=method), dtype=np.float64)
    else:
        H = np.asarray(H, dtype=np.float64)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return {"vc": sd, "names": names}
    if H.shape[0] == 0:
        return {"vc": sd, "names": names}

    evals, evecs = np.linalg.eigh(H)
    keep = evals > np.max(evals) * np.finfo(np.float64).eps**0.75
    rank = int(np.sum(keep))
    inv_vals = np.zeros_like(evals)
    inv_vals[keep] = 1.0 / evals[keep]
    V = evecs @ (inv_vals[:, None] * evecs.T)

    crit = float(norm.ppf(1.0 - (1.0 - float(conf_lev)) / 2.0))
    lsd = np.log(sd[_free_smoothing_mask(model)])
    J = -0.5 * np.eye(V.shape[0], dtype=np.float64)
    V_lsd = J @ V @ J.T
    sd_lsd = np.sqrt(np.clip(np.diag(V_lsd), 0.0, None))
    ci = np.column_stack(
        [
            np.exp(np.clip(lsd, -700.0, 700.0)),
            np.exp(np.clip(lsd - crit * sd_lsd, -700.0, 700.0)),
            np.exp(np.clip(lsd + crit * sd_lsd, -700.0, 700.0)),
        ]
    )
    free_idx = np.flatnonzero(_free_smoothing_mask(model))
    return {
        "vc": ci,
        "names": [names[i] for i in free_idx],
        "rank": rank,
        "rank_hess": int(H.shape[0]),
        "conf_lev": float(conf_lev),
        "all": sd,
        "all_names": names,
    }


def one_se_rule(model, candidate_indices: list[int] | None = None) -> np.ndarray:
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    V = sp_vcov(model, edge_correct=False)
    if V is None:
        raise RuntimeError("one_se_rule requires ML/REML/LAML smoothing covariance.")

    free_idx = np.flatnonzero(_free_smoothing_mask(model))
    if candidate_indices is None:
        sub_idx = np.arange(free_idx.size, dtype=int)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=int).ravel()
        pos = []
        for idx in candidate_indices:
            hit = np.flatnonzero(free_idx == int(idx))
            if hit.size == 0:
                continue
            pos.append(int(hit[0]))
        if len(pos) == 0:
            raise ValueError("candidate_indices does not contain any estimated smoothing parameters.")
        sub_idx = np.asarray(pos, dtype=int)

    V_sub = np.asarray(V[np.ix_(sub_idx, sub_idx)], dtype=np.float64)
    d = np.sqrt(np.clip(np.diag(V_sub), 0.0, None))
    if np.any(d <= 0.0):
        raise RuntimeError("one_se_rule requires positive smoothing-parameter standard errors.")
    alpha = float(np.sqrt(2.0 * len(d)) / (d @ np.linalg.solve(V_sub, d)))
    step = alpha * d

    sp = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    log_sp = np.log(np.clip(sp[free_idx], 1e-300, None))
    log_sp[sub_idx] = log_sp[sub_idx] + step
    sp[free_idx] = np.exp(log_sp)
    return sp


__all__ = ["sp_vcov", "gam_vcomp", "one_se_rule"]
