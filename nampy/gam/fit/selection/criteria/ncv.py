"""Neighbourhood cross-validation from ``gam.fit3``/``src/ncv.c`` algebra.

Known offsets remain in held-out predictors and deleted Hessians retain signed
Newton weights, including the cases where upstream RNcv downdates differ.

References
----------
- Wood, S. N. (2024). On Neighbourhood Cross Validation.
  https://arxiv.org/abs/2404.16490
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ....model_state import _n_smoothing_params
from .pirls.common import _prior_weights
from .pirls.derivatives import _gcv_ubre_derivative_state


def normalize_nei(nei, n: int, *, index_base=None) -> dict:
    """Validate an mgcv ``nei`` list and convert indices to zero based.

    The default input convention is mgcv's one-based indexing.  Pass
    ``index_base=0`` either as a keyword or as an element of ``nei`` for native
    Python indices.
    """
    n = int(n)
    if nei is None:
        idx = np.arange(n, dtype=np.int64)
        ends = np.arange(1, n + 1, dtype=np.int64)
        return {
            "a": idx.copy(),
            "ma": ends.copy(),
            "d": idx.copy(),
            "md": ends.copy(),
            "jackknife": -1,
            "gamma": 1.0,
            "index_base": 0,
        }
    if not isinstance(nei, Mapping):
        raise TypeError("nei must be a mapping or None.")
    source = dict(nei)
    base = source.pop("index_base", 1) if index_base is None else index_base
    if int(base) not in {0, 1}:
        raise ValueError("nei index_base must be 0 or 1.")
    base = int(base)
    if source.get("a") is None or source.get("ma") is None:
        # estimate.gam resets the neighbourhood indices without discarding
        # other options, including the separate post-fit jackknife gamma.
        source.update(
            a=np.arange(base, n + base),
            ma=np.arange(1, n + 1),
            d=np.arange(base, n + base),
            md=np.arange(1, n + 1),
        )
    if source.get("d") is None:
        if len(np.asarray(source["ma"]).ravel()) != n:
            raise ValueError("unclear which points NCV neighbourhoods belong to")
        source["d"] = np.arange(base, n + base)
        source["md"] = np.arange(1, n + 1)
    elif source.get("md") is None:
        raise ValueError("nei md must be supplied when d is supplied")
    a = np.rint(np.asarray(source["a"], dtype=np.float64)).astype(np.int64) - base
    d = np.rint(np.asarray(source["d"], dtype=np.float64)).astype(np.int64) - base
    ma = np.rint(np.asarray(source["ma"], dtype=np.float64)).astype(np.int64)
    md = np.rint(np.asarray(source.get("md"), dtype=np.float64)).astype(np.int64)
    if ma.ndim != 1 or md.ndim != 1 or a.ndim != 1 or d.ndim != 1:
        raise ValueError("nei a, d, ma and md must be one-dimensional.")
    if ma.size != md.size:
        raise ValueError("for NCV number of dropped and predicted neighbourhoods must match")
    if np.any(a < 0) or np.any(a >= n) or np.any(d < 0) or np.any(d >= n):
        raise ValueError("nei indexes non-existent points")
    if ma.size == 0 or ma[0] < 1 or ma[-1] > a.size or np.any(np.diff(ma) < 1):
        raise ValueError("nei ma is faulty")
    if md.size == 0 or md[0] < 1 or md[-1] > d.size or np.any(np.diff(md) < 1):
        raise ValueError("nei md is faulty")
    return {
        **source,
        "a": a,
        "ma": ma,
        "d": d,
        "md": md,
        "jackknife": -1 if source.get("jackknife") is None else source["jackknife"],
        "gamma": 1.0 if source.get("gamma") is None else float(source["gamma"]),
        "index_base": 0,
    }


def _deviance_contributions(family, y, mu, weights):
    return np.asarray(
        [
            family.deviance(
                np.asarray([yi]), np.asarray([mui]), weights=np.asarray([wi])
            )
            for yi, mui, wi in zip(y, mu, weights, strict=True)
        ],
        dtype=np.float64,
    )


def _ncv_state(model, y, log_sp, *, qapprox=False):
    # The criterion remains defined with fixed penalties or no penalties.
    # Only its smoothing-parameter gradient is empty in those cases.
    sol, kernel, free_mask = _gcv_ubre_derivative_state(
        model, y, log_sp, require_free=False
    )
    nei = normalize_nei(getattr(model, "nei", None), int(model.n_samples_))
    X = np.asarray(kernel.current.X, dtype=np.float64)
    A = np.asarray(kernel.current.A, dtype=np.float64)
    W = np.asarray(kernel.current.W, dtype=np.float64)
    eta = np.asarray(sol.fit_result.eta, dtype=np.float64)
    mu = np.asarray(sol.fit_result.mu, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    prior = _prior_weights(model, yv)
    variance = np.asarray(model.family.variance(mu), dtype=np.float64)
    mu_eta = np.asarray(model.family.mu_eta(eta), dtype=np.float64)
    residual = yv - mu
    ww = prior * residual * mu_eta / variance
    mu_eta2 = -np.asarray(model.family.d2link(mu), dtype=np.float64) * mu_eta**3
    ww_eta = prior * (
        -mu_eta * mu_eta / variance
        + residual * mu_eta2 / variance
        - residual
        * mu_eta
        * np.asarray(model.family.dvar(mu), dtype=np.float64)
        * mu_eta
        / (variance * variance)
    )
    nsp = int(_n_smoothing_params(model) or 0)
    # gam.fit3 initializes the whole output, including trailing prediction
    # indices beyond the final md endpoint accepted by estimate.gam.
    eta_cv = np.zeros(nei["d"].size, dtype=np.float64)
    deta_cv = np.zeros((nei["d"].size, nsp), dtype=np.float64)
    a0 = d0 = 0
    for a1, d1 in zip(nei["ma"], nei["md"], strict=True):
        ai = nei["a"][a0:a1]
        di = nei["d"][d0:d1]
        Xa = X[ai]
        A_minus = A - Xa.T @ (W[ai, None] * Xa)
        b_drop = Xa.T @ ww[ai]
        try:
            delta = -np.linalg.solve(A_minus, b_drop)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(A_minus, hermitian=True) @ b_drop
        eta_cv[d0:d1] = eta[di] + X[di] @ delta
        for j in range(nsp):
            deta = np.asarray(kernel.ift.deta[j], dtype=np.float64)
            dA_minus = np.asarray(kernel.ift.dA[j], dtype=np.float64) - Xa.T @ (
                np.asarray(kernel.ift.dW_obs[j], dtype=np.float64)[ai, None] * Xa
            )
            db_drop = Xa.T @ (ww_eta[ai] * deta[ai])
            try:
                ddelta = -np.linalg.solve(A_minus, dA_minus @ delta + db_drop)
            except np.linalg.LinAlgError:
                ddelta = -np.linalg.pinv(A_minus, hermitian=True) @ (
                    dA_minus @ delta + db_drop
                )
            deta_cv[d0:d1, j] = deta[di] + X[di] @ ddelta
        a0, d0 = int(a1), int(d1)
    predicted = nei["d"]
    # gam.fit3 uses the model's gamma for selection. nei$gamma belongs to
    # gam.outer's separate post-fit neighbourhood/jackknife evaluation.
    gamma = float(getattr(model, "score_gamma", 1.0))
    if qapprox:
        diff = eta_cv - eta[predicted]
        dev_obs = _deviance_contributions(model.family, yv, mu, prior)
        score = float(
            np.sum(dev_obs[predicted])
            + gamma
            * np.sum(-2.0 * ww[predicted] * diff + W[predicted] * diff * diff)
        )
        deta = np.column_stack(kernel.ift.deta) if nsp else np.empty((len(yv), 0))
        dweight = (
            np.column_stack(kernel.ift.dW_obs)
            if nsp
            else np.empty((len(yv), 0), dtype=np.float64)
        )
        grad_rows = (
            -2.0
            * ww[predicted, None]
            * ((1.0 - gamma) * deta[predicted] + gamma * deta_cv)
            + 2.0 * gamma * W[predicted, None] * deta_cv * diff[:, None]
            + gamma * dweight[predicted] * diff[:, None] ** 2
        )
    else:
        eta_cv = gamma * eta_cv - (gamma - 1.0) * eta[predicted]
        if gamma != 1.0 and nsp:
            deta = np.column_stack(kernel.ift.deta)
            deta_cv = gamma * deta_cv - (gamma - 1.0) * deta[predicted]
        mu_cv = np.asarray(model.family.inverse_link(eta_cv), dtype=np.float64)
        score = float(model.family.deviance(yv[predicted], mu_cv, weights=prior[predicted]))
        var_cv = np.asarray(model.family.variance(mu_cv), dtype=np.float64)
        mu_eta_cv = np.asarray(model.family.mu_eta(eta_cv), dtype=np.float64)
        ww_cv = prior[predicted] * (yv[predicted] - mu_cv) * mu_eta_cv / var_cv
        ww_cv[~np.isfinite(ww_cv)] = 0.0
        grad_rows = -2.0 * ww_cv[:, None] * deta_cv
    grad = np.sum(grad_rows, axis=0) if nsp else np.empty(0)
    return score, grad, free_mask, eta_cv, grad_rows


def criterion_ncv_pirls(model, y, log_sp, *, qapprox=False):
    return float(_ncv_state(model, y, log_sp, qapprox=qapprox)[0])


def criterion_gradient_ncv_pirls(model, y, log_sp, *, qapprox=False):
    _score, grad, free_mask, _eta, _rows = _ncv_state(
        model, y, log_sp, qapprox=qapprox
    )
    return np.asarray(grad[free_mask], dtype=np.float64)


__all__ = ["criterion_gradient_ncv_pirls", "criterion_ncv_pirls", "normalize_nei"]
