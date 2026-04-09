"""Shared helpers for smoothing optimization (bounds, default initial lambda)."""

import numpy as np

from ...fit.penalized_system import build_full_design
from ..criteria import resolve_ml_reml_scoring_backend


def supports_criterion_gradient(model, method):
    method = str(method).lower()
    return method in {"gcv", "ubre", "aic", "ubreaic", "ml", "reml", "laml"}


def supports_criterion_hessian(model, method):
    method = str(method).lower()
    if method in {"gcv", "ubre", "aic", "ubreaic"}:
        return True
    if method not in {"ml", "reml", "laml"}:
        return False
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend == "gaussian_dynamic" and method in {"reml", "laml"}:
        return True
    if (
        backend == "pirls_laplace"
        and method in {"reml", "laml", "ml"}
        and (
            getattr(model.family, "known_scale", None) is not None
            or str(getattr(model.family, "name", "")).lower() == "gamma"
        )
        and bool(
            getattr(model.family, "supports_exact_pirls_second_derivatives", False)
        )
    ):
        return True
    return False


def _project_to_bounds(x, bounds):
    x = np.asarray(x, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(bounds):
        x[i] = min(max(x[i], lo), hi)
    return x


def _initial_smoothing_params_from_design_balance(model, y):
    penalty_blocks = getattr(model, "penalty_blocks_", None)
    n_sp = int(getattr(model, "n_smoothing_params_", 0) or 0)
    if not penalty_blocks or n_sp == 0:
        return None

    X = build_full_design(model.Z, fit_intercept=model.fit_intercept)
    y = np.asarray(y, dtype=np.float64).ravel()

    try:
        mu0 = np.asarray(model.family.initialize_mu(y), dtype=np.float64)
        eta0 = np.asarray(model.family.link(mu0), dtype=np.float64)
        mu_eta = np.asarray(model.family.mu_eta(eta0), dtype=np.float64)
        var_mu = np.asarray(model.family.variance(mu0), dtype=np.float64)
    except Exception:
        return None

    # Heuristic from weighted column norms vs penalty diagonals (Wood-style init):
    # w <- sqrt(weights * mu.eta(eta)^2 / variance(mu))
    w_used = np.asarray(
        mu_eta * mu_eta / np.maximum(var_mu, 1e-12),
        dtype=np.float64,
    )

    weights = np.sqrt(np.clip(w_used, 1e-12, None))
    Xw = weights[:, None] * X
    ldxx = np.sum(Xw * Xw, axis=0)
    ldss = np.zeros_like(ldxx)
    def_sp = np.zeros(n_sp, dtype=np.float64)
    counts = np.zeros(n_sp, dtype=np.int64)
    penalized = np.zeros_like(ldxx, dtype=bool)

    coef_offset = 1 if bool(getattr(model, "fit_intercept", False)) else 0

    for pb in penalty_blocks:
        S = np.asarray(pb.matrix, dtype=np.float64)
        if S.size == 0:
            continue
        start = coef_offset + int(pb.coef_slice.start)
        stop = coef_offset + int(pb.coef_slice.stop)
        dS = np.diag(np.abs(S))
        if dS.size == 0:
            continue

        maS = float(np.max(np.abs(S)))
        if not np.isfinite(maS) or maS <= 0.0:
            continue
        thresh = np.finfo(np.float64).eps ** 0.8 * maS
        rsS = np.mean(np.abs(S), axis=1)
        csS = np.mean(np.abs(S), axis=0)
        ind = (rsS > thresh) & (csS > thresh) & (dS > thresh)
        if not np.any(ind):
            continue

        xx = ldxx[start:stop][ind]
        ss = dS[ind]
        if xx.size == 0 or ss.size == 0:
            continue

        sizeXX = float(np.mean(xx))
        sizeS = float(np.mean(ss))
        if not np.isfinite(sizeXX) or not np.isfinite(sizeS) or sizeS <= 0.0:
            continue

        lam = sizeXX / sizeS
        j = int(pb.smoothing_index)
        def_sp[j] += lam
        counts[j] += 1
        ldss[start:stop] += lam * np.diag(S)
        penalized[start:stop] |= ind

    ok = counts > 0
    if not np.any(ok):
        return None
    def_sp[ok] /= counts[ok]
    def_sp[~ok] = 1.0

    use = (ldss > 0.0) & penalized & (ldxx > 0.0)
    if np.any(use):
        xx = ldxx[use]
        ss = ldss[use]
        ratio = float(np.mean(xx / (xx + ss)))
        while ratio > 0.4:
            def_sp *= 10.0
            ss *= 10.0
            ratio = float(np.mean(xx / (xx + ss)))
        while ratio < 0.4:
            def_sp /= 10.0
            ss /= 10.0
            ratio = float(np.mean(xx / (xx + ss)))

    def_sp = np.maximum(def_sp, 1e-12)
    return def_sp


def _initial_smoothing_params_mgcv_style(model, y):
    penalty_blocks = getattr(model, "penalty_blocks_", None)
    n_sp = int(getattr(model, "n_smoothing_params_", 0) or 0)
    if not penalty_blocks or n_sp == 0:
        return None

    X = build_full_design(model.Z, fit_intercept=model.fit_intercept)
    y = np.asarray(y, dtype=np.float64).ravel()
    nobs, q = X.shape

    try:
        mu0 = np.asarray(model.family.initialize_mu(y), dtype=np.float64)
        eta0 = np.asarray(model.family.link(mu0), dtype=np.float64)
        mu_eta = np.asarray(model.family.mu_eta(eta0), dtype=np.float64)
        var_mu = np.asarray(model.family.variance(mu0), dtype=np.float64)
    except Exception:
        return None

    # mgcv::initial.spg for ordinary families:
    #   w <- sqrt(weights * mu.eta(eta)^2 / variance(mu))
    w = np.sqrt(np.clip(mu_eta * mu_eta / np.maximum(var_mu, 1e-12), 1e-12, None))
    Xw = w[:, None] * X

    def_sp = np.zeros(n_sp, dtype=np.float64)
    ldxx = np.sum(Xw * Xw, axis=0)
    ldss = np.zeros(q, dtype=np.float64)
    penalized = np.zeros(q, dtype=bool)

    coef_offset = 1 if bool(getattr(model, "fit_intercept", False)) else 0
    seen = np.zeros(n_sp, dtype=bool)

    for pb in penalty_blocks:
        j = int(pb.smoothing_index)
        if seen[j]:
            continue
        seen[j] = True

        S = np.asarray(pb.matrix, dtype=np.float64)
        if S.size == 0:
            continue

        start = coef_offset + int(pb.coef_slice.start)
        stop = coef_offset + int(pb.coef_slice.stop)
        maS = float(np.max(np.abs(S)))
        if not np.isfinite(maS) or maS <= 0.0:
            continue

        rsS = np.mean(np.abs(S), axis=1)
        csS = np.mean(np.abs(S), axis=0)
        dS = np.diag(np.abs(S))
        thresh = np.finfo(np.float64).eps ** 0.8 * maS
        ind = (rsS > thresh) & (csS > thresh) & (dS > thresh)
        if not np.any(ind):
            continue

        xx = ldxx[start:stop][ind]
        ss = np.diag(S)[ind]
        if xx.size == 0 or ss.size == 0:
            continue

        size_xx = float(np.mean(xx))
        size_s = float(np.mean(ss))
        if not np.isfinite(size_xx) or not np.isfinite(size_s) or size_s <= 0.0:
            continue

        lam = size_xx / size_s
        def_sp[j] = lam
        ldss[start:stop] += lam * np.diag(S)
        penalized[start:stop] |= ind

    ok = def_sp > 0.0
    if not np.any(ok):
        return None
    def_sp[~ok] = 1.0

    use = (ldss > 0.0) & penalized & (ldxx > 0.0)
    if np.any(use):
        xx = ldxx[use]
        ss = ldss[use]
        ratio = float(np.mean(xx / (xx + ss)))
        while ratio > 0.4:
            def_sp *= 10.0
            ss *= 10.0
            ratio = float(np.mean(xx / (xx + ss)))
        while ratio < 0.4:
            def_sp /= 10.0
            ss /= 10.0
            ratio = float(np.mean(xx / (xx + ss)))

    return np.maximum(def_sp, 1e-12)
