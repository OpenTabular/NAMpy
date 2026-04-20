"""Shared helpers for smoothing optimization (bounds, default initial lambda)."""

import numpy as np

from ..._mgcv_constants import EIG_TOL_POWER
from ..._model_state import (
    _coef_column_offset,
    _design_matrix,
    _fit_intercept,
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ...fit.penalized_system import build_full_design
from ..criteria import resolve_ml_reml_scoring_backend


def _r_matrix_norm_max_abs(M) -> float:
    """Mirror R ``norm(M, "M")`` used by ``mgcv::initial.spg``."""
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return 0.0
    return float(np.max(np.abs(M)))


def _r_matrix_norm_one(M) -> float:
    """Mirror R matrix ``norm()`` default one-norm."""
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return 0.0
    if M.ndim == 1:
        return float(np.sum(np.abs(M)))
    return float(np.max(np.sum(np.abs(M), axis=0)))


def supports_criterion_gradient(model, method):
    method = str(method).lower()
    if method in {"gcv", "ncv", "qncv", "ubre", "aic", "ubreaic"}:
        return True
    if method not in {"ml", "reml", "laml"}:
        return False
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend in {"gaussian_exact", "gaussian_dynamic"} and method in {"reml", "laml"}:
        return True
    if backend == "general_fit5":
        return True
    if (
        backend == "pirls_laplace"
        and method in {"reml", "laml", "ml"}
        and (
            getattr(model.family, "known_scale", None) is not None
            or str(getattr(model.family, "name", "")).lower() == "gamma"
        )
        and bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False))
    ):
        return True
    return False


def supports_criterion_hessian(model, method):
    method = str(method).lower()
    if method in {"gcv", "ubre", "aic", "ubreaic"}:
        return True
    if method in {"ncv", "qncv"}:
        return False
    if method not in {"ml", "reml", "laml"}:
        return False
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend in {"gaussian_exact", "gaussian_dynamic"} and method in {"reml", "laml"}:
        return True
    if backend == "general_fit5":
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
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    n_sp = _n_smoothing_params(model)
    if not penalty_blocks or n_sp == 0:
        return None

    X = build_full_design(_design_matrix(model), fit_intercept=_fit_intercept(model))
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

    coef_offset = _coef_column_offset(model)

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
        thresh = np.finfo(np.float64).eps ** EIG_TOL_POWER * maS
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
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    n_sp = _n_smoothing_params(model)
    if not penalty_blocks or n_sp == 0:
        return None

    family_class = str(
        getattr(getattr(model, "family", None), "family_class", "")
    ).lower()
    if family_class == "general":
        try:
            from scipy.linalg.lapack import get_lapack_funcs

            from ...families.gamlss._base import _pen_reg
            from ...fit.solvers.general_fit5 import build_gam_fit5_setup_state
            from ..reparam import build_estimate_gam_setup_state

            fit5_setup = build_gam_fit5_setup_state(
                model,
                np.ones(n_sp, dtype=np.float64),
                score_type="REML",
            )
            exact_setup = build_estimate_gam_setup_state(model)
            X = np.asarray(fit5_setup.X_initial, dtype=np.float64)
            weights = (
                np.ones_like(np.asarray(y, dtype=np.float64).ravel(), dtype=np.float64)
                if getattr(model, "prior_weights_", None) is None
                else np.asarray(model.prior_weights_, dtype=np.float64).ravel()
            )
            yv = np.asarray(y, dtype=np.float64).ravel()
            E_init = np.asarray(exact_setup.Eb, dtype=np.float64)
            if str(getattr(model.family, "name", "")).lower() == "gaulss":
                start = np.zeros(X.shape[1], dtype=np.float64)
                X1 = np.asarray(X[:, fit5_setup.jj[0]], dtype=np.float64)
                E1 = np.asarray(E_init[:, fit5_setup.jj[0]], dtype=np.float64)
                yt1 = yv.copy()
                start1 = _pen_reg(X1, E1, yt1)
                start[np.asarray(fit5_setup.jj[0], dtype=int)] = start1

                mu_init = np.asarray(
                    model.family.linfo[0].linkinv(X1 @ start1),
                    dtype=np.float64,
                )
                lres1 = np.log(np.maximum(np.abs(yv - mu_init), 1e-300))
                X2 = np.asarray(X[:, fit5_setup.jj[1]], dtype=np.float64)
                E2 = np.asarray(E_init[:, fit5_setup.jj[1]], dtype=np.float64)
                start2 = _pen_reg(X2, E2, lres1)
                start[np.asarray(fit5_setup.jj[1], dtype=int)] = start2
            else:
                start = np.asarray(
                    model.family.initialize(
                        yv,
                        X,
                        fit5_setup.jj,
                        offset=fit5_setup.offset_list,
                        weights=weights,
                        E=E_init,
                    ),
                    dtype=np.float64,
                )
            lbb = np.asarray(
                model.family.ll(
                    yv,
                    X,
                    fit5_setup.jj,
                    start,
                    weights,
                    offset=fit5_setup.offset_list,
                    deriv=1,
                )["lbb"],
                dtype=np.float64,
            )
            full_idx = np.asarray(fit5_setup.layout.reduced_to_full_idx, dtype=int)
            pstrf = get_lapack_funcs("pstrf", dtype=np.float64)
            def_sp = np.zeros(n_sp, dtype=np.float64)
            seen = np.zeros(n_sp, dtype=bool)

            for pb in penalty_blocks:
                j = int(pb.smoothing_index)
                if seen[j]:
                    continue
                seen[j] = True

                S = np.asarray(pb.matrix, dtype=np.float64)
                if S.size == 0:
                    continue
                ind = np.asarray(full_idx[pb.coef_slice], dtype=int)
                if ind.size == 0:
                    continue
                block_lbb = np.asarray(lbb[np.ix_(ind, ind)], dtype=np.float64)
                rank_i = (
                    int(pb.rank)
                    if getattr(pb, "rank", None) is not None
                    else int(np.linalg.matrix_rank(S))
                )

                if rank_i < S.shape[1]:
                    _R, piv, _rank_p, _info = pstrf(
                        np.asarray(S, dtype=np.float64), lower=0
                    )
                    piv = np.asarray(piv, dtype=int).ravel() - 1
                    Z = np.asarray(S[:, piv[:rank_i]], dtype=np.float64)
                    zn = _r_matrix_norm_one(Z)
                    if not np.isfinite(zn) or zn <= 0.0:
                        continue
                    Z = Z / zn
                    ZHZ = -np.asarray(Z.T @ block_lbb @ Z, dtype=np.float64)
                    ZSZ = np.asarray(Z.T @ S @ Z, dtype=np.float64)
                else:
                    ZHZ = -np.asarray(block_lbb, dtype=np.float64)
                    ZSZ = np.asarray(S, dtype=np.float64)

                num = _r_matrix_norm_max_abs(ZHZ)
                den = _r_matrix_norm_max_abs(ZSZ)
                if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
                    continue
                def_sp[j] = 0.3 * num / den

            ok = def_sp > 0.0
            if not np.any(ok):
                return None
            def_sp[~ok] = 1.0
            return np.maximum(def_sp, 1e-12)
        except Exception:
            return None

    X = build_full_design(_design_matrix(model), fit_intercept=_fit_intercept(model))
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

    coef_offset = _coef_column_offset(model)
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
        thresh = np.finfo(np.float64).eps ** EIG_TOL_POWER * maS
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
