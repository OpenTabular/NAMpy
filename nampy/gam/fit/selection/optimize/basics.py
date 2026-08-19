"""Shared helpers for smoothing optimization (bounds, default initial lambda)."""

import numpy as np

from ...._mgcv_constants import EIG_TOL_POWER, LOG_GUARD_MIN
from ....linalg import pivoted_cholesky
from ....linalg.norms import r_matrix_norm_max_abs
from ....model_state import (
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ...backends import GENERAL_FAMILY_BACKEND
from ...capabilities import uses_closed_form_solver
from ..criteria import resolve_ml_reml_scoring_backend


def _initial_gaussian_scale_as_sp(model, y) -> float:
    """Mirror ``mgcv::get.null.coef`` / ``scale.as.sp`` initialization."""
    yv = np.asarray(y, dtype=np.float64).ravel()
    mu_null = np.repeat(float(np.mean(yv)), int(model.n_samples_))
    null_scale = float(
        model.family.deviance(
            yv,
            mu_null,
            weights=getattr(model, "prior_weights_", None),
        )
    ) / float(model.n_samples_)
    return float(max(null_scale / 10.0, LOG_GUARD_MIN))


def supports_criterion_gradient(model, method):
    method = str(method).lower()
    if method in {"gcv", "ubre", "aic", "ubreaic"}:
        return bool(
            uses_closed_form_solver(model)
            or getattr(model.family, "supports_exact_pirls_first_derivatives", False)
        )
    if method not in {"ml", "reml", "laml"}:
        return False
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend in {"gaussian_exact", "gaussian_dynamic"}:
        return True
    if backend == GENERAL_FAMILY_BACKEND:
        return True
    if (
        backend == "pirls_laplace"
        and method in {"reml", "laml", "ml"}
        and (
            getattr(model.family, "known_scale", None) is not None
            or str(getattr(model.family, "name", "")).lower()
            in {"gamma", "gaussian"}
        )
        and bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False))
    ):
        return True
    return False


def supports_criterion_hessian(model, method):
    method = str(method).lower()
    if method in {"gcv", "ubre", "aic", "ubreaic"}:
        return bool(
            uses_closed_form_solver(model)
            or getattr(model.family, "supports_exact_pirls_second_derivatives", False)
        )
    if method not in {"ml", "reml", "laml"}:
        return False
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend in {"gaussian_exact", "gaussian_dynamic"}:
        return True
    if backend == GENERAL_FAMILY_BACKEND:
        return True
    if (
        backend == "pirls_laplace"
        and method in {"reml", "laml", "ml"}
        and (
            getattr(model.family, "known_scale", None) is not None
            or str(getattr(model.family, "name", "")).lower()
            in {"gamma", "gaussian"}
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


def _initial_smoothing_params_from_packed_penalties(
    X_weighted: np.ndarray,
    penalties: list[np.ndarray],
    offsets_1based: np.ndarray,
    *,
    L: np.ndarray | None = None,
    lsp0: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Mirror `mgcv::initial.sp()` plus the linked-penalty regression in
    `mgcv::initial.spg()` for ordinary families.

    `penalties` and `offsets_1based` are the packed `estimate.gam` penalty
    blocks (`G$S`, `G$off`), not full assembled penalty matrices.
    """
    X_weighted = np.asarray(X_weighted, dtype=np.float64)
    if X_weighted.ndim != 2 or len(penalties) == 0:
        return None

    ldxx = np.sum(X_weighted * X_weighted, axis=0)
    ldss = np.zeros_like(ldxx, dtype=np.float64)
    penalized = np.zeros_like(ldxx, dtype=bool)
    def_sp = np.zeros(len(penalties), dtype=np.float64)

    for i, (S_i, off_i) in enumerate(
        zip(penalties, offsets_1based, strict=True)
    ):
        S_i = np.asarray(S_i, dtype=np.float64)
        if S_i.ndim != 2 or S_i.shape[0] != S_i.shape[1] or S_i.size == 0:
            return None

        maS = float(np.max(np.abs(S_i)))
        if not np.isfinite(maS) or maS <= 0.0:
            return None

        rsS = np.mean(np.abs(S_i), axis=1)
        csS = np.mean(np.abs(S_i), axis=0)
        dS_abs = np.diag(np.abs(S_i))
        thresh = np.finfo(np.float64).eps ** EIG_TOL_POWER * maS
        active = (rsS > thresh) & (csS > thresh) & (dS_abs > thresh)
        if not np.any(active):
            return None

        start = int(off_i) - 1
        stop = start + int(S_i.shape[1])
        if start < 0 or stop > ldxx.size:
            return None

        xx = np.asarray(ldxx[start:stop][active], dtype=np.float64)
        ss = np.asarray(np.diag(S_i)[active], dtype=np.float64)
        if xx.size == 0 or ss.size == 0:
            return None

        size_xx = float(np.mean(xx))
        size_s = float(np.mean(ss))
        if not np.isfinite(size_xx) or not np.isfinite(size_s) or size_s <= 0.0:
            return None

        def_sp[i] = size_xx / size_s
        ldss[start:stop] += float(def_sp[i]) * np.diag(S_i)
        penalized[start:stop] |= active

    use = (ldss > 0.0) & penalized & (ldxx > 0.0)
    if not np.any(use):
        return None

    xx = np.asarray(ldxx[use], dtype=np.float64)
    ss = np.asarray(ldss[use], dtype=np.float64)
    ratio = float(np.mean(xx / (xx + ss)))
    while ratio > 0.4:
        def_sp *= 10.0
        ss *= 10.0
        ratio = float(np.mean(xx / (xx + ss)))
    while ratio < 0.4:
        def_sp /= 10.0
        ss /= 10.0
        ratio = float(np.mean(xx / (xx + ss)))

    if L is not None:
        L = np.asarray(L, dtype=np.float64)
        lsp = np.log(np.maximum(def_sp, 1e-300))
        if lsp0 is None:
            lsp0 = np.zeros(L.shape[0], dtype=np.float64)
        else:
            lsp0 = np.asarray(lsp0, dtype=np.float64).ravel()
        if lsp0.shape != (L.shape[0],):
            return None
        rhs = np.asarray(lsp - lsp0, dtype=np.float64)
        coef, *_ = np.linalg.lstsq(L, rhs, rcond=None)
        def_sp = np.exp(np.asarray(coef, dtype=np.float64))

    return np.maximum(np.asarray(def_sp, dtype=np.float64), 1e-12)


def _initial_smoothing_params_from_design(model, y):
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    n_sp = _n_smoothing_params(model)
    if not penalty_blocks or n_sp == 0:
        return None

    family_class = str(
        getattr(getattr(model, "family", None), "family_class", "")
    ).lower()
    if family_class == "general":
        try:
            from ....families.gamlss._base import (
                _qr_coef_pivoted,
                scipy_qr,
            )
            from ....linalg import upper_triangular_rrank
            from ...solvers.general_family.fixed_smoothing import (
                build_general_family_setup_state,
            )
            from ..reparam import build_estimate_gam_setup_state

            def _r_default_matrix_norm(x):
                x = np.asarray(x, dtype=np.float64)
                if x.ndim != 2 or x.size == 0:
                    return float(np.max(np.abs(x))) if x.size else 0.0
                return float(np.max(np.sum(np.abs(x), axis=0)))

            def _pen_reg_initial_spg(X, E, y):
                # Mirror `mgcv/R/gamlss.r::pen.reg()` inside
                # `mgcv::initial.spg()`. This path needs R's default
                # `norm(R)`/`norm(E)` behavior (matrix type "O": maximum column
                # sum), not the row-sum helper used elsewhere in NAMpy.
                X = np.asarray(X, dtype=np.float64)
                E = np.asarray(E, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)

                if float(np.sum(np.abs(E))) == 0.0:
                    return _qr_coef_pivoted(X, y)

                Qx, R_piv, piv = scipy_qr(
                    X,
                    mode="economic",
                    pivoting=True,
                    check_finite=False,
                )
                r = int(R_piv.shape[1])
                rr = upper_triangular_rrank(R_piv)

                R = np.zeros_like(R_piv)
                R[:, np.asarray(piv, dtype=int)] = R_piv
                Qy = np.asarray(Qx.T @ y, dtype=np.float64)[:r]

                norm_R = _r_default_matrix_norm(R)
                norm_E = _r_default_matrix_norm(E)
                if not np.isfinite(norm_R) or norm_R <= 0.0:
                    return np.zeros(r, dtype=np.float64)
                if not np.isfinite(norm_E) or norm_E <= 0.0:
                    return _qr_coef_pivoted(X, y)

                k = 0.01 * norm_R / norm_E

                def _qrr_stats(k_scale):
                    A = np.vstack([R, E * float(k_scale)])
                    Q, Rq, pivq = scipy_qr(
                        A,
                        mode="economic",
                        pivoting=True,
                        check_finite=False,
                    )
                    edf = float(np.sum(np.asarray(Q[:r, :], dtype=np.float64) ** 2))
                    rank_q = upper_triangular_rrank(Rq)
                    return A, Q, Rq, pivq, edf, rank_q

                A, Qq, Rq, pivq, edf, rank_q = _qrr_stats(k)
                re = (
                    min(int(np.sum(np.sum(np.abs(E), axis=0) != 0.0)), int(E.shape[0]))
                    - rank_q
                    + rr
                )

                while edf > rr - 0.1 * re:
                    k *= 10.0
                    A, Qq, Rq, pivq, edf, rank_q = _qrr_stats(k)

                while edf < 0.85 * rr:
                    k /= 5.0
                    A, Qq, Rq, pivq, edf, rank_q = _qrr_stats(k)

                coef = _qr_coef_pivoted(
                    A,
                    np.concatenate([Qy, np.zeros(E.shape[0], dtype=np.float64)]),
                )
                coef[~np.isfinite(coef)] = 0.0
                return coef

            fit5_setup = build_general_family_setup_state(
                model,
                np.ones(n_sp, dtype=np.float64),
                score_type="REML",
            )
            exact_setup = build_estimate_gam_setup_state(model)
            X = np.asarray(fit5_setup.X_initial, dtype=np.float64).copy()
            E_init = np.asarray(exact_setup.Eb, dtype=np.float64).copy()
            weights = (
                np.ones_like(np.asarray(y, dtype=np.float64).ravel(), dtype=np.float64)
                if getattr(model, "prior_weights_", None) is None
                else np.asarray(model.prior_weights_, dtype=np.float64).ravel()
            )
            yv = np.asarray(y, dtype=np.float64).ravel()

            if str(getattr(model.family, "name", "")).lower() == "gaulss":
                start = np.zeros(X.shape[1], dtype=np.float64)
                X1 = np.asarray(X[:, fit5_setup.jj[0]], dtype=np.float64)
                E1 = np.asarray(E_init[:, fit5_setup.jj[0]], dtype=np.float64)
                yt1 = yv.copy()
                start1 = _pen_reg_initial_spg(X1, E1, yt1)
                start[np.asarray(fit5_setup.jj[0], dtype=int)] = start1

                mu_init = np.asarray(
                    model.family.linfo[0].linkinv(X1 @ start1),
                    dtype=np.float64,
                )
                lres1 = np.log(np.maximum(np.abs(yv - mu_init), 1e-300))
                X2 = np.asarray(X[:, fit5_setup.jj[1]], dtype=np.float64)
                E2 = np.asarray(E_init[:, fit5_setup.jj[1]], dtype=np.float64)
                start2 = _pen_reg_initial_spg(X2, E2, lres1)
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
            def_sp = np.zeros(len(exact_setup.S), dtype=np.float64)

            for i, S_i in enumerate(exact_setup.S):
                S_i = np.asarray(S_i, dtype=np.float64)
                if S_i.size == 0:
                    continue
                start_idx = int(exact_setup.off[i]) - 1
                stop = start_idx + int(S_i.shape[1])
                if start_idx < 0 or stop > lbb.shape[0]:
                    continue
                block_lbb = np.asarray(
                    lbb[start_idx:stop, start_idx:stop], dtype=np.float64
                )
                rank_i = int(exact_setup.rank[i])

                if rank_i < S_i.shape[1]:
                    _R, piv, _rank_p = pivoted_cholesky(S_i)
                    Z = np.asarray(S_i[:, piv[:rank_i]], dtype=np.float64)
                    zn = _r_default_matrix_norm(Z)
                    if not np.isfinite(zn) or zn <= 0.0:
                        continue
                    Z = Z / zn
                    ZHZ = -np.asarray(Z.T @ block_lbb @ Z, dtype=np.float64)
                    ZSZ = np.asarray(Z.T @ S_i @ Z, dtype=np.float64)
                else:
                    ZHZ = -np.asarray(block_lbb, dtype=np.float64)
                    ZSZ = np.asarray(S_i, dtype=np.float64)

                num = r_matrix_norm_max_abs(ZHZ)
                den = r_matrix_norm_max_abs(ZSZ)
                if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
                    continue
                def_sp[i] = 0.3 * num / den

            ok = def_sp > 0.0
            if not np.any(ok):
                return None
            def_sp[~ok] = 1.0
            if exact_setup.L is not None:
                L = np.asarray(exact_setup.L, dtype=np.float64)
                lsp0 = np.asarray(exact_setup.lsp0, dtype=np.float64).ravel()
                lsp = np.log(np.maximum(def_sp, 1e-300))
                rhs = np.asarray(lsp - lsp0, dtype=np.float64)
                coef, *_ = np.linalg.lstsq(L, rhs, rcond=None)
                def_sp = np.exp(np.asarray(coef, dtype=np.float64))
            return np.maximum(def_sp, 1e-12)
        except Exception:
            return None

    from ..reparam import build_estimate_gam_setup_state

    setup = build_estimate_gam_setup_state(model)
    X = np.asarray(setup.X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    nobs, _q = X.shape

    prior_weights = getattr(model, "prior_weights_", None)
    if prior_weights is None:
        prior_w = np.ones(nobs, dtype=np.float64)
    else:
        prior_w = np.asarray(prior_weights, dtype=np.float64).ravel()
        if prior_w.shape != (nobs,) or np.any(~np.isfinite(prior_w)):
            return None
    try:
        try:
            mu0 = np.asarray(
                model.family.initialize_mu(y, weights=prior_w), dtype=np.float64
            )
        except TypeError:
            mu0 = np.asarray(model.family.initialize_mu(y), dtype=np.float64)
        eta0 = np.asarray(model.family.link(mu0), dtype=np.float64)
        mu_eta = np.asarray(model.family.mu_eta(eta0), dtype=np.float64)
        var_mu = np.asarray(model.family.variance(mu0), dtype=np.float64)
    except Exception:
        return None

    # mgcv::initial.spg for ordinary families:
    #   w <- sqrt(weights * mu.eta(eta)^2 / variance(mu))
    w = np.sqrt(
        np.clip(prior_w * mu_eta * mu_eta / np.maximum(var_mu, 1e-12), 1e-12, None)
    )
    return _initial_smoothing_params_from_packed_penalties(
        w[:, None] * X,
        [np.asarray(S_i, dtype=np.float64) for S_i in list(setup.S)],
        np.asarray(setup.off, dtype=np.int64),
        L=(
            None
            if getattr(setup, "L", None) is None
            else np.asarray(setup.L, dtype=np.float64)
        ),
        lsp0=np.asarray(setup.lsp0, dtype=np.float64),
    )
