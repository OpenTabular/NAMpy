import numpy as np

from ..._model_state import (
    _design_matrix,
    _fit_intercept,
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ..linalg.stacked_qr import (
    _scatter_pivoted_rank_matrix_to_full,
    balanced_penalty_template_sqrt_for_rank,
    build_penalized_qr_state_nonnegative,
    gaussian_design_needs_stacked_qr_fit,
    pls_fit1_nonneg_w,
    solve_gaussian_penalized_ls_stacked_qr,
)
from ..penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
)
from ..state import FitCoreSolution
from .irls_core import irls_core


def _prior_weights_vector(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(int(n), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (int(n),):
        raise ValueError(f"prior_weights must have shape ({n},), got {w.shape}.")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("prior_weights must be finite and non-negative.")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("prior_weights must sum to a positive value.")
    return w


def _penalized_system_needs_stacked_qr(
    X: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    *,
    cond_tol: float = 1e12,
) -> bool:
    """
    Detect numerically singular Gaussian penalized systems.

    Mirror the cases where mgcv's `magic`/`gam.fit3` covariance root drops
    effectively infinite-smoothing directions even though the raw design matrix
    itself is full rank.
    """
    XtWX = np.asarray(X.T @ (np.asarray(w, dtype=np.float64)[:, None] * X))
    A = np.asarray(XtWX + S, dtype=np.float64)
    if np.linalg.matrix_rank(A) < A.shape[0]:
        return True
    try:
        cond_A = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return True
    return (not np.isfinite(cond_A)) or cond_A > float(cond_tol)


def _gaussian_fit3_gdi_beta_full(model, X, smoothing_params, z_work, w):
    """
    Mirror `mgcv::gam.fit3()` post-loop `gdi1()` coefficient overwrite for Gaussian.

    `gdi1()` works on the current `gam.fit3` reparameterized system, not on the
    raw prediction-parameterization `sqrt(P)` solve.
    """
    from scipy.linalg.blas import dgemv

    from ...smoothing_selection.reparam import build_penalty_reparameterization_state

    X = np.asarray(X, dtype=np.float64)
    sp = np.asarray(smoothing_params, dtype=np.float64).ravel()
    z_work = np.asarray(z_work, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()

    canonical = build_penalty_reparameterization_state(model, X, sp, deriv=0)
    X_canon = np.asarray(
        X @ np.asarray(canonical.T, dtype=np.float64), dtype=np.float64
    )
    q_full = int(X_canon.shape[1])
    q_range = int(q_full - int(canonical.Mp))
    n_sp = int(_n_smoothing_params(model) or 0)

    root_blocks = []
    for root in list(canonical.rp.get("rS", []))[:n_sp]:
        root = np.asarray(root, dtype=np.float64)
        if root.size == 0:
            root_full = np.empty((q_full, 0), dtype=np.float64)
        else:
            root_full = np.zeros((q_full, int(root.shape[1])), dtype=np.float64)
            root_full[:q_range, :] = root
        root_blocks.append(root_full)
    rS = (
        np.concatenate(root_blocks, axis=1)
        if root_blocks
        else np.empty((q_full, 0), dtype=np.float64)
    )

    qr_state = build_penalized_qr_state_nonnegative(
        X_canon,
        z_work,
        w,
        penalty_sqrt_E=np.asarray(canonical.Sr, dtype=np.float64),
        penalty_rank_Es=np.asarray(canonical.Eb, dtype=np.float64),
        rS=rS,
        rank_tol=np.finfo(np.float64).eps * 100.0,
        reml=True,
    )
    coef_canon = np.asarray(qr_state.beta_full, dtype=np.float64)
    coef_full = np.asarray(
        np.asarray(canonical.T, dtype=np.float64) @ coef_canon,
        dtype=np.float64,
    )
    eta_fit = np.asarray(
        dgemv(1.0, np.asfortranarray(X_canon), coef_canon),
        dtype=np.float64,
    )
    # mgcv/src/gdi.c:2262-2292: the covariance root `rV` is built from the SAME
    # canonical factorization as the coefficients, with zero rows at dropped
    # canonical coordinates, then mapped back through `T`
    # (mgcv/R/gam.fit3.r:813-815). Building Vp from a separate natural-design
    # solve would put the drop gauge on a different column.
    rV_canon = _scatter_pivoted_rank_matrix_to_full(
        np.asarray(qr_state.P, dtype=np.float64),
        qr_state.kept_original_indices,
        qr_state.pivot1,
        q_full,
    )
    rV_full = np.asarray(
        np.asarray(canonical.T, dtype=np.float64) @ rV_canon, dtype=np.float64
    )
    return coef_full, eta_fit, rV_full


def _gaussian_fit3_pls_current_eta(model, X, smoothing_params, z_work, w):
    """
    Mirror `mgcv::gam.fit3()` inner-loop `eta <- x %*% start` on current-sp state.

    This is the Gaussian exact deviance retained on the fit object before the
    later `gdi1()` overwrite of the reported coefficients / fitted values.
    """
    from scipy.linalg.blas import dgemv

    from ...smoothing_selection.reparam import build_penalty_reparameterization_state

    X = np.asarray(X, dtype=np.float64)
    sp = np.asarray(smoothing_params, dtype=np.float64).ravel()
    z_work = np.asarray(z_work, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()

    canonical = build_penalty_reparameterization_state(model, X, sp, deriv=0)
    X_canon = np.asarray(
        X @ np.asarray(canonical.T, dtype=np.float64), dtype=np.float64
    )
    coef_canon, _penalty = pls_fit1_nonneg_w(
        X_canon,
        z_work,
        w,
        w * z_work,
        penalty_sqrt_E=np.asarray(canonical.Sr, dtype=np.float64),
        penalty_rank_Es=np.asarray(canonical.Eb, dtype=np.float64),
        rank_tol=np.finfo(np.float64).eps * 100.0,
    )
    return np.asarray(
        dgemv(1.0, np.asfortranarray(X_canon), coef_canon),
        dtype=np.float64,
    )


def solve_gaussian_fit(model, y, smoothing_params, weights=None):
    y = model.family.validate_y(y)
    n = int(y.shape[0])
    w = _prior_weights_vector(weights, n)
    fi = _fit_intercept(model)
    Z = np.asarray(_design_matrix(model), dtype=np.float64)
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    n_coef = _n_coef(model)
    X = build_full_design(Z, fit_intercept=fi)
    S = build_full_penalty_from_blocks(
        penalty_blocks=penalty_blocks,
        smoothing_params=smoothing_params,
        fit_intercept=fi,
        n_coef=n_coef,
    )
    rank_rows = balanced_penalty_template_sqrt_for_rank(
        penalty_blocks,
        fit_intercept=fi,
        n_coef=int(n_coef),
    )

    force_stacked_qr = (
        bool(getattr(model, "_use_stacked_qr", False))
        or gaussian_design_needs_stacked_qr_fit(model)
        or _penalized_system_needs_stacked_qr(X, w, S)
    )
    coef_method = "householder"

    # No null-space gauge pin: the coefficient and covariance representatives
    # are overwritten below from the canonical `gdi1` factorization, whose
    # dropped-coordinate zero-fill IS mgcv's rank-deficiency gauge
    # (mgcv/src/gdi.c:2253-2292). Upstream has no penalty-minimizing pin.
    null_gauge = False

    sol = irls_core(
        X,
        y,
        model.family,
        S,
        offset=model.offset_train_,
        weights=w,
        fit_intercept=fi,
        max_iter=1,
        tol=float(getattr(model, "irls_tol", 1e-7)),
        max_step_halving=int(getattr(model, "max_step_halving", 25)),
        penalty_rank_rows=rank_rows,
        force_stacked_qr=force_stacked_qr,
        coef_method=coef_method,
        near_singular_null_pin=null_gauge,
    )
    reported_deviance = float(sol["deviance"])

    eta = np.asarray(sol["eta"], dtype=np.float64)
    if force_stacked_qr:
        # Recompute EDF/covariance from stacked-QR rank-reduced factors. This mirrors
        # mgcv's `rV %*% t(rV)` post-processing for near-singular Gaussian REML fits
        # (for example `bs="re"` at the lambda boundary), instead of inverting the
        # singular full `X'WX + S`.
        y_eff = y if model.offset_train_ is None else (y - model.offset_train_)
        stacked = solve_gaussian_penalized_ls_stacked_qr(
            X,
            y_eff,
            w,
            S,
            penalty_blocks=penalty_blocks,
            fit_intercept=fi,
            n_coef=int(n_coef),
            coef_method=coef_method,
            near_singular_null_pin=null_gauge,
        )
        A = np.asarray(stacked["A"], dtype=np.float64)
        A_inv = np.asarray(stacked["A_inv"], dtype=np.float64)
        XtWX = np.asarray(stacked["XtWX"], dtype=np.float64)
        H_coef_stable = np.asarray(stacked["coef_hat_matrix"], dtype=np.float64)
        cov_root = np.asarray(stacked["covariance_root"], dtype=np.float64)
        coef_full = np.asarray(stacked["coef_full"], dtype=np.float64)
        # Mirror mgcv/R/gam.fit3.r: after the stable `pls_fit1()` / `gdi1()`
        # solve, the reported Gaussian linear predictor is recomputed as
        # `x %*% coef + offset`, not taken from the raw Householder eta buffer.
        eta = np.asarray(X @ coef_full, dtype=np.float64)
        if model.offset_train_ is not None:
            eta = eta + np.asarray(model.offset_train_, dtype=np.float64).ravel()
        mu = np.asarray(model.family.inverse_link(eta), dtype=np.float64)
        # Mirror mgcv's post-fit EDF trace from the stacked-QR covariance root
        # (`rV`) rather than from an explicit `A^{-1} X'WX` product. The two are
        # algebraically identical, but the root-based Frobenius form is
        # materially more stable in nearly saturated aliased designs such as
        # intercept + `bs="fs"`, where tiny EDF differences feed directly into
        # the Gaussian scale estimate.
        WX_rV = np.asarray(stacked["WX_sqrt"], dtype=np.float64) @ np.asarray(
            stacked["covariance_root"], dtype=np.float64
        )
        trace_H = float(np.sum(WX_rV * WX_rV))
        scale = model.family.estimate_dispersion(y, eta, edf=trace_H, weights=w)
        Vp = np.asarray(scale * (cov_root @ cov_root.T), dtype=np.float64)
        Vp = 0.5 * (Vp + Vp.T)
        H_coef = np.asarray(H_coef_stable, dtype=np.float64)
        Vf = np.asarray(H_coef @ Vp, dtype=np.float64)
        sol["A"] = A
        sol["A_inv"] = A_inv
        sol["XtWX"] = XtWX
        sol["log_det_XtWX_plus_penalty"] = float(stacked["log_det_XtWX_plus_penalty"])
        sol["trace_H"] = trace_H
        sol["edf"] = trace_H
        sol["cov_bayes"] = Vp
        sol["cov_freq"] = Vf
        sol["H_coef"] = H_coef
        # Mirror `mgcv/R/gam.fit3.r`: for stacked-QR Gaussian fits, the
        # current-sp `pls_fit1()` / post-loop `gdi1()` state is evaluated on the
        # reparameterized system, and its `eta <- x %*% start` BLAS multiply is
        # also the deviance path retained on the fit object for exact Gaussian
        # REML/ML scoring.
        eta_dev = _gaussian_fit3_pls_current_eta(
            model,
            X,
            smoothing_params,
            sol["working_response"],
            w,
        )
        coef_full, eta_fit, gdi_rank_root = _gaussian_fit3_gdi_beta_full(
            model,
            X,
            smoothing_params,
            sol["working_response"],
            w,
        )
        if model.offset_train_ is not None:
            eta_dev = (
                eta_dev + np.asarray(model.offset_train_, dtype=np.float64).ravel()
            )
        mu_dev = np.asarray(model.family.inverse_link(eta_dev), dtype=np.float64)
        reported_deviance = float(model.family.deviance(y, mu_dev, weights=w))
        eta = np.asarray(eta_fit, dtype=np.float64)
        if model.offset_train_ is not None:
            eta = eta + np.asarray(model.offset_train_, dtype=np.float64).ravel()
        mu = np.asarray(model.family.inverse_link(eta), dtype=np.float64)
        scale = model.family.estimate_dispersion(y, eta, edf=trace_H, weights=w)
        # mgcv/R/gam.fit3.r:959 `Vb <- rV %*% t(rV) * scale` with `rV` from the
        # same canonical gdi1 factorization as the coefficients, so the
        # rank-deficiency gauge (zero row at dropped canonical coordinates)
        # lands on the same position in coef and Vp.
        Vp = np.asarray(
            scale * (gdi_rank_root @ gdi_rank_root.T), dtype=np.float64
        )
        Vp = 0.5 * (Vp + Vp.T)
        H_coef = np.asarray(H_coef_stable, dtype=np.float64)
        Vf = np.asarray(H_coef @ Vp, dtype=np.float64)
        sol["cov_bayes"] = Vp
        sol["cov_freq"] = Vf
        sol["H_coef"] = H_coef
        sol["penalty_quadratic"] = float(coef_full @ (S @ coef_full))
        sol["coef_full"] = coef_full.copy()
        sol["coef"] = coef_full.copy()
        sol["eta"] = eta.copy()
        sol["linear_predictor"] = eta.copy()
        sol["mu"] = mu.copy()
        if fi and coef_full.size:
            sol["intercept"] = float(coef_full[0])
            sol["beta"] = np.asarray(coef_full[1:], dtype=np.float64).copy()
        else:
            sol["intercept"] = 0.0
            sol["beta"] = coef_full.copy()
        sol["penalty_quadratic"] = float(
            sol.get("penalty_quadratic", stacked["penalty_quadratic"])
        )
    else:
        scale = model.family.estimate_dispersion(y, eta, edf=sol["trace_H"], weights=w)
    resid = y - eta
    wrss = float(np.sum(w * resid * resid))
    sol["rss"] = wrss
    sol["deviance"] = reported_deviance
    sol["scale"] = float(scale)
    sol["working_weights"] = w.copy()
    sol["fisher_weights"] = w.copy()
    sol["working_response"] = (
        y.copy() if model.offset_train_ is None else (y - model.offset_train_).copy()
    )
    # The Gaussian exact solver operates on the fit matrix assembled by
    # gam.setup. If setup also built a distinct prediction matrix, mgcv applies
    # `G$P` after fitting (mgcv/R/mgcv.r) to move coefficients/covariances into
    # the public prediction parameterization.
    sol["coef_space"] = "fit"
    sol["cov_bayes_space"] = "fit"
    sol["cov_freq_space"] = "fit"
    sol["loglik"] = float(
        np.sum(
            w
            * np.asarray(
                model.family.loglik_obs(y, eta, scale=scale),
                dtype=np.float64,
            )
        )
    )

    return FitCoreSolution.from_dict(sol)
