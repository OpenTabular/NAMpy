import numpy as np

from ...linalg import balanced_penalty_template_sqrt_for_rank
from ...model_state import (
    _design_matrix,
    _fit_intercept,
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ..penalized_system import (
    build_full_design,
    build_full_penalty_from_blocks,
)
from ..state import FitCoreSolution
from .irls_core import irls_core
from .stacked_qr import (
    _scatter_pivoted_rank_matrix_to_full,
    build_penalized_qr_state_nonnegative,
    pls_fit1_nonneg_w,
)


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


def _gaussian_fit3_gdi_beta_full(
    model,
    X,
    smoothing_params,
    z_work,
    w,
    *,
    return_hat_matrix=False,
):
    """
    Mirror `mgcv::gam.fit3()` post-loop `gdi1()` coefficient overwrite for Gaussian.

    `gdi1()` works on the current `gam.fit3` reparameterized system, not on the
    raw prediction-parameterization `sqrt(P)` solve.
    """
    from ..selection.reparam import build_penalty_reparameterization_state

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
    eta_fit = np.asarray(X_canon @ coef_canon, dtype=np.float64)
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
    if not return_hat_matrix:
        return coef_full, eta_fit, rV_full
    if np.any(w < 0.0):
        raise ValueError("Gaussian post-fit weights must be non-negative.")

    # mgcv/R/gam.fit3.r::gam.fit3.post.proc (lines 961-965): retain the
    # Householder ``K`` returned by gdi1 and form
    #
    #   F <- (rV %*% t(K)) %*% (sqrt(weights) * X)
    #
    # in that operand order.  ``rV rV' X'WX`` is algebraically equivalent,
    # but catastrophically loses the null-space cancellation for intercept +
    # ``bs="re"`` when lambda is at the near-zero REML boundary.
    PKt = np.asarray(rV_full @ np.asarray(qr_state.K, dtype=np.float64).T)
    H_coef = np.asarray(PKt @ (np.sqrt(w)[:, None] * X), dtype=np.float64)
    return coef_full, eta_fit, rV_full, H_coef


def _gaussian_fit3_pls_current_eta(model, X, smoothing_params, z_work, w):
    """
    Mirror `mgcv::gam.fit3()` inner-loop `eta <- x %*% start` on current-sp state.

    This is the Gaussian exact deviance retained on the fit object before the
    later `gdi1()` overwrite of the reported coefficients / fitted values.
    """
    from ..selection.reparam import build_penalty_reparameterization_state

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
    return np.asarray(X_canon @ coef_canon, dtype=np.float64)


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
    )

    # Mirror gam.fit3's unconditional current-sp `pls_fit1()` state followed by
    # the post-loop `gdi1()` coefficient/covariance overwrite. Upstream does not
    # select this path from a condition number or from the model's term types.
    eta_dev = _gaussian_fit3_pls_current_eta(
        model,
        X,
        smoothing_params,
        sol["working_response"],
        w,
    )
    coef_full, eta_fit, gdi_rank_root, H_coef = _gaussian_fit3_gdi_beta_full(
        model,
        X,
        smoothing_params,
        sol["working_response"],
        w,
        return_hat_matrix=True,
    )
    if model.offset_train_ is not None:
        train_offset = np.asarray(model.offset_train_, dtype=np.float64).ravel()
        eta_dev = eta_dev + train_offset
        eta_fit = eta_fit + train_offset
    mu_dev = np.asarray(model.family.inverse_link(eta_dev), dtype=np.float64)
    reported_deviance = float(model.family.deviance(y, mu_dev, weights=w))
    eta = np.asarray(eta_fit, dtype=np.float64)
    mu = np.asarray(model.family.inverse_link(eta), dtype=np.float64)
    H_coef = np.asarray(H_coef, dtype=np.float64)
    trace_H = float(np.trace(H_coef))
    scale = model.family.estimate_dispersion(y, eta, edf=trace_H, weights=w)
    # mgcv/R/gam.fit3.r::gam.fit3.post.proc forms Vb from the `rV` returned by
    # this same gdi1 factorization, retaining its dropped-coordinate zero fill.
    Vp = np.asarray(scale * (gdi_rank_root @ gdi_rank_root.T), dtype=np.float64)
    Vp = 0.5 * (Vp + Vp.T)
    Vf = np.asarray(H_coef @ Vp, dtype=np.float64)
    sol["cov_bayes"] = Vp
    sol["cov_freq"] = Vf
    sol["H_coef"] = H_coef
    sol["trace_H"] = trace_H
    sol["edf"] = trace_H
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
