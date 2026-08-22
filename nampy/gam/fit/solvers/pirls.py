"""
Model-level entry point for PIRLS fits.
"""

import numpy as np
from scipy.linalg import solve_triangular

from ...model_state import (
    _coef_column_offset,
    _design_matrix,
    _fit_intercept,
    _fit_workspace,
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ..penalized_system import build_full_design, build_full_penalty_from_blocks
from ..state import FitCoreSolution
from .irls_core import (
    _mgcv_null_coef,
    irls_core,
)
from .stacked_qr import (
    _pivoted_economic_qr,
    _scatter_pivoted_rank_matrix_to_full,
    build_penalized_qr_state_nonnegative,
)

_MGCV_GAM_FIT3_RANK_TOL = float(np.finfo(np.float64).eps * 100.0)
_MGCV_GAM_FIT4_RANK_TOL = float(np.finfo(np.float64).eps**0.75)


def _canonical_penalty_root_stack(model, canonical, q_full):
    """Pack current-sp roots as expected by mgcv/src/gdi.c::gdiPK()."""
    q_range = int(q_full - int(canonical.Mp))
    roots = []
    for root in list(canonical.rp.get("rS", []))[
        : int(_n_smoothing_params(model) or 0)
    ]:
        root = np.asarray(root, dtype=np.float64)
        root_full = np.zeros((q_full, int(root.shape[1])), dtype=np.float64)
        if root.size:
            root_full[:q_range, :] = root
        roots.append(root_full)
    if not roots:
        return np.empty((q_full, 0), dtype=np.float64)
    return np.concatenate(roots, axis=1)


def _pirls_gdi_report_state(
    model,
    X,
    smoothing_params,
    sol,
    *,
    canonical=None,
    public_penalty=None,
):
    """Apply gam.fit3/gam.fit4's final canonical coefficient and rV gauge."""
    from ..selection.reparam import build_penalty_reparameterization_state

    X = np.asarray(X, dtype=np.float64)
    sp = np.asarray(smoothing_params, dtype=np.float64).ravel()
    if canonical is None:
        canonical = build_penalty_reparameterization_state(model, X, sp, deriv=0)
    transform = np.asarray(canonical.T, dtype=np.float64)
    X_canonical = np.asarray(X @ transform, dtype=np.float64)
    q_full = int(X_canonical.shape[1])

    working_weights = np.asarray(sol["working_weights"], dtype=np.float64)
    working_response = np.asarray(sol["working_response"], dtype=np.float64)
    good = (
        np.isfinite(working_weights)
        & (working_weights != 0.0)
        & np.isfinite(working_response)
    )
    if not np.any(good):
        raise RuntimeError(
            "PIRLS finalization has no informative observations in the "
            "current mgcv working system."
        )

    rank_tol = (
        _MGCV_GAM_FIT4_RANK_TOL
        if str(getattr(model.family, "family_class", "")).lower() == "extended"
        else _MGCV_GAM_FIT3_RANK_TOL
    )
    qr_state = build_penalized_qr_state_nonnegative(
        X_canonical[good, :],
        working_response[good],
        working_weights[good],
        penalty_sqrt_E=np.asarray(canonical.Sr, dtype=np.float64),
        penalty_rank_Es=np.asarray(canonical.Eb, dtype=np.float64),
        rS=_canonical_penalty_root_stack(model, canonical, q_full),
        rank_tol=rank_tol,
        reml=True,
    )

    coef_canonical = np.asarray(qr_state.beta_full, dtype=np.float64)
    coef_full = np.asarray(transform @ coef_canonical, dtype=np.float64)

    # mgcv/src/gdi.c::{gdi1,gdi2} first fixes the rank/drop/pivot gauge using
    # the Newton system above. It then builds rV from Fisher weights on exactly
    # that reduced, pivoted parameter space (gdi.c:2253-2292, 2772-2824); it
    # does not run a second full-space rank reveal that may drop another alias.
    kept = np.asarray(qr_state.kept_original_indices, dtype=np.int64)
    pivot1 = np.asarray(qr_state.pivot1, dtype=np.int64)
    ordered = kept[pivot1]
    rank = int(qr_state.rank)
    fisher_weights = np.asarray(sol["fisher_weights"], dtype=np.float64)
    fisher_good = np.asarray(fisher_weights[good], dtype=np.float64)
    if np.any(~np.isfinite(fisher_good)) or np.any(fisher_good <= 0.0):
        raise RuntimeError(
            "PIRLS finalization requires finite positive Fisher weights for "
            "the informative observations."
        )

    X_rank = np.asarray(X_canonical[good, :][:, ordered], dtype=np.float64)
    E_rank = np.asarray(canonical.Sr, dtype=np.float64)[:, ordered]
    augmented = np.vstack(
        [np.sqrt(fisher_good)[:, None] * X_rank, E_rank]
    )
    _q_cov, r_cov, pivot_cov = _pivoted_economic_qr(augmented)
    upper_r = np.triu(np.asarray(r_cov[:rank, :rank], dtype=np.float64))
    inverse_r = solve_triangular(
        upper_r,
        np.eye(rank, dtype=np.float64),
        lower=False,
        check_finite=False,
    )
    root_rank = _scatter_pivoted_rank_matrix_to_full(
        inverse_r,
        tuple(range(rank)),
        np.asarray(pivot_cov[:rank], dtype=np.int64),
        rank,
    )
    root_canonical = np.zeros((q_full, rank), dtype=np.float64)
    root_canonical[ordered, :] = root_rank
    covariance_root = np.asarray(transform @ root_canonical, dtype=np.float64)

    offset = getattr(model, "offset_train_", None)
    eta = np.asarray(X @ coef_full, dtype=np.float64)
    if offset is not None:
        eta = eta + np.asarray(offset, dtype=np.float64).ravel()
    mu = np.asarray(model.family.inverse_link(eta), dtype=np.float64)
    scale = float(sol["scale"])
    Vp = np.asarray(scale * (covariance_root @ covariance_root.T), dtype=np.float64)
    Vp = 0.5 * (Vp + Vp.T)
    fisher_full = np.asarray(sol["fisher_weights"], dtype=np.float64)
    XtWX = np.asarray(X.T @ (fisher_full[:, None] * X), dtype=np.float64)
    A_inv = np.asarray(covariance_root @ covariance_root.T, dtype=np.float64)
    H_coef = np.asarray(A_inv @ XtWX, dtype=np.float64)
    Vf = np.asarray(H_coef @ Vp, dtype=np.float64)
    trace_H = float(
        np.sum(
            (
                np.sqrt(np.maximum(fisher_full, 0.0))[:, None]
                * (X @ covariance_root)
            )
            ** 2
        )
    )

    sol["coef_full"] = coef_full.copy()
    sol["coef"] = coef_full.copy()
    sol["eta"] = eta.copy()
    sol["linear_predictor"] = eta.copy()
    sol["mu"] = mu.copy()
    sol["cov_bayes"] = Vp
    sol["cov_freq"] = Vf
    sol["A_inv"] = A_inv
    sol["XtWX"] = XtWX
    if public_penalty is None:
        public_penalty = np.asarray(sol["P"], dtype=np.float64)
    public_penalty = np.asarray(public_penalty, dtype=np.float64)
    sol["A"] = np.asarray(XtWX + public_penalty, dtype=np.float64)
    sol["P"] = public_penalty.copy()
    sol["penalty_matrix"] = public_penalty.copy()
    sol["X"] = X.copy()
    sol["H_coef"] = H_coef
    sol["trace_H"] = trace_H
    sol["edf"] = trace_H
    sol["penalized_system_rank"] = rank
    sol["dropped_column_indices"] = np.asarray(qr_state.drop, dtype=np.int64)
    sol["penalty_quadratic"] = float(
        coef_full @ (public_penalty @ coef_full)
    )
    if _fit_intercept(model) and coef_full.size:
        sol["intercept"] = float(coef_full[0])
        sol["beta"] = coef_full[1:].copy()
    else:
        sol["intercept"] = 0.0
        sol["beta"] = coef_full.copy()
    return sol


def solve_pirls_fit(
    model,
    y,
    smoothing_params,
    weights=None,
    *,
    scale_reference: float | None = None,
):
    coef_start = _fit_workspace(model).get("pirls_eval_start", None)
    if coef_start is None:
        coef_start = _fit_workspace(model).get("pirls_coef_start", None)
    if coef_start is not None:
        coef_start = np.asarray(coef_start, dtype=np.float64).ravel()
        Z = np.asarray(_design_matrix(model), dtype=np.float64)
        if coef_start.shape != (int(Z.shape[1] + _coef_column_offset(model)),):
            coef_start = None

    etastart = _fit_workspace(model).get("pirls_eval_eta_start", None)
    if etastart is None:
        etastart = _fit_workspace(model).get("pirls_eta_start", None)
    if etastart is not None:
        etastart = np.asarray(etastart, dtype=np.float64).ravel()
        if etastart.shape != (int(model.n_samples_),):
            etastart = None

    mustart = _fit_workspace(model).get("pirls_eval_mu_start", None)
    if mustart is None:
        mustart = _fit_workspace(model).get("pirls_mu_start", None)
    if mustart is not None:
        mustart = np.asarray(mustart, dtype=np.float64).ravel()
        if mustart.shape != (int(model.n_samples_),):
            mustart = None

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
    from ..selection.reparam import (
        build_penalty_reparameterization_state,
    )

    canonical = build_penalty_reparameterization_state(
        model,
        X,
        np.asarray(smoothing_params, dtype=np.float64),
        deriv=0,
    )
    transform = np.asarray(canonical.T, dtype=np.float64)
    X_canonical = np.asarray(X @ transform, dtype=np.float64)
    S_canonical = np.asarray(canonical.St, dtype=np.float64)

    if coef_start is not None:
        # mgcv/R/gam.fit3.r:169: start <- t(T) %*% start.
        coef_start = np.asarray(transform.T @ coef_start, dtype=np.float64)
    null_coef = np.asarray(
        transform.T @ _mgcv_null_coef(X, y, model.family),
        dtype=np.float64,
    )

    disable_theta_efs = bool(_fit_workspace(model).get("pirls_disable_theta_efs", False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    if disable_theta_efs:
        model.family._disable_theta_efs = True

    try:
        sol = irls_core(
            X_canonical,
            y,
            model.family,
            S_canonical,
            offset=model.offset_train_,
            weights=weights,
            fit_intercept=fi,
            max_iter=int(getattr(model, "max_irls_iter", 200)),
            tol=float(getattr(model, "irls_tol", 1e-7)),
            max_step_halving=int(getattr(model, "max_step_halving", 25)),
            coef_start=coef_start,
            null_coef=null_coef,
            etastart=etastart,
            mustart=mustart,
            penalty_sqrt_E=np.asarray(canonical.Sr, dtype=np.float64),
            penalty_rank_rows=np.asarray(canonical.Eb, dtype=np.float64),
            # `mgcv/R/gam.fit3.r::gam.fit3` uses its `scale` argument in the
            # PIRLS convergence test. Joint scale objectives must therefore
            # supply the current outer scale, rather than using an estimated
            # deviance surrogate inside the inner iteration.
            scale_reference=scale_reference,
        )
    finally:
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)

    sol = _pirls_gdi_report_state(
        model,
        X,
        smoothing_params,
        sol,
        canonical=canonical,
        public_penalty=S,
    )

    coef_out = np.asarray(sol["coef_full"], dtype=np.float64).copy()
    _fit_workspace(model).pirls_last_coef = coef_out
    _fit_workspace(model).pirls_last_eta = np.asarray(sol["eta"], dtype=np.float64).copy()
    _fit_workspace(model).pirls_last_mu = np.asarray(sol["mu"], dtype=np.float64).copy()
    _fit_workspace(model).pirls_last_inner_trace = list(sol.get("inner_trace", []) or [])
    if not bool(_fit_workspace(model).get("pirls_lock_start", False)):
        _fit_workspace(model).pirls_coef_start = coef_out.copy()
        _fit_workspace(model).pirls_eta_start = np.asarray(sol["eta"], dtype=np.float64).copy()
        _fit_workspace(model).pirls_mu_start = np.asarray(sol["mu"], dtype=np.float64).copy()
    return FitCoreSolution.from_dict(sol)
