"""Exact first/second derivatives of PIRLS Laplace ML/REML criteria."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, polygamma

from ..._model_state import _coef_column_offset, _n_smoothing_params
from ...fit.linalg.matrix_reindexing import (
    drop_columns_dense,
    drop_rows_dense,
    permute_columns,
    permute_rows,
)
from ...fit.linalg.stacked_qr import (
    build_penalized_qr_state_nonnegative,
    penalty_sqrt_rows,
)
from ...fit.model_ops import (
    can_use_simple_ml_reml_structure,
    expand_smoothing_params_from_log,
    solve_pirls_given_smoothing,
)
from ..reparam import (
    _stable_penalty_logdet_derivatives,
    _static_penalty_null_dim,
    build_canonical_gam_reparam_state,
)
from .pirls import _gamma_profile_objective_curvature, _solve_gamma_profile_scale
from .pirls_reml_derivative_blocks import (
    _deviance_chained_to_smoothing,
    _deviance_coefficient_derivatives,
    _hat_matrix_trace_and_sp_derivatives,
    _logdet_penalized_system_derivatives,
    _penalty_quadratic_and_sp_derivatives,
    _quadratic_form_in_beta_directions,
    _working_weight_derivatives_wrt_linpred,
)


@dataclass
class _GDI1CurrentSpState:
    """Current-sp canonical state corresponding to `gam.fit4()` + `gdiPK()` setup."""

    canonical: object
    X: np.ndarray
    beta: np.ndarray
    W: np.ndarray
    XtWX: np.ndarray
    P: np.ndarray
    A: np.ndarray
    A_inv: np.ndarray
    penalized_system_rank: int
    dropped_column_indices: np.ndarray
    pivot1: np.ndarray


@dataclass
class _GDI1IFTState:
    """Implicit-function derivatives corresponding to `mgcv::ift1()` outputs."""

    P_derivs: list[np.ndarray]
    dbeta: list[np.ndarray]
    deta: list[np.ndarray]
    dA: list[np.ndarray]
    dXtWX: list[np.ndarray]
    d2beta_mat: list[list[np.ndarray]]
    d2A_mat: list[list[np.ndarray]]
    d2XtWX_mat: list[list[np.ndarray]]
    dW_obs: list[np.ndarray] | None = None
    d2W_obs_mat: list[list[np.ndarray]] | None = None


@dataclass
class _GDI1Kernel:
    """Python `gdi1` decomposition on canonical `U1/UrS/rp/T/St/Sr/Eb/Mp` state."""

    current: _GDI1CurrentSpState
    ift: _GDI1IFTState
    D1: np.ndarray
    D2: np.ndarray
    bSb: float
    bSb1: np.ndarray
    bSb2: np.ndarray
    det1: np.ndarray
    det2: np.ndarray
    trA: float
    trA1: np.ndarray
    trA2: np.ndarray
    K: float
    K1: np.ndarray
    K2: np.ndarray
    dVkk: np.ndarray


@dataclass
class _GDIPKState:
    """Python analogue of `gdiPK()` setup after current-sp canonical reparameterization."""

    q_total: int
    q_range: int
    q_null: int
    rows_E: int
    rank: int
    range_idx: np.ndarray
    null_idx: np.ndarray
    kept_idx: np.ndarray
    dropped_idx: np.ndarray
    n_drop: int
    pivot1: np.ndarray
    PKtz: np.ndarray
    K: np.ndarray | None
    P: np.ndarray | None
    Rh: np.ndarray | None
    rS_work: np.ndarray | None
    ldet_XWX_plus_S: float | None


@dataclass
class _GDIPKSetup:
    """Single `gdiPK()`-shaped setup bundle for canonical state plus rank metadata."""

    current: _GDI1CurrentSpState
    pk: _GDIPKState


@dataclass
class _GDI2Kernel:
    """Extended-family / joint-parameter analogue of `mgcv::gdi2()` kernel state."""

    gdi1: _GDI1Kernel
    phi: float | None
    phi_curv: float | None
    Dp: float
    Dp1: np.ndarray
    Dp2: np.ndarray | None
    ift: "_GDI2IFTState | None" = None
    D1_full: np.ndarray | None = None
    D2_full: np.ndarray | None = None
    K1_full: np.ndarray | None = None
    K2_full: np.ndarray | None = None
    extra_name: str | None = None
    extra_value: float | None = None


@dataclass
class _GDI2IFTState:
    """`mgcv::ift2()`-shaped joint derivative state over `[log(theta), log(sp)]`."""

    P_derivs: list[np.ndarray]
    dbeta: list[np.ndarray]
    deta: list[np.ndarray]
    dA: list[np.ndarray]
    dXtWX: list[np.ndarray]
    d2beta_mat: list[list[np.ndarray]]
    d2A_mat: list[list[np.ndarray]]
    d2XtWX_mat: list[list[np.ndarray]]
    ntheta: int


def _gamma_saturated_loglik_scale_derivatives(y, phi, weights=None):
    """
    `mgcv`-style Gamma saturated log-likelihood derivatives w.r.t. scale `phi`.

    Mirrors `mgcv::fix.family.ls()` Gamma derivatives.
    """
    y = np.asarray(y, dtype=np.float64)
    phi = float(phi)
    if not np.isfinite(phi) or phi <= 0.0:
        return np.nan, np.nan
    if weights is None:
        weights = np.ones_like(y, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    mask = weights > 0.0
    if not np.any(mask):
        return 0.0, 0.0
    w = weights[mask]
    scale_w = phi / w
    psi0 = digamma(1.0 / scale_w)
    psi1 = polygamma(1, 1.0 / scale_w)
    ls1 = np.sum((psi0 + np.log(scale_w)) / (scale_w**2 * w))
    ls2 = np.sum(
        (-psi1 / scale_w + (1.0 - 2.0 * np.log(scale_w) - 2.0 * psi0))
        / (scale_w**3 * (w**2))
    )
    return float(ls1), float(ls2)


def _prior_weights(model, y):
    w = getattr(model, "prior_weights_", None)
    if w is None:
        return np.ones_like(np.asarray(y, dtype=np.float64), dtype=np.float64)
    return np.asarray(w, dtype=np.float64)


def _drop_permute_columns(x, drop, pivot1):
    return permute_columns(drop_columns_dense(x, drop), pivot1, reverse=False)


def _drop_permute_symmetric(mat, drop, pivot1):
    out = drop_rows_dense(np.asarray(mat, dtype=np.float64), drop)
    out = drop_columns_dense(out, drop)
    out = permute_rows(out, pivot1, reverse=False)
    return permute_columns(out, pivot1, reverse=False)


def _negbin_ddeta_logtheta(family, y, mu, weights, *, deriv):
    """
    Port of `mgcv::nb()$Dd` + `dDeta()` for log-link negative binomial.

    Raw `Dd` ownership lives on family object. This helper only chains `Dmu`
    derivatives to `eta`, mirroring `gam.fit4.r::dDeta()`.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-14, None)
    wt = np.asarray(weights, dtype=np.float64)
    dd = family.Dd(y, mu, family.getTheta(False), wt, level=int(deriv))

    out = {
        "Deta": np.asarray(dd["Dmu"], dtype=np.float64) * mu,
        "Deta2": np.asarray(dd["Dmu2"], dtype=np.float64) * (mu**2)
        + np.asarray(dd["Dmu"], dtype=np.float64) * mu,
    }
    if deriv <= 0:
        return out

    out["Dth"] = np.asarray(dd["Dth"], dtype=np.float64)
    out["Detath"] = np.asarray(dd["Dmuth"], dtype=np.float64) * mu
    out["Deta3"] = (
        np.asarray(dd["Dmu3"], dtype=np.float64) * (mu**3)
        + 3.0 * np.asarray(dd["Dmu2"], dtype=np.float64) * (mu**2)
        + np.asarray(dd["Dmu"], dtype=np.float64) * mu
    )
    out["Deta2th"] = (
        np.asarray(dd["Dmu2th"], dtype=np.float64) * (mu**2)
        + np.asarray(dd["Dmuth"], dtype=np.float64) * mu
    )
    if deriv <= 1:
        return out

    out["Dth2"] = np.asarray(dd["Dth2"], dtype=np.float64)
    out["Detath2"] = np.asarray(dd["Dmuth2"], dtype=np.float64) * mu
    out["Deta4"] = (
        np.asarray(dd["Dmu4"], dtype=np.float64) * (mu**4)
        + 6.0 * np.asarray(dd["Dmu3"], dtype=np.float64) * (mu**3)
        + 7.0 * np.asarray(dd["Dmu2"], dtype=np.float64) * (mu**2)
        + np.asarray(dd["Dmu"], dtype=np.float64) * mu
    )
    out["Deta3th"] = (
        np.asarray(dd["Dmu3th"], dtype=np.float64) * (mu**3)
        + 3.0 * np.asarray(dd["Dmu2th"], dtype=np.float64) * (mu**2)
        + np.asarray(dd["Dmuth"], dtype=np.float64) * mu
    )
    out["Deta2th2"] = (
        np.asarray(dd["Dmu2th2"], dtype=np.float64) * (mu**2)
        + np.asarray(dd["Dmuth2"], dtype=np.float64) * mu
    )
    return out


def _gamma_joint_kernel_state(model, y, log_sp, method):
    method = str(method).upper()
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    gdi2 = _gdi2_joint_kernel(model, y, sol, sp, method=method, need_hessian=True)
    state = {
        "K": gdi2.gdi1.K,
        "K1": gdi2.gdi1.K1,
        "K2": gdi2.gdi1.K2,
        "phi": gdi2.phi,
        "phi_curv": gdi2.phi_curv,
        "scale_est": float(sol["scale"]),
        "Dp": gdi2.Dp,
        "Dp1": gdi2.Dp1,
        "Dp2": gdi2.Dp2,
    }
    model._pirls_reml_gamma_state_ = state
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    return state, mp


def criterion_gradient_ml_reml_pirls_gamma_joint(model, y, log_sp, log_phi, method):
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gamma":
        raise NotImplementedError(
            "Joint PIRLS Gamma derivatives are implemented only for family='gamma'."
        )

    state, mp = _gamma_joint_kernel_state(model, y, log_sp, method)
    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        n_free = (
            int(np.sum(~np.asarray(model.smoothing_fixed_mask_, dtype=bool)))
            if model.smoothing_fixed_mask_ is not None
            else int(_n_smoothing_params(model) or 0)
        )
        return np.full(n_free + 1, np.nan, dtype=np.float64)

    _, score_lphi, _ = _gamma_profile_objective_curvature(
        model,
        y,
        float(state["Dp"]),
        phi,
        mp,
        method=str(method).upper(),
    )
    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    gamma = float(model.score_gamma)
    grad_sp = np.asarray(state["Dp1"], dtype=np.float64) / (2.0 * phi * gamma) + (
        np.asarray(state["K1"], dtype=np.float64)
    )
    return np.concatenate(
        [
            np.asarray(grad_sp[free_mask], dtype=np.float64),
            np.array([float(score_lphi)], dtype=np.float64),
        ]
    )


def criterion_hessian_ml_reml_pirls_gamma_joint(model, y, log_sp, log_phi, method):
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gamma":
        raise NotImplementedError(
            "Joint PIRLS Gamma derivatives are implemented only for family='gamma'."
        )

    state, mp = _gamma_joint_kernel_state(model, y, log_sp, method)
    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        n_free = (
            int(np.sum(~np.asarray(model.smoothing_fixed_mask_, dtype=bool)))
            if model.smoothing_fixed_mask_ is not None
            else int(_n_smoothing_params(model) or 0)
        )
        return np.full((n_free + 1, n_free + 1), np.nan, dtype=np.float64)

    _, _, curv_lphi = _gamma_profile_objective_curvature(
        model,
        y,
        float(state["Dp"]),
        phi,
        mp,
        method=str(method).upper(),
    )
    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask

    gamma = float(model.score_gamma)
    H_sp = np.asarray(state["Dp2"], dtype=np.float64) / (2.0 * phi * gamma) + (
        np.asarray(state["K2"], dtype=np.float64)
    )
    cross = -np.asarray(state["Dp1"], dtype=np.float64) / (2.0 * phi * gamma)
    H_free = np.asarray(H_sp[np.ix_(free_mask, free_mask)], dtype=np.float64)
    cross_free = np.asarray(cross[free_mask], dtype=np.float64)
    out = np.zeros(
        (int(np.sum(free_mask)) + 1, int(np.sum(free_mask)) + 1), dtype=np.float64
    )
    out[:-1, :-1] = H_free
    out[:-1, -1] = cross_free
    out[-1, :-1] = cross_free
    out[-1, -1] = float(curv_lphi)
    return out


def _negbin_joint_kernel_state(model, y, log_sp, log_theta, method):
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "negbin":
        raise NotImplementedError(
            "Joint PIRLS NegBin derivatives are implemented only for family='negbin'."
        )
    theta = float(np.exp(float(log_theta)))
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError("log_theta must map to finite positive theta.")
    prev_log_theta = float(model.family.getTheta(False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    try:
        model.family.putTheta(float(log_theta))
        model.family._disable_theta_efs = True
        sp = expand_smoothing_params_from_log(model, log_sp)
        sol = solve_pirls_given_smoothing(model, y, sp)
        gdi2 = _gdi2_joint_kernel(
            model, y, sol, sp, method=str(method).upper(), need_hessian=True
        )
        state = {
            "theta": float(theta),
            "scale_est": float(sol["scale"]),
            "Dp": float(gdi2.Dp),
            "Dp1": np.asarray(gdi2.Dp1, dtype=np.float64),
            "Dp2": np.asarray(gdi2.Dp2, dtype=np.float64),
            "K1": np.asarray(gdi2.K1_full, dtype=np.float64),
            "K2": np.asarray(gdi2.K2_full, dtype=np.float64),
        }
        model._pirls_reml_negbin_state_ = state
        return state
    finally:
        model.family.putTheta(prev_log_theta)
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)


def criterion_gradient_ml_reml_pirls_negbin_joint(model, y, log_sp, log_theta, method):
    state = _negbin_joint_kernel_state(model, y, log_sp, log_theta, method)
    gamma = float(model.score_gamma)
    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    grad_full = np.asarray(state["Dp1"], dtype=np.float64) / (
        2.0 * float(state["scale_est"]) * gamma
    ) + np.asarray(state["K1"], dtype=np.float64)
    return np.concatenate(
        [
            np.asarray(grad_full[1:][free_mask], dtype=np.float64),
            np.array([float(grad_full[0])], dtype=np.float64),
        ]
    )


def criterion_hessian_ml_reml_pirls_negbin_joint(model, y, log_sp, log_theta, method):
    state = _negbin_joint_kernel_state(model, y, log_sp, log_theta, method)
    gamma = float(model.score_gamma)
    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    H_full = np.asarray(state["Dp2"], dtype=np.float64) / (
        2.0 * float(state["scale_est"]) * gamma
    ) + np.asarray(state["K2"], dtype=np.float64)
    sp_idx = 1 + np.flatnonzero(free_mask)
    keep = np.concatenate([sp_idx, np.array([0], dtype=np.int64)])
    return np.asarray(H_full[np.ix_(keep, keep)], dtype=np.float64)


def _gdi_pk_setup(model, sol, sp, *, deriv):
    """
    Single `mgcv::gdiPK()`-shaped setup routine.

    Owns current-sp canonical reparameterization plus solver rank/drop metadata.
    """
    canonical = build_canonical_gam_reparam_state(model, sol["X"], sp, deriv=deriv)
    X = np.asarray(sol["X"], dtype=np.float64) @ np.asarray(
        canonical.T, dtype=np.float64
    )
    beta = np.asarray(
        np.asarray(canonical.T, dtype=np.float64).T
        @ np.asarray(sol["coef_full"], dtype=np.float64),
        dtype=np.float64,
    )
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    XtWX = X.T @ (W[:, None] * X)
    P = np.asarray(canonical.St, dtype=np.float64)
    A = XtWX + P
    q_total_full = int(np.asarray(A, dtype=np.float64).shape[0])
    q_null_full = int(canonical.Mp)
    q_range_full = int(q_total_full - q_null_full)
    penalty_sqrt, penalty_rank_rows = penalty_sqrt_rows(np.asarray(P, dtype=np.float64))
    root_cols = []
    rSncol = []
    roots = list(canonical.rp.get("rS", []))
    rows_e = q_range_full
    for j in range(int(_n_smoothing_params(model) or 0)):
        if j < len(roots):
            root = np.asarray(roots[j], dtype=np.float64)
            if root.size:
                root_full = np.zeros(
                    (q_total_full, int(root.shape[1])), dtype=np.float64
                )
                root_full[:rows_e, :] = root
            else:
                root_full = np.empty((q_total_full, 0), dtype=np.float64)
        else:
            root_full = np.empty((q_total_full, 0), dtype=np.float64)
        root_cols.append(root_full)
        rSncol.append(int(root_full.shape[1]))
    rS = (
        np.concatenate(root_cols, axis=1)
        if root_cols
        else np.empty((q_total_full, 0), dtype=np.float64)
    )
    qr_state = build_penalized_qr_state_nonnegative(
        np.asarray(X, dtype=np.float64),
        np.asarray(X @ beta, dtype=np.float64),
        np.abs(np.asarray(W, dtype=np.float64)),
        penalty_sqrt_E=np.asarray(penalty_sqrt, dtype=np.float64),
        penalty_rank_Es=np.asarray(penalty_rank_rows, dtype=np.float64),
        rS=np.asarray(rS, dtype=np.float64),
        rank_tol=1e-10,
        reml=True,
        Mp=int(q_null_full),
    )
    rank = int(qr_state.rank)
    dropped_idx = np.asarray(qr_state.drop, dtype=np.int64)
    kept_mask = np.ones(q_total_full, dtype=bool)
    if dropped_idx.size:
        kept_mask[dropped_idx] = False
    kept_idx = np.flatnonzero(kept_mask).astype(np.int64, copy=False)
    pivot1 = np.asarray(qr_state.pivot1, dtype=np.int64)
    pk_tz = np.asarray(qr_state.PKtz, dtype=np.float64)
    K = np.asarray(qr_state.K, dtype=np.float64)
    Pk = np.asarray(qr_state.P, dtype=np.float64)
    Rh = np.asarray(qr_state.Rh, dtype=np.float64)
    rS_work = np.asarray(qr_state.rS_work, dtype=np.float64)
    ldet_xwxs = float(qr_state.ldet_XWX_plus_S)
    X_rank = _drop_permute_columns(X, dropped_idx, pivot1)
    beta_rank = permute_rows(
        drop_rows_dense(beta[:, None], dropped_idx), pivot1, reverse=False
    ).ravel()
    P_rank = _drop_permute_symmetric(P, dropped_idx, pivot1)
    XtWX_rank = X_rank.T @ (W[:, None] * X_rank)
    A_rank = XtWX_rank + P_rank
    cA, loA = cho_factor(A_rank, check_finite=False)
    A_inv_rank = cho_solve((cA, loA), np.eye(A_rank.shape[0]), check_finite=False)
    nulli_full = np.concatenate(
        [
            -np.ones(q_range_full, dtype=np.float64),
            np.ones(q_null_full, dtype=np.float64),
        ]
    )
    nulli_rank = permute_rows(
        drop_rows_dense(nulli_full[:, None], dropped_idx), pivot1, reverse=False
    ).ravel()
    q_total = int(rank)
    q_null = int(np.sum(nulli_rank > 0.0))
    q_range = int(q_total - q_null)
    current = _GDI1CurrentSpState(
        canonical=canonical,
        X=X_rank,
        beta=np.asarray(beta_rank, dtype=np.float64),
        W=W,
        XtWX=XtWX_rank,
        P=P_rank,
        A=A_rank,
        A_inv=A_inv_rank,
        penalized_system_rank=rank,
        dropped_column_indices=dropped_idx,
        pivot1=pivot1,
    )
    range_idx = np.flatnonzero(nulli_rank <= 0.0).astype(np.int64, copy=False)
    null_idx = np.flatnonzero(nulli_rank > 0.0).astype(np.int64, copy=False)
    rows_E = q_range
    return _GDIPKSetup(
        current=current,
        pk=_GDIPKState(
            q_total=q_total,
            q_range=q_range,
            q_null=q_null,
            rows_E=rows_E,
            rank=rank,
            range_idx=range_idx,
            null_idx=null_idx,
            kept_idx=kept_idx,
            dropped_idx=dropped_idx,
            n_drop=int(dropped_idx.size),
            pivot1=pivot1,
            PKtz=pk_tz,
            K=K,
            P=Pk,
            Rh=Rh,
            rS_work=rS_work,
            ldet_XWX_plus_S=ldet_xwxs,
        ),
    )


def _canonical_penalty_derivative_matrices(canonical, sp, n_sp):
    q = int(np.asarray(canonical.St, dtype=np.float64).shape[0])
    rows_e = q - int(canonical.Mp)
    mats = [np.zeros((q, q), dtype=np.float64) for _ in range(int(n_sp))]
    roots = list(canonical.rp.get("rS", []))
    for j in range(min(int(n_sp), len(roots))):
        root = np.asarray(roots[j], dtype=np.float64)
        if root.size == 0:
            continue
        root_full = np.zeros((q, int(root.shape[1])), dtype=np.float64)
        root_full[:rows_e, :] = root
        mats[j] = float(sp[j]) * (root_full @ root_full.T)
    return mats


def _canonical_logdet_term_derivatives(model, sp, A, A_inv, dA, d2A_mat):
    A = np.asarray(A, dtype=np.float64)
    try:
        cA, _ = cho_factor(A, check_finite=False)
    except np.linalg.LinAlgError:
        n_sp = len(dA)
        return (
            np.nan,
            np.full(n_sp, np.nan, dtype=np.float64),
            np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        )

    logdet_A = 2.0 * float(np.sum(np.log(np.abs(np.diag(cA)))))
    logdet_S, detS1, detS2 = _stable_penalty_logdet_derivatives(model, sp, order=2)
    if not np.isfinite(logdet_S):
        n_sp = len(dA)
        return (
            np.inf,
            np.full(n_sp, np.nan, dtype=np.float64),
            np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        )

    detA1, detA2 = _logdet_penalized_system_derivatives(
        A_inv=np.asarray(A_inv, dtype=np.float64),
        dA=dA,
        d2A_mat=d2A_mat,
    )
    return (
        0.5 * (logdet_A - logdet_S),
        0.5 * (detA1 - detS1),
        0.5 * (detA2 - detS2),
    )


def _gdi1_ift1_state(model, y, sol, sp, current, pk_state):
    """
    Port-shaped `ift1` stage on current canonical state.

    Uses `mgcv::ift1()` variable roles: `dbeta`, `deta`, `dA`, `d2beta`, `d2eta`.
    """
    X = np.asarray(current.X, dtype=np.float64)
    beta = np.asarray(current.beta, dtype=np.float64)
    eta = np.asarray(sol["eta"], dtype=np.float64)
    W = np.asarray(current.W, dtype=np.float64)
    A_inv = np.asarray(current.A_inv, dtype=np.float64)
    P_derivs = [
        _drop_permute_symmetric(Pj, current.dropped_column_indices, current.pivot1)
        for Pj in _canonical_penalty_derivative_matrices(
            current.canonical, sp, int(_n_smoothing_params(model) or 0)
        )
    ]
    if str(getattr(model.family, "name", "")).lower() == "negbin":
        dd = _negbin_ddeta_logtheta(
            model.family,
            y,
            np.asarray(sol["mu"], dtype=np.float64),
            _prior_weights(model, y),
            deriv=2,
        )
        n_sp = len(P_derivs)
        dbeta = [None] * n_sp
        deta = [None] * n_sp
        dA = [None] * n_sp
        dXtWX = [None] * n_sp
        d2beta_mat = [[None] * n_sp for _ in range(n_sp)]
        d2A_mat = [[None] * n_sp for _ in range(n_sp)]
        d2XtWX_mat = [[None] * n_sp for _ in range(n_sp)]

        for j, Pj in enumerate(P_derivs):
            dbeta_j = -(A_inv @ (Pj @ beta)) if np.any(Pj) else np.zeros_like(beta)
            deta_j = X @ dbeta_j
            dW_j = 0.5 * np.asarray(dd["Deta3"], dtype=np.float64) * deta_j
            dXtWX_j = X.T @ (dW_j[:, None] * X)
            dbeta[j] = dbeta_j
            deta[j] = deta_j
            dXtWX[j] = dXtWX_j
            dA[j] = dXtWX_j + Pj

        for j, Pj in enumerate(P_derivs):
            for k in range(j, n_sp):
                delta_jk = 1.0 if j == k else 0.0
                d2beta_jk = -(
                    A_inv @ (dA[k] @ dbeta[j] + Pj @ dbeta[k] + delta_jk * (Pj @ beta))
                )
                d2eta_jk = X @ d2beta_jk
                d2W_jk = 0.5 * (
                    np.asarray(dd["Deta4"], dtype=np.float64) * deta[j] * deta[k]
                    + np.asarray(dd["Deta3"], dtype=np.float64) * d2eta_jk
                )
                d2XtWX_jk = X.T @ (d2W_jk[:, None] * X)
                d2A_jk = d2XtWX_jk + (Pj if j == k else 0.0)
                d2beta_mat[j][k] = d2beta_jk
                d2beta_mat[k][j] = d2beta_jk
                d2XtWX_mat[j][k] = d2XtWX_jk
                d2XtWX_mat[k][j] = d2XtWX_jk
                d2A_mat[j][k] = d2A_jk
                d2A_mat[k][j] = d2A_jk

        return _GDI1IFTState(
            P_derivs=P_derivs,
            dbeta=dbeta,
            deta=deta,
            dA=dA,
            dXtWX=dXtWX,
            d2beta_mat=d2beta_mat,
            d2A_mat=d2A_mat,
            d2XtWX_mat=d2XtWX_mat,
            dW_obs=[
                0.5
                * np.asarray(dd["Deta3"], dtype=np.float64)
                * np.asarray(v, dtype=np.float64)
                for v in deta
            ],
            d2W_obs_mat=[
                [
                    0.5
                    * (
                        np.asarray(dd["Deta4"], dtype=np.float64)
                        * np.asarray(deta[j], dtype=np.float64)
                        * np.asarray(deta[k], dtype=np.float64)
                        + np.asarray(dd["Deta3"], dtype=np.float64)
                        * (X @ np.asarray(d2beta_mat[j][k], dtype=np.float64))
                    )
                    for k in range(n_sp)
                ]
                for j in range(n_sp)
            ],
        )

    dW_eta, d2W_eta = _working_weight_derivatives_wrt_linpred(
        model, y, eta, sol["mu"], W
    )

    n_sp = len(P_derivs)
    dbeta = [None] * n_sp
    deta = [None] * n_sp
    dA = [None] * n_sp
    dXtWX = [None] * n_sp
    d2beta_mat = [[None] * n_sp for _ in range(n_sp)]
    d2A_mat = [[None] * n_sp for _ in range(n_sp)]
    d2XtWX_mat = [[None] * n_sp for _ in range(n_sp)]

    for j, Pj in enumerate(P_derivs):
        dbeta_j = -(A_inv @ (Pj @ beta)) if np.any(Pj) else np.zeros_like(beta)
        deta_j = X @ dbeta_j
        dW_j = dW_eta * deta_j
        dXtWX_j = X.T @ (dW_j[:, None] * X)
        dbeta[j] = dbeta_j
        deta[j] = deta_j
        dXtWX[j] = dXtWX_j
        dA[j] = dXtWX_j + Pj

    for j, Pj in enumerate(P_derivs):
        for k in range(j, n_sp):
            delta_jk = 1.0 if j == k else 0.0
            d2beta_jk = -(
                A_inv @ (dA[k] @ dbeta[j] + Pj @ dbeta[k] + delta_jk * (Pj @ beta))
            )
            d2eta_jk = X @ d2beta_jk
            d2W_jk = d2W_eta * deta[j] * deta[k] + dW_eta * d2eta_jk
            d2XtWX_jk = X.T @ (d2W_jk[:, None] * X)
            d2A_jk = d2XtWX_jk + (Pj if j == k else 0.0)
            d2beta_mat[j][k] = d2beta_jk
            d2beta_mat[k][j] = d2beta_jk
            d2XtWX_mat[j][k] = d2XtWX_jk
            d2XtWX_mat[k][j] = d2XtWX_jk
            d2A_mat[j][k] = d2A_jk
            d2A_mat[k][j] = d2A_jk

    return _GDI1IFTState(
        P_derivs=P_derivs,
        dbeta=dbeta,
        deta=deta,
        dA=dA,
        dXtWX=dXtWX,
        d2beta_mat=d2beta_mat,
        d2A_mat=d2A_mat,
        d2XtWX_mat=d2XtWX_mat,
        dW_obs=[dW_eta * np.asarray(v, dtype=np.float64) for v in deta],
        d2W_obs_mat=[
            [
                d2W_eta
                * np.asarray(deta[j], dtype=np.float64)
                * np.asarray(deta[k], dtype=np.float64)
                + dW_eta * (X @ np.asarray(d2beta_mat[j][k], dtype=np.float64))
                for k in range(n_sp)
            ]
            for j in range(n_sp)
        ],
    )


def _gdi1_deviance_terms(model, y, sol, current, ift):
    if str(getattr(model.family, "name", "")).lower() == "negbin":
        dd = _negbin_ddeta_logtheta(
            model.family,
            y,
            np.asarray(sol["mu"], dtype=np.float64),
            _prior_weights(model, y),
            deriv=0,
        )
        n_sp = len(ift.dbeta)
        D1 = np.zeros(n_sp, dtype=np.float64)
        D2 = np.zeros((n_sp, n_sp), dtype=np.float64)
        X = np.asarray(current.X, dtype=np.float64)
        for j in range(n_sp):
            deta_j = np.asarray(ift.deta[j], dtype=np.float64)
            D1[j] = float(np.sum(np.asarray(dd["Deta"], dtype=np.float64) * deta_j))
        for j in range(n_sp):
            for k in range(j, n_sp):
                val = float(
                    np.sum(
                        np.asarray(dd["Deta2"], dtype=np.float64)
                        * np.asarray(ift.deta[j], dtype=np.float64)
                        * np.asarray(ift.deta[k], dtype=np.float64)
                        + np.asarray(dd["Deta"], dtype=np.float64)
                        * (X @ np.asarray(ift.d2beta_mat[j][k], dtype=np.float64))
                    )
                )
                D2[j, k] = D2[k, j] = val
        return D1, D2

    dev_grad, dev_hess = _deviance_coefficient_derivatives(
        model,
        y,
        np.asarray(sol["eta"], dtype=np.float64),
        sol["mu"],
        np.asarray(current.W, dtype=np.float64),
        np.asarray(current.X, dtype=np.float64),
    )
    return _deviance_chained_to_smoothing(dev_grad, dev_hess, ift.dbeta, ift.d2beta_mat)


def _gdi1_bsb_terms(current, ift):
    return _penalty_quadratic_and_sp_derivatives(
        beta=np.asarray(current.beta, dtype=np.float64),
        P_total=np.asarray(current.P, dtype=np.float64),
        P_derivs=ift.P_derivs,
        dbeta_cols=ift.dbeta,
        d2beta_mat=ift.d2beta_mat,
    )


def _schur_logdet_terms(A_rr, A_rf, A_ff, dArr, dArf, dAff, d2Arr, d2Arf, d2Aff):
    p = int(np.asarray(A_ff, dtype=np.float64).shape[0])
    if p == 0:
        return 0.0, np.empty((0,), dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    cR, loR = cho_factor(A_rr, check_finite=False)
    Rinv = cho_solve((cR, loR), np.eye(A_rr.shape[0]), check_finite=False)
    C = A_ff - A_rf.T @ Rinv @ A_rf
    cC, loC = cho_factor(C, check_finite=False)
    Cinv = cho_solve((cC, loC), np.eye(p), check_finite=False)
    n_sp = len(dArr)
    dC = [None] * n_sp
    d1 = np.zeros(n_sp, dtype=np.float64)
    d2 = np.zeros((n_sp, n_sp), dtype=np.float64)
    for j in range(n_sp):
        dRinv_j = -Rinv @ dArr[j] @ Rinv
        dC_j = (
            dAff[j]
            - dArf[j].T @ Rinv @ A_rf
            - A_rf.T @ Rinv @ dArf[j]
            - A_rf.T @ dRinv_j @ A_rf
        )
        dC[j] = dC_j
        d1[j] = float(np.trace(Cinv @ dC_j))
    for j in range(n_sp):
        for k in range(j, n_sp):
            dRinv_k = -Rinv @ dArr[k] @ Rinv
            d2Rinv_jk = (
                Rinv @ dArr[k] @ Rinv @ dArr[j] @ Rinv
                - Rinv @ d2Arr[j][k] @ Rinv
                + Rinv @ dArr[j] @ Rinv @ dArr[k] @ Rinv
            )
            d2C_jk = (
                d2Aff[j][k]
                - d2Arf[j][k].T @ Rinv @ A_rf
                - dArf[j].T @ dRinv_k @ A_rf
                - dArf[j].T @ Rinv @ dArf[k]
                - dArf[k].T @ Rinv @ dArf[j]
                - A_rf.T @ dRinv_k @ dArf[j]
                - A_rf.T @ Rinv @ d2Arf[j][k]
                - dArf[k].T @ dRinv_j @ A_rf
                - A_rf.T @ d2Rinv_jk @ A_rf
                - A_rf.T @ dRinv_j @ dArf[k]
            )
            d2[j, k] = d2[k, j] = float(
                np.trace(Cinv @ d2C_jk - Cinv @ dC[k] @ Cinv @ dC[j])
            )
    logdet_C = 2.0 * float(np.sum(np.log(np.abs(np.diag(cC)))))
    return 0.5 * logdet_C, 0.5 * d1, 0.5 * d2


def _gdi1_reml_penalty_terms(model, sp, current, ift, pk_state, *, method):
    """
    Structured determinant penalty stage corresponding to `get_ddetXWXpS()`
    plus REML Schur-complement augmentation on current-sp canonical state.
    """
    q_range = int(pk_state.q_range)
    q_total = int(pk_state.q_total)
    A = np.asarray(current.A, dtype=np.float64)
    if q_range == 0:
        n_sp = len(ift.dA)
        zero = np.zeros(n_sp, dtype=np.float64)
        return 0.0, zero, np.zeros((n_sp, n_sp), dtype=np.float64)

    Arr = A[:q_range, :q_range]
    dArr = [np.asarray(dAj[:q_range, :q_range], dtype=np.float64) for dAj in ift.dA]
    d2Arr = [
        [np.asarray(d2A[:q_range, :q_range], dtype=np.float64) for d2A in row]
        for row in ift.d2A_mat
    ]
    cR, loR = cho_factor(Arr, check_finite=False)
    logdet_R = 2.0 * float(np.sum(np.log(np.abs(np.diag(cR)))))
    logdet_S, detS1, detS2 = _stable_penalty_logdet_derivatives(model, sp, order=2)
    if not np.isfinite(logdet_S):
        n_sp = len(ift.dA)
        return (
            np.inf,
            np.full(n_sp, np.nan, dtype=np.float64),
            np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        )
    Rinv = cho_solve((cR, loR), np.eye(q_range), check_finite=False)
    detR1, detR2 = _logdet_penalized_system_derivatives(Rinv, dArr, d2Arr)
    K = 0.5 * (logdet_R - logdet_S)
    K1 = 0.5 * (detR1 - detS1)
    K2 = 0.5 * (detR2 - detS2)

    if str(method).upper() != "REML" or q_total == q_range:
        return K, K1, K2

    A_rf = np.asarray(A[:q_range, q_range:], dtype=np.float64)
    A_ff = np.asarray(A[q_range:, q_range:], dtype=np.float64)
    dArf = [np.asarray(dAj[:q_range, q_range:], dtype=np.float64) for dAj in ift.dA]
    dAff = [np.asarray(dAj[q_range:, q_range:], dtype=np.float64) for dAj in ift.dA]
    d2Arf = [
        [np.asarray(d2A[:q_range, q_range:], dtype=np.float64) for d2A in row]
        for row in ift.d2A_mat
    ]
    d2Aff = [
        [np.asarray(d2A[q_range:, q_range:], dtype=np.float64) for d2A in row]
        for row in ift.d2A_mat
    ]
    C, C1, C2 = _schur_logdet_terms(
        Arr, A_rf, A_ff, dArr, dArf, dAff, d2Arr, d2Arf, d2Aff
    )
    return K + C, K1 + C1, K2 + C2


def _get_ddetXWXpS_qr_terms(model, sp, current, ift, pk_state):
    """
    Port-shaped Python analogue of `mgcv::get_ddetXWXpS()` on QR `gdiPK` state.
    """
    if (
        pk_state.P is None
        or pk_state.K is None
        or pk_state.rS_work is None
        or ift.dW_obs is None
        or ift.d2W_obs_mat is None
    ):
        return None

    P = np.asarray(pk_state.P, dtype=np.float64)
    K = np.asarray(pk_state.K, dtype=np.float64)
    rS_work = np.asarray(pk_state.rS_work, dtype=np.float64)
    rank = int(pk_state.rank)
    n_sp = len(ift.dW_obs)
    if rank == 0:
        zero = np.zeros(n_sp, dtype=np.float64)
        return 0.0, zero, np.zeros((n_sp, n_sp), dtype=np.float64)

    wabs = np.clip(np.abs(np.asarray(current.W, dtype=np.float64)), 1e-300, None)
    Tk = [np.asarray(dw, dtype=np.float64) / wabs for dw in ift.dW_obs]
    Tkm = [
        [np.asarray(d2w, dtype=np.float64) / wabs for d2w in row]
        for row in ift.d2W_obs_mat
    ]
    diagKKt = np.sum(K * K, axis=1)

    KtTK = [K.T @ (tk[:, None] * K) for tk in Tk]
    det1_x = np.array([float(np.dot(tk, diagKKt)) for tk in Tk], dtype=np.float64)

    roots = list(current.canonical.rp.get("rS", []))
    col_off = 0
    PtSP = []
    trPtSP = np.zeros(n_sp, dtype=np.float64)
    for m in range(n_sp):
        ncol = int(np.asarray(roots[m]).shape[1]) if m < len(roots) else 0
        root_block = rS_work[:, col_off : col_off + ncol]
        col_off += root_block.shape[1]
        PtrSm = (
            P.T @ root_block
            if root_block.size
            else np.empty((rank, 0), dtype=np.float64)
        )
        PtSP_m = (
            PtrSm @ PtrSm.T if PtrSm.size else np.zeros((rank, rank), dtype=np.float64)
        )
        PtSP.append(PtSP_m)
        trPtSP[m] = float(sp[m] * np.sum(PtrSm * PtrSm))
        det1_x[m] += trPtSP[m]

    det2_x = np.zeros((n_sp, n_sp), dtype=np.float64)
    for m in range(n_sp):
        for k in range(m, n_sp):
            val = float(np.dot(Tkm[m][k], diagKKt))
            val -= float(np.sum(KtTK[k] * KtTK[m].T))
            if k == m:
                val += trPtSP[m]
            val -= float(sp[m] * np.sum(KtTK[k] * PtSP[m].T))
            val -= float(sp[k] * np.sum(KtTK[m] * PtSP[k].T))
            val -= float(sp[m] * sp[k] * np.sum(PtSP[k] * PtSP[m].T))
            det2_x[k, m] = det2_x[m, k] = val

    logdet_S, detS1, detS2 = _stable_penalty_logdet_derivatives(model, sp, order=2)
    if not np.isfinite(logdet_S):
        return (
            np.inf,
            np.full(n_sp, np.nan, dtype=np.float64),
            np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        )
    K0 = 0.5 * (float(pk_state.ldet_XWX_plus_S) - logdet_S)
    K1 = 0.5 * (det1_x - detS1)
    K2 = 0.5 * (det2_x - detS2)
    return K0, K1, K2


def _ml_penalty1_terms(model, sp, current, ift, pk_state):
    """
    `mgcv::MLpenalty1()`-shaped penalty stage on canonical current-sp state.

    On the current Python side, the canonical range-space block already plays the
    role of the rank-truncated `R` factor used by upstream MLpenalty1.
    """
    return _gdi1_reml_penalty_terms(model, sp, current, ift, pk_state, method="ML")


def _get_ddetXWXpS_terms(model, sp, current, ift, pk_state):
    """
    `mgcv::get_ddetXWXpS()`-shaped REML determinant penalty stage.
    """
    qr_terms = _get_ddetXWXpS_qr_terms(model, sp, current, ift, pk_state)
    if qr_terms is not None:
        return qr_terms
    return _gdi1_reml_penalty_terms(model, sp, current, ift, pk_state, method="REML")


def _gdi1_det_terms(model, sp, current, ift, pk_state, *, method):
    det1, det2 = _logdet_penalized_system_derivatives(
        A_inv=np.asarray(current.A_inv, dtype=np.float64),
        dA=ift.dA,
        d2A_mat=ift.d2A_mat,
    )
    trA, trA1, trA2 = _hat_matrix_trace_and_sp_derivatives(
        A_inv=np.asarray(current.A_inv, dtype=np.float64),
        XtWX=np.asarray(current.XtWX, dtype=np.float64),
        dA=ift.dA,
        d2A_mat=ift.d2A_mat,
        dXtWX=ift.dXtWX,
        d2XtWX_mat=ift.d2XtWX_mat,
    )
    if str(method).upper() == "ML":
        K, K1, K2 = _ml_penalty1_terms(model, sp, current, ift, pk_state)
    else:
        K, K1, K2 = _get_ddetXWXpS_terms(model, sp, current, ift, pk_state)
    dVkk = _quadratic_form_in_beta_directions(
        np.asarray(current.A, dtype=np.float64), ift.dbeta
    )
    return det1, det2, trA, trA1, trA2, K, K1, K2, dVkk


def _gdi1_kernel(model, y, sol, sp, *, method):
    """Structured Python analogue of `mgcv::gdi1()` on canonical current-sp state."""
    setup = _gdi_pk_setup(model, sol, sp, deriv=2)
    current = setup.current
    pk_state = setup.pk
    ift = _gdi1_ift1_state(model, y, sol, sp, current, pk_state)
    D1, D2 = _gdi1_deviance_terms(model, y, sol, current, ift)
    bSb, bSb1, bSb2 = _gdi1_bsb_terms(current, ift)
    det1, det2, trA, trA1, trA2, K, K1, K2, dVkk = _gdi1_det_terms(
        model, sp, current, ift, pk_state, method=method
    )
    return _GDI1Kernel(
        current=current,
        ift=ift,
        D1=np.asarray(D1, dtype=np.float64),
        D2=np.asarray(D2, dtype=np.float64),
        bSb=float(bSb),
        bSb1=np.asarray(bSb1, dtype=np.float64),
        bSb2=np.asarray(bSb2, dtype=np.float64),
        det1=np.asarray(det1, dtype=np.float64),
        det2=np.asarray(det2, dtype=np.float64),
        trA=float(trA),
        trA1=np.asarray(trA1, dtype=np.float64),
        trA2=np.asarray(trA2, dtype=np.float64),
        K=float(K),
        K1=np.asarray(K1, dtype=np.float64),
        K2=np.asarray(K2, dtype=np.float64),
        dVkk=np.asarray(dVkk, dtype=np.float64),
    )


def _gdi2_penalty_terms(model, sp, current, dA, d2A_mat, pk_state, *, n_theta, method):
    ntot = len(dA)
    q_range = int(pk_state.q_range)
    q_total = int(pk_state.q_total)
    if q_range == 0:
        zero = np.zeros(ntot, dtype=np.float64)
        return 0.0, zero, np.zeros((ntot, ntot), dtype=np.float64)

    A = np.asarray(current.A, dtype=np.float64)
    Arr = A[:q_range, :q_range]
    dArr = [np.asarray(dAj[:q_range, :q_range], dtype=np.float64) for dAj in dA]
    d2Arr = [
        [np.asarray(d2A[:q_range, :q_range], dtype=np.float64) for d2A in row]
        for row in d2A_mat
    ]
    cR, loR = cho_factor(Arr, check_finite=False)
    logdet_R = 2.0 * float(np.sum(np.log(np.abs(np.diag(cR)))))
    Rinv = cho_solve((cR, loR), np.eye(q_range), check_finite=False)
    detR1, detR2 = _logdet_penalized_system_derivatives(Rinv, dArr, d2Arr)

    logdet_S, detS1_sp, detS2_sp = _stable_penalty_logdet_derivatives(
        model, sp, order=2
    )
    if not np.isfinite(logdet_S):
        return (
            np.inf,
            np.full(ntot, np.nan, dtype=np.float64),
            np.full((ntot, ntot), np.nan, dtype=np.float64),
        )
    detS1 = np.zeros(ntot, dtype=np.float64)
    detS2 = np.zeros((ntot, ntot), dtype=np.float64)
    detS1[int(n_theta) :] = np.asarray(detS1_sp, dtype=np.float64)
    detS2[int(n_theta) :, int(n_theta) :] = np.asarray(detS2_sp, dtype=np.float64)
    K = 0.5 * (logdet_R - logdet_S)
    K1 = 0.5 * (detR1 - detS1)
    K2 = 0.5 * (detR2 - detS2)

    if str(method).upper() != "REML" or q_total == q_range:
        return K, K1, K2

    A_rf = np.asarray(A[:q_range, q_range:], dtype=np.float64)
    A_ff = np.asarray(A[q_range:, q_range:], dtype=np.float64)
    dArf = [np.asarray(dAj[:q_range, q_range:], dtype=np.float64) for dAj in dA]
    dAff = [np.asarray(dAj[q_range:, q_range:], dtype=np.float64) for dAj in dA]
    d2Arf = [
        [np.asarray(d2A[:q_range, q_range:], dtype=np.float64) for d2A in row]
        for row in d2A_mat
    ]
    d2Aff = [
        [np.asarray(d2A[q_range:, q_range:], dtype=np.float64) for d2A in row]
        for row in d2A_mat
    ]
    C, C1, C2 = _schur_logdet_terms(
        Arr, A_rf, A_ff, dArr, dArf, dAff, d2Arr, d2Arf, d2Aff
    )
    return K + C, K1 + C1, K2 + C2


def _gdi2_ift2_state_negbin(model, y, sol, sp, current, pk_state):
    """
    Port of `mgcv::ift2()` for single-theta negative binomial on canonical state.
    Parameter order is `[log(theta), log(sp_1), ..., log(sp_m)]`.
    """
    X = np.asarray(current.X, dtype=np.float64)
    beta = np.asarray(current.beta, dtype=np.float64)
    mu = np.asarray(sol["mu"], dtype=np.float64)
    A_inv = np.asarray(current.A_inv, dtype=np.float64)
    weights = _prior_weights(model, y)
    dd = _negbin_ddeta_logtheta(model.family, y, mu, weights, deriv=2)
    P_sp = [
        _drop_permute_symmetric(Pj, current.dropped_column_indices, current.pivot1)
        for Pj in _canonical_penalty_derivative_matrices(
            current.canonical, sp, int(_n_smoothing_params(model) or 0)
        )
    ]

    ntheta = 1
    ntot = ntheta + len(P_sp)
    zeroP = np.zeros_like(np.asarray(current.P, dtype=np.float64))
    P_derivs = [zeroP] + [np.asarray(Pj, dtype=np.float64) for Pj in P_sp]

    dbeta = [None] * ntot
    deta = [None] * ntot
    dA = [None] * ntot
    dXtWX = [None] * ntot
    d2beta_mat = [[None] * ntot for _ in range(ntot)]
    d2A_mat = [[None] * ntot for _ in range(ntot)]
    d2XtWX_mat = [[None] * ntot for _ in range(ntot)]

    for i in range(ntot):
        if i == 0:
            rhs = -0.5 * (X.T @ np.asarray(dd["Detath"], dtype=np.float64))
        else:
            rhs = -(P_derivs[i] @ beta)
        dbeta_i = A_inv @ rhs
        deta_i = X @ dbeta_i
        if i == 0:
            dW_i = 0.5 * (
                np.asarray(dd["Deta3"], dtype=np.float64) * deta_i
                + np.asarray(dd["Deta2th"], dtype=np.float64)
            )
        else:
            dW_i = 0.5 * np.asarray(dd["Deta3"], dtype=np.float64) * deta_i
        dXtWX_i = X.T @ (dW_i[:, None] * X)
        dbeta[i] = dbeta_i
        deta[i] = deta_i
        dXtWX[i] = dXtWX_i
        dA[i] = dXtWX_i + P_derivs[i]

    for i in range(ntot):
        for k in range(i, ntot):
            rhs = -0.5 * (
                X.T
                @ (
                    np.asarray(dd["Deta3"], dtype=np.float64)
                    * np.asarray(deta[i], dtype=np.float64)
                    * np.asarray(deta[k], dtype=np.float64)
                )
            )
            if k == 0:
                rhs -= 0.5 * (
                    X.T
                    @ (
                        np.asarray(dd["Deta2th"], dtype=np.float64)
                        * np.asarray(deta[i], dtype=np.float64)
                    )
                )
            else:
                rhs -= P_derivs[k] @ np.asarray(dbeta[i], dtype=np.float64)
            if i == 0:
                rhs -= 0.5 * (
                    X.T
                    @ (
                        np.asarray(dd["Deta2th"], dtype=np.float64)
                        * np.asarray(deta[k], dtype=np.float64)
                    )
                )
            else:
                rhs -= P_derivs[i] @ np.asarray(dbeta[k], dtype=np.float64)
            if i == 0 and k == 0:
                rhs -= 0.5 * (X.T @ np.asarray(dd["Detath2"], dtype=np.float64))
            elif i == k and i > 0:
                rhs -= P_derivs[i] @ beta
            d2beta_ik = A_inv @ rhs
            d2eta_ik = X @ d2beta_ik
            d2W_ik = 0.5 * (
                np.asarray(dd["Deta4"], dtype=np.float64)
                * np.asarray(deta[i], dtype=np.float64)
                * np.asarray(deta[k], dtype=np.float64)
                + np.asarray(dd["Deta3"], dtype=np.float64) * d2eta_ik
            )
            if i == 0:
                d2W_ik += 0.5 * (
                    np.asarray(dd["Deta3th"], dtype=np.float64)
                    * np.asarray(deta[k], dtype=np.float64)
                )
            if k == 0:
                d2W_ik += 0.5 * (
                    np.asarray(dd["Deta3th"], dtype=np.float64)
                    * np.asarray(deta[i], dtype=np.float64)
                )
            if i == 0 and k == 0:
                d2W_ik += 0.5 * np.asarray(dd["Deta2th2"], dtype=np.float64)
            d2XtWX_ik = X.T @ (d2W_ik[:, None] * X)
            d2A_ik = d2XtWX_ik + (P_derivs[i] if (i == k and i > 0) else 0.0)
            d2beta_mat[i][k] = d2beta_ik
            d2beta_mat[k][i] = d2beta_ik
            d2XtWX_mat[i][k] = d2XtWX_ik
            d2XtWX_mat[k][i] = d2XtWX_ik
            d2A_mat[i][k] = d2A_ik
            d2A_mat[k][i] = d2A_ik

    return _GDI2IFTState(
        P_derivs=P_derivs,
        dbeta=dbeta,
        deta=deta,
        dA=dA,
        dXtWX=dXtWX,
        d2beta_mat=d2beta_mat,
        d2A_mat=d2A_mat,
        d2XtWX_mat=d2XtWX_mat,
        ntheta=ntheta,
    )


def _gdi2_negbin_joint_kernel(model, y, sol, sp, *, method, need_hessian):
    """Port of `mgcv::gdi2()` for negative-binomial `log(theta)` branch."""
    setup = _gdi_pk_setup(model, sol, sp, deriv=2)
    current = setup.current
    pk_state = setup.pk
    ift = _gdi2_ift2_state_negbin(model, y, sol, sp, current, pk_state)
    weights = _prior_weights(model, y)
    dd = _negbin_ddeta_logtheta(
        model.family,
        y,
        np.asarray(sol["mu"], dtype=np.float64),
        weights,
        deriv=2,
    )
    ntot = len(ift.dbeta)

    D1 = np.zeros(ntot, dtype=np.float64)
    for i in range(ntot):
        D1[i] = float(
            np.sum(
                np.asarray(dd["Deta"], dtype=np.float64)
                * np.asarray(ift.deta[i], dtype=np.float64)
            )
        )
        if i == 0:
            D1[i] += float(np.sum(np.asarray(dd["Dth"], dtype=np.float64)))

    D2 = None
    if need_hessian:
        D2 = np.zeros((ntot, ntot), dtype=np.float64)
        for i in range(ntot):
            for k in range(i, ntot):
                val = float(
                    np.sum(
                        np.asarray(dd["Deta2"], dtype=np.float64)
                        * np.asarray(ift.deta[i], dtype=np.float64)
                        * np.asarray(ift.deta[k], dtype=np.float64)
                        + np.asarray(dd["Deta"], dtype=np.float64)
                        * (
                            current.X
                            @ np.asarray(ift.d2beta_mat[i][k], dtype=np.float64)
                        )
                    )
                )
                if i == 0:
                    val += float(
                        np.sum(
                            np.asarray(dd["Detath"], dtype=np.float64)
                            * np.asarray(ift.deta[k], dtype=np.float64)
                        )
                    )
                if k == 0:
                    val += float(
                        np.sum(
                            np.asarray(dd["Detath"], dtype=np.float64)
                            * np.asarray(ift.deta[i], dtype=np.float64)
                        )
                    )
                if i == 0 and k == 0:
                    val += float(np.sum(np.asarray(dd["Dth2"], dtype=np.float64)))
                D2[i, k] = D2[k, i] = val

    bSb, bSb1, bSb2 = _penalty_quadratic_and_sp_derivatives(
        beta=np.asarray(current.beta, dtype=np.float64),
        P_total=np.asarray(current.P, dtype=np.float64),
        P_derivs=ift.P_derivs,
        dbeta_cols=ift.dbeta,
        d2beta_mat=ift.d2beta_mat,
    )
    K, K1, K2 = _gdi2_penalty_terms(
        model,
        sp,
        current,
        ift.dA,
        ift.d2A_mat,
        pk_state,
        n_theta=ift.ntheta,
        method=method,
    )

    gdi1 = _gdi1_kernel(model, y, sol, sp, method=method)
    Dp = float(sol["deviance"]) + float(bSb)
    Dp1 = np.asarray(D1 + bSb1, dtype=np.float64)
    Dp2 = None if D2 is None else np.asarray(D2 + bSb2, dtype=np.float64)
    return _GDI2Kernel(
        gdi1=gdi1,
        phi=None,
        phi_curv=None,
        Dp=float(Dp),
        Dp1=Dp1,
        Dp2=Dp2,
        ift=ift,
        D1_full=np.asarray(D1, dtype=np.float64),
        D2_full=None if D2 is None else np.asarray(D2, dtype=np.float64),
        K1_full=np.asarray(K1, dtype=np.float64),
        K2_full=np.asarray(K2, dtype=np.float64),
        extra_name="log_theta",
        extra_value=float(model.family.getTheta(False)),
    )


def _gdi2_gamma_joint_kernel(model, y, sol, sp, *, method, need_hessian):
    """
    Current Gamma branch is Python analogue of `gdi2` extended-family staging:
    smoothing kernel + profiled extra parameter (`log(phi)` here).
    """
    gdi1 = _gdi1_kernel(model, y, sol, sp, method=method)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    Dp = float(sol["deviance"]) + float(gdi1.bSb)
    phi = _solve_gamma_profile_scale(
        model,
        y,
        Dp,
        mp,
        method=method,
        init_scale=float(sol["scale"]),
    )
    if not np.isfinite(phi) or phi <= 0.0:
        raise RuntimeError(
            "Gamma exact PIRLS derivatives require positive profile scale."
        )
    Dp1 = np.asarray(gdi1.D1 + gdi1.bSb1, dtype=np.float64)
    if not need_hessian:
        return _GDI2Kernel(
            gdi1=gdi1,
            phi=float(phi),
            phi_curv=None,
            Dp=float(Dp),
            Dp1=Dp1,
            Dp2=None,
            extra_name="log_phi",
            extra_value=float(np.log(phi)),
        )
    _, _, phi_curv = _gamma_profile_objective_curvature(
        model, y, Dp, phi, mp, method=method
    )
    if not np.isfinite(phi_curv) or abs(phi_curv) <= 1e-14:
        raise RuntimeError(
            "Gamma exact PIRLS derivatives require finite profile curvature."
        )
    Dp2 = np.asarray(gdi1.D2 + gdi1.bSb2, dtype=np.float64)
    return _GDI2Kernel(
        gdi1=gdi1,
        phi=float(phi),
        phi_curv=float(phi_curv),
        Dp=float(Dp),
        Dp1=Dp1,
        Dp2=Dp2,
        extra_name="log_phi",
        extra_value=float(np.log(phi)),
    )


def _gdi2_joint_kernel(model, y, sol, sp, *, method, need_hessian):
    """
    Generic extended-family entry mirroring `gdi2()` dispatch.

    Current full port exists for Gamma profile-scale branch. Other extra-parameter
    families still need dedicated ports of `ift2` / theta-coupled derivative terms.
    """
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name == "gamma":
        return _gdi2_gamma_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    if family_name == "negbin":
        return _gdi2_negbin_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    raise NotImplementedError(
        "Generic `gdi2` current-sp extended-family port is not complete yet "
        f"for family={model.family.name!r}."
    )


def criterion_gradient_ml_reml_pirls_exact(model, y, log_sp, method):
    if not can_use_simple_ml_reml_structure(model):
        raise NotImplementedError(
            "Exact PIRLS ML/REML gradients are currently available only for "
            "terms whose penalties do not couple disconnected support "
            "components through null-space penalties."
        )

    if not bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False)):
        raise NotImplementedError(
            f"Family {model.family.name!r} does not yet provide exact PIRLS first derivatives."
        )
    family_name = str(getattr(model.family, "name", "")).lower()
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("score_gamma must be finite and positive.")
    if getattr(model.family, "known_scale", None) is None and family_name != "gamma":
        raise NotImplementedError(
            "Exact PIRLS ML/REML gradients are currently implemented only for "
            "fixed-scale families, plus Gamma via the profiled scale branch."
        )
    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    if int(np.sum(free_mask)) == 0:
        return np.empty((0,), dtype=np.float64)

    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    kernel = _gdi1_kernel(model, y, sol, sp, method=method)
    scale = float(sol["scale"])

    if getattr(model.family, "known_scale", None) is None and family_name == "gamma":
        gdi2 = _gdi2_joint_kernel(model, y, sol, sp, method=method, need_hessian=False)
        model._pirls_reml_gamma_state_ = {
            "K": float(kernel.K),
            "K1": np.asarray(kernel.K1, dtype=np.float64),
            "phi": float(gdi2.phi),
            "scale_est": float(sol["scale"]),
            "Dp": float(gdi2.Dp),
            "Dp1": np.asarray(gdi2.Dp1, dtype=np.float64),
        }
        return np.asarray(gdi2.Dp1[free_mask], dtype=np.float64) / (
            2.0 * gdi2.phi * gamma
        ) + np.asarray(kernel.K1[free_mask], dtype=np.float64)

    grad_full = np.asarray(kernel.D1 + kernel.bSb1, dtype=np.float64) / (
        2.0 * scale * gamma
    ) + np.asarray(kernel.K1, dtype=np.float64)
    return np.asarray(grad_full[free_mask], dtype=np.float64)


def criterion_hessian_ml_reml_pirls_exact(model, y, log_sp, method):
    if not can_use_simple_ml_reml_structure(model):
        raise NotImplementedError(
            "Exact PIRLS ML/REML Hessians are currently available only for "
            "terms whose penalties do not couple disconnected support "
            "components through null-space penalties."
        )

    if not bool(
        getattr(model.family, "supports_exact_pirls_second_derivatives", False)
    ):
        raise NotImplementedError(
            f"Family {model.family.name!r} does not yet provide exact PIRLS second derivatives."
        )
    family_name = str(getattr(model.family, "name", "")).lower()
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("score_gamma must be finite and positive.")
    if getattr(model.family, "known_scale", None) is None and family_name != "gamma":
        raise NotImplementedError(
            "Exact PIRLS ML/REML Hessians are currently implemented only for fixed-scale families, "
            "plus Gamma via the profiled scale branch."
        )

    free_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~free_mask
    free_idx = np.flatnonzero(free_mask)
    n_free = int(free_idx.size)
    if n_free == 0:
        return np.empty((0, 0), dtype=np.float64)

    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    kernel = _gdi1_kernel(model, y, sol, sp, method=method)
    scale = float(sol["scale"])

    detXWXS1 = detXWXS2 = None
    P1 = P2 = phi1 = phi2 = None
    full_grad = full_hess = None

    try:
        if getattr(model.family, "known_scale", None) is None:
            gdi2 = _gdi2_joint_kernel(
                model, y, sol, sp, method=method, need_hessian=True
            )
            joint_grad = kernel.K1 + gdi2.Dp1 / (2.0 * gdi2.phi * gamma)
            cross = -gdi2.Dp1 / (2.0 * gdi2.phi * gamma)
            full_grad = joint_grad
            full_hess = (
                kernel.K2
                + gdi2.Dp2 / (2.0 * gdi2.phi * gamma)
                - np.outer(cross, cross) / gdi2.phi_curv
            )
            model._pirls_reml_gamma_state_ = {
                "K": float(kernel.K),
                "K1": np.asarray(kernel.K1, dtype=np.float64),
                "K2": np.asarray(kernel.K2, dtype=np.float64),
                "phi": float(gdi2.phi),
                "scale_est": float(sol["scale"]),
                "phi_curv": float(gdi2.phi_curv),
                "Dp": float(gdi2.Dp),
                "Dp1": np.asarray(gdi2.Dp1, dtype=np.float64),
                "Dp2": np.asarray(gdi2.Dp2, dtype=np.float64),
            }
        else:
            full_grad = (
                np.asarray(kernel.D1, dtype=np.float64)
                + np.asarray(kernel.bSb1, dtype=np.float64)
            ) / (2.0 * scale * gamma) + kernel.K1
            full_hess = (
                np.asarray(kernel.D2, dtype=np.float64)
                + np.asarray(kernel.bSb2, dtype=np.float64)
            ) / (2.0 * scale * gamma) + kernel.K2
        model._pirls_reml_derivative_kernel_state_ = {
            "bSb": kernel.bSb,
            "bSb1": kernel.bSb1,
            "bSb2": kernel.bSb2,
            "dVkk": kernel.dVkk,
            "det1": kernel.det1,
            "det2": kernel.det2,
            "trA": kernel.trA,
            "trA1": kernel.trA1,
            "trA2": kernel.trA2,
            "D1": kernel.D1,
            "D2": kernel.D2,
            "P1": P1,
            "P2": P2,
            "phi1": phi1,
            "phi2": phi2,
            "full_grad": full_grad,
            "full_hess": full_hess,
            "penalty_grad_raw": None,
            "penalty_hess_raw": None,
            "detXWXS1": detXWXS1,
            "detXWXS2": detXWXS2,
        }
    except Exception:
        model._pirls_reml_derivative_kernel_state_ = None

    if full_hess is None:
        raise RuntimeError("Exact PIRLS Hessian assembly did not produce a result.")
    return np.asarray(full_hess[np.ix_(free_idx, free_idx)], dtype=np.float64)
