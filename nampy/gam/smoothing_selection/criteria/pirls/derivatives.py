"""Exact first/second derivatives of PIRLS Laplace ML/REML criteria."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import digamma, polygamma

from ...._model_state import (
    _coef_column_offset,
    _n_smoothing_params,
    _penalty_blocks_seq,
)
from ....fit.backends import (
    solve_gaussian_given_smoothing,
    solve_pirls_given_smoothing,
)
from ....fit.capabilities import (
    can_use_simple_ml_reml_structure,
    uses_closed_form_solver,
)
from ....fit.linalg.matrix_reindexing import (
    drop_columns_dense,
    drop_rows_dense,
    permute_columns,
    permute_rows,
    restore_dropped_rows,
)
from ....fit.linalg.stacked_qr import (
    build_penalized_qr_state_nonnegative,
)
from ....fit.smoothing_params import expand_smoothing_params_from_log
from ...reparam import (
    _full_design_matrix,
    _stable_penalty_logdet_derivatives,
    _static_penalty_null_dim,
    build_penalty_reparameterization_state,
)
from .reml_blocks import (
    _deviance_chained_to_smoothing,
    _hat_matrix_trace_and_sp_derivatives,
    _logdet_penalized_system_derivatives,
    _penalty_quadratic_and_sp_derivatives,
    _quadratic_form_in_beta_directions,
    _working_weight_derivatives_wrt_linpred,
)
from .value import _gamma_profile_objective_curvature, _solve_gamma_profile_scale

_MGCV_GAM_FIT3_RANK_TOL = float(np.finfo(np.float64).eps * 100.0)
_MGCV_GAM_FIT4_RANK_TOL = float(np.finfo(np.float64).eps ** 0.75)


def _free_smoothing_mask(model) -> np.ndarray:
    fixed_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    return ~fixed_mask


def _fit3_gcv_ubre_kernel(model, y, log_sp):
    """
    Build the `gam.fit3` derivative state used by GCV/UBRE/AIC.

    Upstream reference: `mgcv/R/gam.fit3.r::gam.fit3`, GCV/UBRE branch using
    `oo$D1`, `oo$D2`, `oo$trA1`, and `oo$trA2`.
    """
    free_mask = _free_smoothing_mask(model)
    if int(np.sum(free_mask)) == 0:
        return None, None, free_mask

    sp = expand_smoothing_params_from_log(model, log_sp)
    if uses_closed_form_solver(model):
        sol = solve_gaussian_given_smoothing(model, y, sp)
    else:
        if not can_use_simple_ml_reml_structure(model):
            raise NotImplementedError(
                "Exact gam.fit3 GCV/UBRE/AIC derivatives are currently available "
                "only for penalty structures supported by the upstream-mirrored "
                "gdi1 derivative path."
            )
        if not bool(
            getattr(model.family, "supports_exact_pirls_second_derivatives", False)
        ):
            raise NotImplementedError(
                f"Family {model.family.name!r} does not yet provide exact PIRLS "
                "second derivatives required by gam.fit3 GCV/UBRE/AIC scoring."
            )
        sol = solve_pirls_given_smoothing(model, y, sp)

    kernel = _gdi1_kernel(model, y, sol, sp, method="GCV")
    return sol, kernel, free_mask


def _fit3_gcv_ubre_full_derivatives(model, y, log_sp, method):
    method = str(method).lower()
    if method not in {"gcv", "ubre", "aic", "ubreaic"}:
        raise ValueError(f"Unsupported gam.fit3 GCV/UBRE method {method!r}.")

    sol, kernel, free_mask = _fit3_gcv_ubre_kernel(model, y, log_sp)
    if kernel is None:
        empty = np.empty((0,), dtype=np.float64)
        return empty, np.empty((0, 0), dtype=np.float64), free_mask

    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("score_gamma must be finite and positive.")

    nobs = float(model.n_samples_)
    if not np.isfinite(nobs) or nobs <= 0.0:
        raise ValueError("model.n_samples_ must be finite and positive.")

    D1 = np.asarray(kernel.D1, dtype=np.float64)
    D2 = np.asarray(kernel.D2, dtype=np.float64)
    trA = float(kernel.trA)
    trA1 = np.asarray(kernel.trA1, dtype=np.float64)
    trA2 = np.asarray(kernel.trA2, dtype=np.float64)
    dev = float(sol["deviance"])

    if method == "gcv":
        delta = nobs - gamma * trA
        delta2 = delta * delta
        delta3 = delta2 * delta
        delta4 = delta2 * delta2
        grad = nobs * D1 / delta2 + 2.0 * nobs * dev * trA1 * gamma / delta3
        hess = (
            2.0
            * gamma
            * nobs
            * (np.outer(trA1, D1) + np.outer(D1, trA1))
            / delta3
            + 6.0
            * nobs
            * dev
            * gamma
            * gamma
            * np.outer(trA1, trA1)
            / delta4
            + nobs * D2 / delta2
            + 2.0 * nobs * dev * gamma * trA2 / delta3
        )
    else:
        scale = getattr(model.family, "known_scale", None)
        if scale is None:
            raise ValueError(
                f"UBRE/AIC requested for family={model.family.name!r}, "
                "but the family does not have known scale."
            )
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Known scale for UBRE/AIC must be finite and positive.")
        grad = D1 / nobs + 2.0 * gamma * scale * trA1 / nobs
        hess = D2 / nobs + 2.0 * gamma * scale * trA2 / nobs

    return (
        np.asarray(grad, dtype=np.float64),
        np.asarray(hess, dtype=np.float64),
        free_mask,
    )


def criterion_gradient_gcv_ubre_pirls_exact(model, y, log_sp, method):
    grad, _hess, free_mask = _fit3_gcv_ubre_full_derivatives(model, y, log_sp, method)
    return np.asarray(grad[free_mask], dtype=np.float64)


def criterion_hessian_gcv_ubre_pirls_exact(model, y, log_sp, method):
    _grad, hess, free_mask = _fit3_gcv_ubre_full_derivatives(model, y, log_sp, method)
    free_idx = np.flatnonzero(free_mask)
    return np.asarray(hess[np.ix_(free_idx, free_idx)], dtype=np.float64)


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
    R: np.ndarray
    rank_root: np.ndarray
    A_inv: np.ndarray
    penalized_system_rank: int
    dropped_column_indices: np.ndarray
    pivot1: np.ndarray
    deviance_hessian_half: np.ndarray


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
    root_blocks: list[np.ndarray] | None = None
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
    ldet_XWX_plus_S: float | None
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
    R: np.ndarray | None
    Vt: np.ndarray | None
    neg_w: int
    Rh: np.ndarray | None
    rS_work: np.ndarray | None
    root_col_counts: tuple[int, ...]
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


def _mgcv_dgemm(left, right, *, transpose_left=False, transpose_right=False):
    """Preserve the matrix-multiply operand order used by ``mgcv_mmult()``."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_was_vector = left.ndim == 1
    right_was_vector = right.ndim == 1
    if left_was_vector:
        left = left.reshape(-1, 1)
    if right_was_vector:
        right = right.reshape(-1, 1)
    left_op = left.T if transpose_left else left
    right_op = right.T if transpose_right else right
    out = left_op @ right_op
    out = np.asarray(out, dtype=np.float64)
    if (left_was_vector or right_was_vector) and out.shape[1] == 1:
        return out[:, 0].copy()
    if (left_was_vector or right_was_vector) and out.shape[0] == 1:
        return out[0, :].copy()
    return out


def _mgcv_triangular_solve(R, rhs, *, transpose=False):
    """Apply the triangular solve used by ``mgcv_forwardsolve/backsolve``."""
    rhs = np.asarray(rhs, dtype=np.float64)
    rhs_was_vector = rhs.ndim == 1
    if rhs_was_vector:
        rhs = rhs.reshape(-1, 1)
    out = solve_triangular(
        np.asarray(R, dtype=np.float64),
        rhs,
        lower=False,
        trans="T" if transpose else "N",
        check_finite=False,
    )
    out = np.asarray(out, dtype=np.float64)
    return out[:, 0].copy() if rhs_was_vector else out


def _apply_pt(pk_state, rhs):
    """Operation-for-operation port of ``mgcv/src/gdi.c::applyPt()``."""
    work = _mgcv_triangular_solve(pk_state.R, rhs, transpose=True)
    if int(pk_state.neg_w) > 0:
        work = _mgcv_dgemm(pk_state.Vt, work)
    return np.asarray(work, dtype=np.float64)


def _apply_p(pk_state, rhs):
    """Operation-for-operation port of ``mgcv/src/gdi.c::applyP()``."""
    work = np.asarray(rhs, dtype=np.float64)
    if int(pk_state.neg_w) > 0:
        work = _mgcv_dgemm(pk_state.Vt, work, transpose_left=True)
    return _mgcv_triangular_solve(pk_state.R, work, transpose=False)


def _apply_penalized_inverse(pk_state, rhs):
    """Apply ``P'`` and then ``P`` exactly as ``mgcv::ift1()`` does."""
    return _apply_p(pk_state, _apply_pt(pk_state, rhs))


def _apply_penalized_inverse_root(rank_root, rhs):
    """Dense-root inverse application retained for the separate ``gdi2`` port."""
    rank_root = np.asarray(rank_root, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    return np.asarray(rank_root @ (rank_root.T @ rhs), dtype=np.float64)


def _mult_sk(root, x):
    """Operation-for-operation port of ``mgcv/src/gdi.c::multSk()``."""
    work = _mgcv_dgemm(root, x, transpose_left=True)
    return _mgcv_dgemm(root, work)


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


def _column_stack_or_empty(cols, nrow):
    if len(cols) == 0:
        return np.empty((int(nrow), 0), dtype=np.float64)
    return np.column_stack([np.asarray(col, dtype=np.float64) for col in cols])


def _restore_pirls_dbeta_to_fit_space(current, dbeta_rank):
    packed = np.asarray(dbeta_rank, dtype=np.float64).reshape(-1, 1)
    pivot1 = np.asarray(current.pivot1, dtype=np.int64)
    dropped = np.asarray(current.dropped_column_indices, dtype=np.int64)
    canonical_T = np.asarray(current.canonical.T, dtype=np.float64)

    unpermuted = permute_rows(packed, pivot1, reverse=True)
    full_canonical = restore_dropped_rows(
        unpermuted,
        int(canonical_T.shape[1]),
        dropped,
    )
    return np.asarray(canonical_T @ full_canonical, dtype=np.float64).ravel()


def _serialize_pirls_postproc_derivatives(kernel: _GDI1Kernel) -> dict[str, object]:
    current = kernel.current
    dbeta = _column_stack_or_empty(
        [
            _restore_pirls_dbeta_to_fit_space(current, col)
            for col in list(kernel.ift.dbeta)
        ],
        int(np.asarray(current.canonical.T, dtype=np.float64).shape[0]),
    )
    dW_obs = (
        None
        if kernel.ift.dW_obs is None
        else _column_stack_or_empty(list(kernel.ift.dW_obs), int(current.X.shape[0]))
    )
    return {
        "dbeta": dbeta,
        "dW_obs": dW_obs,
        "rp": getattr(current.canonical, "rp", None),
    }


def _negbin_ddeta_logtheta(family, y, mu, weights, *, deriv):
    """
    Port of `mgcv::nb()$Dd` + `gam.fit4.r::dDeta()` for negative binomial.

    Raw `Dd` ownership lives on the family object. This helper only chains
    `Dmu` derivatives to `eta`, preserving mgcv's special identity-link branch
    and its ratio-of-link-derivatives transform for non-identity links.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-14, None)
    wt = np.asarray(weights, dtype=np.float64)
    dd = family.Dd(y, mu, family.getTheta(False), wt, level=int(deriv))
    link_name = str(getattr(family, "link_name", getattr(family, "link", ""))).lower()
    if link_name == "identity":
        out = {
            "Deta": np.asarray(dd["Dmu"], dtype=np.float64),
            "Deta2": np.asarray(dd["Dmu2"], dtype=np.float64),
        }
        if deriv <= 0:
            return out
        out["Dth"] = np.asarray(dd["Dth"], dtype=np.float64)
        out["Detath"] = np.asarray(dd["Dmuth"], dtype=np.float64)
        out["Deta3"] = np.asarray(dd["Dmu3"], dtype=np.float64)
        out["Deta2th"] = np.asarray(dd["Dmu2th"], dtype=np.float64)
        if deriv <= 1:
            return out
        out["Deta4"] = np.asarray(dd["Dmu4"], dtype=np.float64)
        out["Dth2"] = np.asarray(dd["Dth2"], dtype=np.float64)
        out["Detath2"] = np.asarray(dd["Dmuth2"], dtype=np.float64)
        out["Deta2th2"] = np.asarray(dd["Dmu2th2"], dtype=np.float64)
        out["Deta3th"] = np.asarray(dd["Dmu3th"], dtype=np.float64)
        return out

    eta = np.asarray(family.link(mu), dtype=np.float64)
    ig1 = np.asarray(family.mu_eta(eta), dtype=np.float64)
    ig12 = ig1**2
    g2g = np.asarray(family.d2link(mu), dtype=np.float64) * ig12

    out = {
        "Deta": np.asarray(dd["Dmu"], dtype=np.float64) * ig1,
        "Deta2": np.asarray(dd["Dmu2"], dtype=np.float64) * ig12
        - np.asarray(dd["Dmu"], dtype=np.float64) * g2g * ig1,
    }
    if deriv <= 0:
        return out

    out["Dth"] = np.asarray(dd["Dth"], dtype=np.float64)
    out["Detath"] = np.asarray(dd["Dmuth"], dtype=np.float64) * ig1
    g3g = np.asarray(family.d3link(mu), dtype=np.float64) * (ig1**3)
    out["Deta3"] = (
        np.asarray(dd["Dmu3"], dtype=np.float64) * (ig1**3)
        - 3.0 * np.asarray(dd["Dmu2"], dtype=np.float64) * g2g * ig12
        + np.asarray(dd["Dmu"], dtype=np.float64) * (3.0 * g2g**2 - g3g) * ig1
    )
    out["Deta2th"] = (
        np.asarray(dd["Dmu2th"], dtype=np.float64) * ig12
        - np.asarray(dd["Dmuth"], dtype=np.float64) * g2g * ig1
    )
    if deriv <= 1:
        return out

    out["Dth2"] = np.asarray(dd["Dth2"], dtype=np.float64)
    out["Detath2"] = np.asarray(dd["Dmuth2"], dtype=np.float64) * ig1
    g4g = np.asarray(family.d4link(mu), dtype=np.float64) * (ig1**4)
    out["Deta4"] = (
        np.asarray(dd["Dmu4"], dtype=np.float64) * (ig12**2)
        - 6.0 * np.asarray(dd["Dmu3"], dtype=np.float64) * (ig1**3) * g2g
        + np.asarray(dd["Dmu2"], dtype=np.float64)
        * (15.0 * g2g**2 - 4.0 * g3g)
        * ig12
        - np.asarray(dd["Dmu"], dtype=np.float64)
        * (15.0 * g2g**3 - 10.0 * g2g * g3g + g4g)
        * ig1
    )
    out["Deta3th"] = (
        np.asarray(dd["Dmu3th"], dtype=np.float64) * (ig1**3)
        - 3.0 * np.asarray(dd["Dmu2th"], dtype=np.float64) * g2g * ig12
        + np.asarray(dd["Dmuth"], dtype=np.float64) * (3.0 * g2g**2 - g3g) * ig1
    )
    out["Deta2th2"] = (
        np.asarray(dd["Dmu2th2"], dtype=np.float64) * ig12
        - np.asarray(dd["Dmuth2"], dtype=np.float64) * g2g * ig1
    )
    return out


def _gamma_joint_kernel_state(model, y, log_sp, method, *, phi):
    method = str(method).upper()
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp, scale_reference=phi)
    kernel = _gdi1_kernel(
        model,
        y,
        sol,
        sp,
        method=method,
        rank_tol=_MGCV_GAM_FIT4_RANK_TOL,
    )
    Dp = float(sol["deviance"]) + float(kernel.bSb)
    Dp1 = np.asarray(kernel.D1 + kernel.bSb1, dtype=np.float64)
    Dp2 = np.asarray(kernel.D2 + kernel.bSb2, dtype=np.float64)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    _ls0, _score_lphi, phi_curv = _gamma_profile_objective_curvature(
        model,
        y,
        Dp,
        float(phi),
        mp,
        method=method,
    )
    state = {
        "K": kernel.K,
        "K1": kernel.K1,
        "K2": kernel.K2,
        "phi": float(phi),
        "phi_curv": phi_curv,
        "scale_est": float(sol["scale"]),
        "Dp": Dp,
        "Dp1": Dp1,
        "Dp2": Dp2,
    }
    model._pirls_reml_gamma_state_ = state
    return state, mp


def criterion_gradient_ml_reml_pirls_gamma_joint(model, y, log_sp, log_phi, method):
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gamma":
        raise NotImplementedError(
            "Joint PIRLS Gamma derivatives are implemented only for family='gamma'."
        )

    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        n_free = (
            int(np.sum(~np.asarray(model.smoothing_fixed_mask_, dtype=bool)))
            if model.smoothing_fixed_mask_ is not None
            else int(_n_smoothing_params(model) or 0)
        )
        return np.full(n_free + 1, np.nan, dtype=np.float64)
    state, mp = _gamma_joint_kernel_state(model, y, log_sp, method, phi=phi)

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

    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        n_free = (
            int(np.sum(~np.asarray(model.smoothing_fixed_mask_, dtype=bool)))
            if model.smoothing_fixed_mask_ is not None
            else int(_n_smoothing_params(model) or 0)
        )
        return np.full((n_free + 1, n_free + 1), np.nan, dtype=np.float64)
    state, mp = _gamma_joint_kernel_state(model, y, log_sp, method, phi=phi)

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


def _gaussian_joint_kernel_state(model, y, log_sp, method, *, phi):
    """Build the ``gam.fit3`` joint smoothing/scale derivative state."""
    method = str(method).upper()
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp, scale_reference=phi)
    kernel = _gdi1_kernel(
        model,
        y,
        sol,
        sp,
        method=method,
        rank_tol=_MGCV_GAM_FIT3_RANK_TOL,
    )
    y_arr = np.asarray(y, dtype=np.float64)
    weights = _prior_weights(model, y_arr)
    ls = np.asarray(
        model.family.ls(y_arr, weights, len(y_arr), float(phi)), dtype=np.float64
    )
    nobs = float(len(y_arr))
    n_true = getattr(model, "n_true_", None)
    if n_true is not None:
        n_true = float(n_true)
        if np.isfinite(n_true) and n_true > 0.0 and nobs > 0.0:
            ls *= n_true / nobs

    state = {
        "Dp": float(sol["deviance"]) + float(kernel.bSb),
        "Dp1": np.asarray(kernel.D1 + kernel.bSb1, dtype=np.float64),
        "Dp2": np.asarray(kernel.D2 + kernel.bSb2, dtype=np.float64),
        "K1": np.asarray(kernel.K1, dtype=np.float64),
        "K2": np.asarray(kernel.K2, dtype=np.float64),
        "ls": ls,
        "phi": float(phi),
    }
    model._pirls_reml_gaussian_state_ = state
    return state


def criterion_gradient_ml_reml_pirls_gaussian_joint(
    model, y, log_sp, log_phi, method
):
    """Joint ``(log sp, log scale)`` gradient from ``mgcv::gam.fit3``."""
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gaussian":
        raise NotImplementedError(
            "Joint PIRLS Gaussian derivatives require family='gaussian'."
        )
    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        return np.full(int(np.sum(_free_smoothing_mask(model))) + 1, np.nan)

    method = str(method).upper()
    state = _gaussian_joint_kernel_state(model, y, log_sp, method, phi=phi)
    gamma = float(model.score_gamma)
    free_mask = _free_smoothing_mask(model)
    grad_sp = state["Dp1"] / (2.0 * phi * gamma) + state["K1"]
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    reml_ind = 1.0 if method == "REML" else 0.0
    ls = state["ls"]
    grad_phi = (
        -state["Dp"] / (2.0 * phi) - float(ls[1]) * phi
    ) / gamma - 0.5 * mp * reml_ind
    return np.concatenate(
        [
            np.asarray(grad_sp[free_mask], dtype=np.float64),
            np.array([float(grad_phi)], dtype=np.float64),
        ]
    )


def criterion_hessian_ml_reml_pirls_gaussian_joint(
    model, y, log_sp, log_phi, method
):
    """Joint ``(log sp, log scale)`` Hessian from ``mgcv::gam.fit3``."""
    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name != "gaussian":
        raise NotImplementedError(
            "Joint PIRLS Gaussian derivatives require family='gaussian'."
        )
    phi = float(np.exp(float(log_phi)))
    n_free = int(np.sum(_free_smoothing_mask(model)))
    if not np.isfinite(phi) or phi <= 0.0:
        return np.full((n_free + 1, n_free + 1), np.nan)

    method = str(method).upper()
    state = _gaussian_joint_kernel_state(model, y, log_sp, method, phi=phi)
    gamma = float(model.score_gamma)
    free_mask = _free_smoothing_mask(model)
    hess_sp = state["Dp2"] / (2.0 * phi * gamma) + state["K2"]
    cross = -state["Dp1"] / (2.0 * phi * gamma)
    ls = state["ls"]
    hess_phi = (
        state["Dp"] / (2.0 * phi)
        - float(ls[2]) * phi * phi
        - float(ls[1]) * phi
    ) / gamma

    out = np.zeros((n_free + 1, n_free + 1), dtype=np.float64)
    out[:-1, :-1] = hess_sp[np.ix_(free_mask, free_mask)]
    out[:-1, -1] = cross[free_mask]
    out[-1, :-1] = cross[free_mask]
    out[-1, -1] = float(hess_phi)
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
        # Mirror gam.fit4.r L730: ls <- family$ls(y,weights,theta,scale)
        # lsth1 and lsth2 are subtracted from REML1/REML2 for the theta block.
        weights_arr = _prior_weights(model, y)
        ls_result = model.family.ls(
            np.asarray(y, dtype=np.float64),
            weights_arr,
            theta=float(log_theta),
            scale=1.0,
        )
        state = {
            "theta": float(theta),
            "scale_est": float(sol["scale"]),
            "Dp": float(gdi2.Dp),
            "Dp1": np.asarray(gdi2.Dp1, dtype=np.float64),
            "Dp2": np.asarray(gdi2.Dp2, dtype=np.float64),
            "K1": np.asarray(gdi2.K1_full, dtype=np.float64),
            "K2": np.asarray(gdi2.K2_full, dtype=np.float64),
            "lsth1": float(ls_result["lsth1"]),
            "lsth2": float(ls_result["lsth2"]),
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
    # Mirror gam.fit4.r L744: REML1 -= c(lsth1, rep(0, length(sp))) / gamma
    # Only the theta component (index 0) gets the ls derivative correction.
    grad_full = grad_full.copy()
    grad_full[0] -= float(state["lsth1"]) / gamma
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
    # Mirror gam.fit4.r L746-748:
    #   ls2 <- D2*0; ls2[1:nt,1:nt] <- lsth2
    #   REML2 <- ((D2+bSb2)/(2*scale) - ls2)/gamma + ldet2/2
    # Only the theta-theta block (index 0 in H_full) gets the ls2 correction.
    H_full = H_full.copy()
    H_full[0, 0] -= float(state["lsth2"]) / gamma
    sp_idx = 1 + np.flatnonzero(free_mask)
    keep = np.concatenate([sp_idx, np.array([0], dtype=np.int64)])
    return np.asarray(H_full[np.ix_(keep, keep)], dtype=np.float64)


def _gdi_pk_setup(model, sol, sp, *, deriv, rank_tol=None):
    """
    Single `mgcv::gdiPK()`-shaped setup routine.

    Owns current-sp canonical reparameterization plus solver rank/drop metadata.
    """
    X_source = np.asarray(_full_design_matrix(model), dtype=np.float64)
    if X_source.ndim != 2:
        raise ValueError("mgcv setup design matrix must be two-dimensional.")
    canonical = build_penalty_reparameterization_state(
        model, X_source, sp, deriv=deriv
    )
    X = np.asarray(X_source, dtype=np.float64) @ np.asarray(
        canonical.T, dtype=np.float64
    )
    W = np.asarray(sol["working_weights"], dtype=np.float64)
    XtWX = X.T @ (W[:, None] * X)
    P = np.asarray(canonical.St, dtype=np.float64)
    A = XtWX + P
    q_total_full = int(np.asarray(A, dtype=np.float64).shape[0])
    q_null_full = int(canonical.Mp)
    q_range_full = int(q_total_full - q_null_full)
    # Mirror `mgcv::gdiPK()` exactly here: use the canonical total-penalty roots
    # produced by the current `gam.fit3/gam.fit4` reparameterization state,
    # rather than rebuilding a fresh dense square root from `St`.
    penalty_sqrt = np.asarray(canonical.Sr, dtype=np.float64)
    penalty_rank_rows = np.asarray(canonical.Eb, dtype=np.float64)
    root_cols = []
    rSncol = []
    roots = _reparam_roots_by_smoothing_parameter(
        model, canonical.rp.get("rS", []), int(_n_smoothing_params(model) or 0)
    )
    rows_e = q_range_full
    for root in roots:
        root = np.asarray(root, dtype=np.float64)
        if root.size:
            root_full = np.zeros((q_total_full, int(root.shape[1])), dtype=np.float64)
            root_full[:rows_e, :] = root
        else:
            root_full = np.empty((q_total_full, 0), dtype=np.float64)
        root_cols.append(root_full)
        rSncol.append(int(root_full.shape[1]))
    rS = (
        np.concatenate(root_cols, axis=1)
        if root_cols
        else np.empty((q_total_full, 0), dtype=np.float64)
    )
    if rank_tol is None:
        rank_tol = _MGCV_GAM_FIT3_RANK_TOL
    working_response = sol.get("working_response", None)
    if working_response is None:
        working_response = sol.get("z", None)
    if working_response is None:
        # Some direct low-level callers provide only the converged linear
        # predictor. `gdiPK` uses this vector for PK'z/rank-space staging; when
        # a working response is unavailable, the fitted predictor is the only
        # mgcv-shaped state present in the supplied solution.
        working_response = sol["eta"]
    qr_state = build_penalized_qr_state_nonnegative(
        np.asarray(X, dtype=np.float64),
        np.asarray(working_response, dtype=np.float64),
        np.asarray(W, dtype=np.float64),
        penalty_sqrt_E=np.asarray(penalty_sqrt, dtype=np.float64),
        penalty_rank_Es=np.asarray(penalty_rank_rows, dtype=np.float64),
        rS=np.asarray(rS, dtype=np.float64),
        rank_tol=float(rank_tol),
        reml=True,
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
    R = np.asarray(qr_state.R, dtype=np.float64)
    Vt = None if qr_state.Vt is None else np.asarray(qr_state.Vt, dtype=np.float64)
    rS_work = np.asarray(qr_state.rS_work, dtype=np.float64)
    ldet_xwxs = float(qr_state.ldet_XWX_plus_S)
    X_rank = _drop_permute_columns(X, dropped_idx, pivot1)
    # Mirror `mgcv/src/gdi.c::gdiPK()` / `ift1()`: downstream derivative and
    # `b'Sb` staging works on the pivoted rank-space coefficient representative
    # `PK'z`, not on a transformed version of the final reported coefficients.
    beta_rank = np.asarray(qr_state.PKtz, dtype=np.float64).ravel()
    P_rank = _drop_permute_symmetric(P, dropped_idx, pivot1)
    XtWX_rank = X_rank.T @ (W[:, None] * X_rank)
    # Mirror `mgcv/src/gdi.c::gdiPK()`: downstream `gdi1/gdi2` carry `Rh` with
    # `Rh'Rh = X'WX + S` in the pivoted reduced parameterization rather than
    # refactorizing a reconstructed normal-equation matrix.
    A_rank = np.asarray(Rh.T @ Rh, dtype=np.float64)
    A_rank = 0.5 * (A_rank + A_rank.T)
    if rank > 0:
        A_inv_rank = np.asarray(Pk @ Pk.T, dtype=np.float64)
        A_inv_rank = 0.5 * (A_inv_rank + A_inv_rank.T)
    else:
        A_inv_rank = np.empty((0, 0), dtype=np.float64)
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
        R=np.asarray(Rh, dtype=np.float64),
        rank_root=np.asarray(Pk, dtype=np.float64),
        A_inv=A_inv_rank,
        penalized_system_rank=rank,
        dropped_column_indices=dropped_idx,
        pivot1=pivot1,
        deviance_hessian_half=np.asarray(
            qr_state.deviance_hessian_half, dtype=np.float64
        ),
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
            R=R,
            Vt=Vt,
            neg_w=int(qr_state.neg_w),
            Rh=Rh,
            rS_work=rS_work,
            root_col_counts=tuple(int(v) for v in rSncol),
            ldet_XWX_plus_S=ldet_xwxs,
        ),
    )


def _reparam_roots_by_smoothing_parameter(model, roots, n_sp):
    roots = [np.asarray(root, dtype=np.float64) for root in list(roots)]
    n_sp = int(n_sp)
    if n_sp <= 0:
        return []
    if not roots:
        return [np.empty((0, 0), dtype=np.float64) for _ in range(n_sp)]

    q = int(roots[0].shape[0])
    grouped = [[] for _ in range(n_sp)]
    penalty_blocks = list(_penalty_blocks_seq(model))
    for i, root in enumerate(roots):
        if int(root.shape[0]) != q:
            raise ValueError("All reparameterized penalty roots must have same rows.")
        if i < len(penalty_blocks):
            sp_idx = int(penalty_blocks[i].smoothing_index)
        else:
            sp_idx = i
        if 0 <= sp_idx < n_sp and root.size:
            grouped[sp_idx].append(root)

    out = []
    for parts in grouped:
        if parts:
            out.append(np.concatenate(parts, axis=1))
        else:
            out.append(np.empty((q, 0), dtype=np.float64))
    return out


def _canonical_penalty_derivative_matrices(model, canonical, sp, n_sp):
    q = int(np.asarray(canonical.St, dtype=np.float64).shape[0])
    rows_e = q - int(canonical.Mp)
    mats = [np.zeros((q, q), dtype=np.float64) for _ in range(int(n_sp))]
    roots = _reparam_roots_by_smoothing_parameter(
        model, canonical.rp.get("rS", []), int(n_sp)
    )
    for j, root in enumerate(roots):
        root = np.asarray(root, dtype=np.float64)
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
    eta = np.asarray(sol.get("gdi1_eta", sol["eta"]), dtype=np.float64)
    mu = np.asarray(sol.get("gdi1_mu", sol["mu"]), dtype=np.float64)
    W = np.asarray(current.W, dtype=np.float64)
    n_sp = int(_n_smoothing_params(model) or 0)
    rank = int(beta.size)
    root_blocks = []
    col_off = 0
    root_col_counts = tuple(getattr(pk_state, "root_col_counts", ()) or ())
    for j in range(n_sp):
        ncol = int(root_col_counts[j]) if j < len(root_col_counts) else 0
        root = np.asarray(
            pk_state.rS_work[:, col_off : col_off + ncol], dtype=np.float64
        )
        root_blocks.append(root)
        col_off += ncol
    P_derivs = []
    for j, root in enumerate(root_blocks):
        if root.size:
            Pj = _mgcv_dgemm(root, root, transpose_right=True)
            for col in range(Pj.shape[1]):
                for row in range(Pj.shape[0]):
                    Pj[row, col] *= float(sp[j])
        else:
            Pj = np.zeros((rank, rank), dtype=np.float64)
        P_derivs.append(np.asarray(Pj, dtype=np.float64))

    family_name = str(getattr(model.family, "name", "")).lower()
    if family_name == "negbin":
        dd = _negbin_ddeta_logtheta(
            model.family,
            y,
            mu,
            _prior_weights(model, y),
            deriv=2,
        )
        dW_eta = 0.5 * np.asarray(dd["Deta3"], dtype=np.float64)
        d2W_eta = 0.5 * np.asarray(dd["Deta4"], dtype=np.float64)
    else:
        dW_eta, d2W_eta = _working_weight_derivatives_wrt_linpred(
            model, y, eta, mu, W
        )

    dbeta_matrix = np.zeros((rank, n_sp), dtype=np.float64, order="F")
    for j, root in enumerate(root_blocks):
        if root.size:
            Skb = np.asarray(_mult_sk(root, beta), dtype=np.float64)
            for row in range(rank):
                Skb[row] *= -float(sp[j])
            work = _apply_pt(pk_state, Skb)
            dbeta_matrix[:, j] = _apply_p(pk_state, work)

    if n_sp:
        eta1_matrix = np.asarray(_mgcv_dgemm(X, dbeta_matrix), dtype=np.float64)
        if n_sp == 1:
            eta1_matrix = eta1_matrix.reshape(-1, 1)
    else:
        eta1_matrix = np.empty((X.shape[0], 0), dtype=np.float64)
    dbeta = [np.asarray(dbeta_matrix[:, j], dtype=np.float64) for j in range(n_sp)]
    deta = [np.asarray(eta1_matrix[:, j], dtype=np.float64) for j in range(n_sp)]
    dA = [None] * n_sp
    dXtWX = [None] * n_sp
    d2beta_mat = [[None] * n_sp for _ in range(n_sp)]
    d2A_mat = [[None] * n_sp for _ in range(n_sp)]
    d2XtWX_mat = [[None] * n_sp for _ in range(n_sp)]

    for j in range(n_sp):
        deta_j = deta[j]
        dW_j = dW_eta * deta_j
        dXtWX_j = X.T @ (dW_j[:, None] * X)
        dXtWX[j] = dXtWX_j
        dA[j] = dXtWX_j + P_derivs[j]

    d2beta_columns = []
    second_pairs = []
    for j, root_j in enumerate(root_blocks):
        for k in range(j, n_sp):
            work = np.empty(X.shape[0], dtype=np.float64)
            for row in range(X.shape[0]):
                work[row] = -deta[j][row] * deta[k][row] * dW_eta[row]
            Skb = np.asarray(
                _mgcv_dgemm(X, work, transpose_left=True), dtype=np.float64
            )
            if root_j.size:
                penalty_work = np.asarray(_mult_sk(root_j, dbeta[k]), dtype=np.float64)
                for row in range(rank):
                    Skb[row] += -float(sp[j]) * penalty_work[row]
            root_k = root_blocks[k]
            if root_k.size:
                penalty_work = np.asarray(_mult_sk(root_k, dbeta[j]), dtype=np.float64)
                for row in range(rank):
                    Skb[row] += -float(sp[k]) * penalty_work[row]
            solve_work = _apply_pt(pk_state, Skb)
            d2beta_jk = np.asarray(_apply_p(pk_state, solve_work), dtype=np.float64)
            if j == k:
                for row in range(rank):
                    d2beta_jk[row] += dbeta[j][row]
            d2beta_columns.append(d2beta_jk)
            second_pairs.append((j, k))
            d2beta_mat[j][k] = d2beta_jk
            d2beta_mat[k][j] = d2beta_jk

    if d2beta_columns:
        d2beta_packed = np.asfortranarray(np.column_stack(d2beta_columns))
        eta2_packed = np.asarray(_mgcv_dgemm(X, d2beta_packed), dtype=np.float64)
        if len(d2beta_columns) == 1:
            eta2_packed = eta2_packed.reshape(-1, 1)
    else:
        eta2_packed = np.empty((X.shape[0], 0), dtype=np.float64)

    for pair_col, (j, k) in enumerate(second_pairs):
        d2beta_jk = d2beta_mat[j][k]
        d2eta_jk = eta2_packed[:, pair_col]
        d2W_jk = d2W_eta * deta[j] * deta[k] + dW_eta * d2eta_jk
        d2XtWX_jk = X.T @ (d2W_jk[:, None] * X)
        d2A_jk = d2XtWX_jk + (P_derivs[j] if j == k else 0.0)
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
        root_blocks=root_blocks,
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
    eta = np.asarray(sol.get("gdi1_eta", sol["eta"]), dtype=np.float64)
    mu = np.asarray(sol.get("gdi1_mu", sol["mu"]), dtype=np.float64)
    if str(getattr(model.family, "name", "")).lower() == "negbin":
        dd = _negbin_ddeta_logtheta(
            model.family,
            y,
            mu,
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

    family = model.family
    mu1 = np.asarray(family.mu_eta(eta), dtype=np.float64)
    variance = np.asarray(family.variance(mu), dtype=np.float64)
    prior_weights = _prior_weights(model, y)
    residual = np.asarray(y, dtype=np.float64) - mu
    v1 = np.empty_like(residual, dtype=np.float64)
    for row in range(residual.size):
        v1[row] = (
            -2.0
            * prior_weights[row]
            * residual[row]
            * mu1[row]
            / variance[row]
        )
    dev_grad = np.asarray(
        _mgcv_dgemm(current.X, v1, transpose_left=True),
        dtype=np.float64,
    )
    # gdi.c::gdiPK() forms X'WX from the unpenalized weighted-design QR factor;
    # gdi1() then doubles it to obtain the coefficient-scale deviance Hessian.
    # Preserve that upstream path instead of reconstructing the same matrix
    # from X, since the two forms cease to be numerically interchangeable when
    # the penalized deviance derivatives are nearly singular.
    dev_hess = 2.0 * np.asarray(
        current.deviance_hessian_half, dtype=np.float64
    )
    return _deviance_chained_to_smoothing(dev_grad, dev_hess, ift.dbeta, ift.d2beta_mat)


def _gdi1_bsb_terms(current, ift, sp):
    """Operation-for-operation port of ``mgcv/src/gdi.c::get_bSb()``."""
    beta = np.asarray(current.beta, dtype=np.float64)
    E = _drop_permute_columns(
        np.asarray(current.canonical.Sr, dtype=np.float64),
        current.dropped_column_indices,
        current.pivot1,
    )
    work = np.asarray(_mgcv_dgemm(E, beta), dtype=np.float64)
    Sb = np.asarray(_mgcv_dgemm(E, work, transpose_left=True), dtype=np.float64)
    bSb = 0.0
    for row in range(beta.size):
        bSb += beta[row] * Sb[row]

    n_sp = len(ift.dbeta)
    if n_sp == 0:
        return bSb, np.empty((0,), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    bSb1 = np.zeros(n_sp, dtype=np.float64)
    bSb2 = np.zeros((n_sp, n_sp), dtype=np.float64)
    root_blocks = list(ift.root_blocks or [])
    if len(root_blocks) != n_sp:
        raise RuntimeError("internal: ift1 penalty-root blocks are incomplete.")
    Skb = []
    for j, root in enumerate(root_blocks):
        if root.size:
            root_work = np.asarray(
                _mgcv_dgemm(root, beta, transpose_left=True), dtype=np.float64
            )
            for col in range(root_work.size):
                root_work[col] *= float(sp[j])
            Skb_j = np.asarray(_mgcv_dgemm(root, root_work), dtype=np.float64)
        else:
            Skb_j = np.zeros_like(beta)
        Skb.append(Skb_j)
        xx = 0.0
        for row in range(beta.size):
            xx += beta[row] * Skb_j[row]
        bSb1[j] = xx

    for m in range(n_sp):
        work1 = np.asarray(_mgcv_dgemm(E, ift.dbeta[m]), dtype=np.float64)
        work = np.asarray(
            _mgcv_dgemm(E, work1, transpose_left=True), dtype=np.float64
        )
        for k in range(m, n_sp):
            xx = 0.0
            for row in range(beta.size):
                xx += ift.d2beta_mat[m][k][row] * Sb[row]
            val = 2.0 * xx
            xx = 0.0
            for row in range(beta.size):
                xx += ift.dbeta[k][row] * work[row]
            val += 2.0 * xx
            xx = 0.0
            for row in range(beta.size):
                xx += ift.dbeta[m][row] * Skb[k][row]
            val += 2.0 * xx
            xx = 0.0
            for row in range(beta.size):
                xx += ift.dbeta[k][row] * Skb[m][row]
            val += 2.0 * xx
            if k == m:
                val += float(bSb1[k])
            bSb2[k, m] = val
            bSb2[m, k] = val

    b1_matrix = np.asfortranarray(np.column_stack(ift.dbeta))
    work = np.asarray(
        _mgcv_dgemm(b1_matrix, Sb, transpose_left=True), dtype=np.float64
    )
    for j in range(n_sp):
        bSb1[j] += 2.0 * work[j]
    return bSb, bSb1, bSb2


def _diag_abt_trace(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    r, c = A.shape
    diag = np.zeros(r, dtype=np.float64)
    if c > 0:
        diag[:] = A[:, 0] * B[:, 0]
        for j in range(1, c):
            diag += A[:, j] * B[:, j]
    return diag, float(np.sum(diag))


def _get_xtwx_dense(X, w):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    r, c = X.shape
    out = np.zeros((c, c), dtype=np.float64)
    if r == 0 or c == 0:
        return out
    work = np.empty(r, dtype=np.float64)
    for i in range(c):
        work[:] = X[:, i] * w
        for j in range(i + 1):
            xx = float(np.dot(X[:, j], work))
            out[i, j] = xx
            out[j, i] = xx
    return out


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
    dRinv = [None] * n_sp
    d1 = np.zeros(n_sp, dtype=np.float64)
    d2 = np.zeros((n_sp, n_sp), dtype=np.float64)
    for j in range(n_sp):
        dRinv_j = -Rinv @ dArr[j] @ Rinv
        dRinv[j] = dRinv_j
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
            dRinv_j = np.asarray(dRinv[j], dtype=np.float64)
            dRinv_k = np.asarray(dRinv[k], dtype=np.float64)
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

    wabs = np.abs(np.asarray(current.W, dtype=np.float64))
    if np.any(~np.isfinite(wabs)) or np.any(wabs == 0.0):
        return None
    Tk = [np.asarray(dw, dtype=np.float64) / wabs for dw in ift.dW_obs]
    Tkm = [
        [np.asarray(d2w, dtype=np.float64) / wabs for d2w in row]
        for row in ift.d2W_obs_mat
    ]
    diagKKt, _ = _diag_abt_trace(K, K)

    KtTK = [_get_xtwx_dense(K, tk) for tk in Tk]
    det1_x = np.array([float(np.dot(tk, diagKKt)) for tk in Tk], dtype=np.float64)

    root_col_counts = tuple(getattr(pk_state, "root_col_counts", ()) or ())
    col_off = 0
    PtSP = []
    trPtSP = np.zeros(n_sp, dtype=np.float64)
    for m in range(n_sp):
        ncol = int(root_col_counts[m]) if m < len(root_col_counts) else 0
        root_block = rS_work[:, col_off : col_off + ncol]
        col_off += root_block.shape[1]
        PtrSm = (
            P.T @ root_block
            if root_block.size
            else np.empty((rank, 0), dtype=np.float64)
        )
        PtSP_m = (
            _get_xtwx_dense(PtrSm.T, np.ones(PtrSm.shape[1], dtype=np.float64))
            if PtrSm.size
            else np.zeros((rank, rank), dtype=np.float64)
        )
        PtSP.append(PtSP_m)
        _, xx = _diag_abt_trace(PtrSm, PtrSm)
        trPtSP[m] = float(sp[m] * xx)
        det1_x[m] += trPtSP[m]

    det2_x = np.zeros((n_sp, n_sp), dtype=np.float64)
    for m in range(n_sp):
        for k in range(m, n_sp):
            val = float(np.dot(Tkm[m][k], diagKKt))
            _, xx = _diag_abt_trace(KtTK[k], KtTK[m])
            val -= xx
            if k == m:
                val += trPtSP[m]
            _, xx = _diag_abt_trace(KtTK[k], PtSP[m])
            val -= float(sp[m] * xx)
            _, xx = _diag_abt_trace(KtTK[m], PtSP[k])
            val -= float(sp[k] * xx)
            _, xx = _diag_abt_trace(PtSP[k], PtSP[m])
            val -= float(sp[m] * sp[k] * xx)
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

    Upstream drops pivoted null-space columns after rank detection, so the range
    block is selected by `nulli` membership rather than by leading columns.
    """
    range_idx = np.asarray(pk_state.range_idx, dtype=np.int64)
    n_sp = len(ift.dA)
    if range_idx.size == 0:
        zero = np.zeros(n_sp, dtype=np.float64)
        return 0.0, zero, np.zeros((n_sp, n_sp), dtype=np.float64)

    A = np.asarray(current.A, dtype=np.float64)
    Arr = A[np.ix_(range_idx, range_idx)]
    dArr = [
        np.asarray(dAj[np.ix_(range_idx, range_idx)], dtype=np.float64)
        for dAj in ift.dA
    ]
    d2Arr = [
        [np.asarray(d2A[np.ix_(range_idx, range_idx)], dtype=np.float64) for d2A in row]
        for row in ift.d2A_mat
    ]
    cR, loR = cho_factor(Arr, check_finite=False)
    logdet_R = 2.0 * float(np.sum(np.log(np.abs(np.diag(cR)))))
    Rinv = cho_solve((cR, loR), np.eye(range_idx.size), check_finite=False)
    detR1, detR2 = _logdet_penalized_system_derivatives(Rinv, dArr, d2Arr)

    logdet_S, detS1, detS2 = _stable_penalty_logdet_derivatives(model, sp, order=2)
    if not np.isfinite(logdet_S):
        return (
            np.inf,
            np.full(n_sp, np.nan, dtype=np.float64),
            np.full((n_sp, n_sp), np.nan, dtype=np.float64),
        )
    return (
        0.5 * (logdet_R - logdet_S),
        0.5 * (detR1 - detS1),
        0.5 * (detR2 - detS2),
    )


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


def _gdi1_kernel(model, y, sol, sp, *, method, rank_tol=None):
    """Structured Python analogue of `mgcv::gdi1()` on canonical current-sp state."""
    setup = _gdi_pk_setup(model, sol, sp, deriv=2, rank_tol=rank_tol)
    current = setup.current
    pk_state = setup.pk
    ift = _gdi1_ift1_state(model, y, sol, sp, current, pk_state)
    D1, D2 = _gdi1_deviance_terms(model, y, sol, current, ift)
    bSb, bSb1, bSb2 = _gdi1_bsb_terms(current, ift, sp)
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
        ldet_XWX_plus_S=float(pk_state.ldet_XWX_plus_S),
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
    if str(method).upper() == "ML":
        range_idx = np.asarray(pk_state.range_idx, dtype=np.int64)
    else:
        range_idx = np.arange(q_range, dtype=np.int64)
    Arr = A[np.ix_(range_idx, range_idx)]
    dArr = [
        np.asarray(dAj[np.ix_(range_idx, range_idx)], dtype=np.float64) for dAj in dA
    ]
    d2Arr = [
        [np.asarray(d2A[np.ix_(range_idx, range_idx)], dtype=np.float64) for d2A in row]
        for row in d2A_mat
    ]
    cR, loR = cho_factor(Arr, check_finite=False)
    logdet_R = 2.0 * float(np.sum(np.log(np.abs(np.diag(cR)))))
    Rinv = cho_solve((cR, loR), np.eye(range_idx.size), check_finite=False)
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
    P_root = np.asarray(pk_state.P, dtype=np.float64)
    weights = _prior_weights(model, y)
    dd = _negbin_ddeta_logtheta(model.family, y, mu, weights, deriv=2)
    P_sp = [
        _drop_permute_symmetric(Pj, current.dropped_column_indices, current.pivot1)
        for Pj in _canonical_penalty_derivative_matrices(
            model, current.canonical, sp, int(_n_smoothing_params(model) or 0)
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
        dbeta_i = _apply_penalized_inverse_root(P_root, rhs)
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
            d2beta_ik = _apply_penalized_inverse_root(P_root, rhs)
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
    setup = _gdi_pk_setup(
        model,
        sol,
        sp,
        deriv=2,
        rank_tol=_MGCV_GAM_FIT4_RANK_TOL,
    )
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

    gdi1 = _gdi1_kernel(
        model,
        y,
        sol,
        sp,
        method=method,
        rank_tol=_MGCV_GAM_FIT4_RANK_TOL,
    )
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
    gdi1 = _gdi1_kernel(
        model,
        y,
        sol,
        sp,
        method=method,
        rank_tol=_MGCV_GAM_FIT4_RANK_TOL,
    )
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


def _gdi2_gaussian_joint_kernel(model, y, sol, sp, *, method, need_hessian):
    """
    Gaussian profiled-scale branch of the joint staging.

    mgcv/R/gam.fit3.r:628-637 borders the REML/ML derivatives with the
    log-scale block. With the gaussian `ls` from mgcv/R/gam.fit3.r:2503-2508
    (`ls[2] = -n/(2*phi)`, `ls[3] = n/(2*phi^2)`), solving `dlr.dlphi = 0`
    gives the closed-form profile scale `phi = Dp / (n - gamma*Mp*remlInd)`
    and the curvature at that point collapses to `Dp / (2*phi*gamma)`.
    """
    method_u = str(method).upper()
    gdi1 = _gdi1_kernel(
        model,
        y,
        sol,
        sp,
        method=method,
        rank_tol=_MGCV_GAM_FIT3_RANK_TOL,
    )
    gamma = float(model.score_gamma)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    Dp = float(sol["deviance"]) + float(gdi1.bSb)
    nobs = float(len(np.asarray(y)))
    n_eff = nobs
    n_true = getattr(model, "n_true_", None)
    if n_true is not None:
        n_true = float(n_true)
        if np.isfinite(n_true) and n_true > 0.0:
            n_eff = n_true
    reml_ind = 1.0 if method_u == "REML" else 0.0
    denom = n_eff - gamma * mp * reml_ind
    if not np.isfinite(denom) or denom <= 0.0 or not np.isfinite(Dp) or Dp <= 0.0:
        raise RuntimeError(
            "Gaussian exact PIRLS derivatives require positive profile scale."
        )
    phi = Dp / denom
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
    phi_curv = denom / (2.0 * gamma)
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


def _gdi2_general_family_kernel(model, y, sol, sp, *, method, need_hessian):
    """
    Port-shaped `gam.fit5()` decomposition for theta-free general families.

    Mirrors `mgcv/R/gam.fit4.r::gam.fit5` where the REML/ML score splits into
    `Dp / (2 * gamma) + K`, with `Dp = -2 * ll + b'Sb`.
    """
    del sol
    from ....fit.solvers.general_family.fixed_smoothing import (
        run_general_family_fixed_smoothing,
    )
    from ....fit.solvers.general_family.newton import _sl_term_mult

    gamma = float(model.score_gamma)
    run = run_general_family_fixed_smoothing(
        model,
        y,
        sp,
        weights=_prior_weights(model, y),
        deriv=2 if need_hessian else 1,
        score_type=method,
    )
    fit = run["fit"]
    setup = run["setup"]

    coef = np.asarray(fit["coef"], dtype=np.float64)
    St_full = np.asarray(fit["St_full"], dtype=np.float64)
    ll_val = float(fit["l"])
    penalty_full = float(coef @ (St_full @ coef))
    Dp = float(-2.0 * ll_val + penalty_full)

    Skb = _sl_term_mult(setup.Sl, coef, full=True)
    Dp1 = np.asarray(
        [float(np.sum(coef * np.asarray(Skb_i, dtype=np.float64))) for Skb_i in Skb],
        dtype=np.float64,
    )
    score1 = np.asarray(fit.get("score1", np.zeros_like(Dp1)), dtype=np.float64)
    K1 = np.asarray(score1 - Dp1 / (2.0 * gamma), dtype=np.float64)

    Dp2 = None
    K2 = None
    if need_hessian:
        db_drho = np.asarray(fit["db_drho"], dtype=np.float64)
        llbb = np.asarray(fit["lbb"], dtype=np.float64)
        n_sp = int(db_drho.shape[1])
        d2pen = np.zeros((n_sp, n_sp), dtype=np.float64)
        d2l = np.zeros((n_sp, n_sp), dtype=np.float64)
        for i in range(n_sp):
            Sd1b = St_full @ db_drho[:, i]
            for j in range(i, n_sp):
                val = 2.0 * float(
                    np.sum(
                        db_drho[:, i] * np.asarray(Skb[j], dtype=np.float64)
                        + db_drho[:, j] * np.asarray(Skb[i], dtype=np.float64)
                        + db_drho[:, j] * Sd1b
                    )
                )
                if i == j:
                    val += float(np.sum(coef * np.asarray(Skb[i], dtype=np.float64)))
                d2pen[i, j] = d2pen[j, i] = val
                d2l[i, j] = d2l[j, i] = float(db_drho[:, i] @ (llbb @ db_drho[:, j]))
        Dp2 = np.asarray(d2pen - 2.0 * d2l, dtype=np.float64)
        score2 = np.asarray(fit["score2"], dtype=np.float64)
        K2 = np.asarray(score2 - Dp2 / (2.0 * gamma), dtype=np.float64)

    return _GDI2Kernel(
        gdi1=None,
        phi=1.0,
        phi_curv=np.inf,
        Dp=Dp,
        Dp1=Dp1,
        Dp2=Dp2,
        K1_full=K1,
        K2_full=K2,
        extra_name=None,
        extra_value=None,
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
    if family_name == "gaussian":
        return _gdi2_gaussian_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    if family_name == "negbin":
        return _gdi2_negbin_joint_kernel(
            model, y, sol, sp, method=method, need_hessian=need_hessian
        )
    if str(getattr(model.family, "family_class", "")).lower() == "general":
        n_theta = int(getattr(model.family, "n_theta", 0) or 0)
        if n_theta != 0:
            raise NotImplementedError(
                "Generic `gdi2` current-sp extended-family port is implemented "
                "only for theta-free general families."
            )
        return _gdi2_general_family_kernel(
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
    if getattr(model.family, "known_scale", None) is None and family_name not in {
        "gamma",
        "gaussian",
    }:
        raise NotImplementedError(
            "Exact PIRLS ML/REML gradients are currently implemented only for "
            "fixed-scale families, plus Gamma and Gaussian via the profiled "
            "scale branch."
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

    if getattr(model.family, "known_scale", None) is None and family_name in {
        "gamma",
        "gaussian",
    }:
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
    if getattr(model.family, "known_scale", None) is None and family_name not in {
        "gamma",
        "gaussian",
    }:
        raise NotImplementedError(
            "Exact PIRLS ML/REML Hessians are currently implemented only for "
            "fixed-scale families, plus Gamma and Gaussian via the profiled "
            "scale branch."
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
    postproc_derivs = _serialize_pirls_postproc_derivatives(kernel)
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
        **postproc_derivs,
    }

    if full_hess is None:
        raise RuntimeError("Exact PIRLS Hessian assembly did not produce a result.")
    return np.asarray(full_hess[np.ix_(free_idx, free_idx)], dtype=np.float64)
