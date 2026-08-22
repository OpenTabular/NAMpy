"""Gamma-family PIRLS joint ML/REML derivative kernels."""

import numpy as np

from .....model_state import _coef_column_offset, _fit_workspace, _n_smoothing_params
from ....backends import solve_pirls_given_smoothing
from ....smoothing_params import expand_smoothing_params_from_log
from ...reparam import _static_penalty_null_dim, can_use_simple_ml_reml_structure
from .common import _prior_weights
from .derivatives import (
    _MGCV_GAM_FIT4_RANK_TOL,
    _free_smoothing_mask,
    _gdi1_kernel,
    _GDI2Kernel,
)


def _gamma_profile_objective_curvature(model, y, Dp, phi, mp, *, method):
    phi = float(phi)
    if not np.isfinite(phi) or phi <= 0.0:
        return np.inf, np.nan, np.nan
    weights = _prior_weights(model, y)
    ls = np.asarray(model.family.ls(y, weights, len(y), phi), dtype=np.float64)
    nobs = float(len(y))
    n_true = getattr(model, "n_true_", None)
    if n_true is None:
        n_true = nobs
    n_true = float(n_true)
    if not np.isfinite(n_true) or n_true <= 0.0 or not np.isfinite(nobs) or nobs <= 0.0:
        fac = 1.0
    else:
        fac = n_true / nobs
    ls *= fac
    reml_ind = 1.0 if method == "REML" else 0.0
    gamma = float(model.score_gamma)
    score_lphi = (-Dp / (2.0 * phi) - ls[1] * phi) / gamma - 0.5 * mp * reml_ind
    curv_lphi = (Dp / (2.0 * phi) - ls[2] * (phi**2) - ls[1] * phi) / gamma
    return float(ls[0]), float(score_lphi), float(curv_lphi)


def _solve_gamma_profile_scale(model, y, Dp, mp, *, method, init_scale):
    phi = float(max(init_scale, 1e-12))
    if not np.isfinite(phi) or phi <= 0.0:
        phi = max(float(Dp) / max(float(len(y)), 1.0), 1e-6)

    for _ in range(40):
        _, score_lphi, curv_lphi = _gamma_profile_objective_curvature(
            model, y, Dp, phi, mp, method=method
        )
        if not np.isfinite(score_lphi) or not np.isfinite(curv_lphi):
            break
        if abs(score_lphi) <= 1e-12:
            return phi
        if abs(curv_lphi) <= 1e-14:
            break
        step = score_lphi / curv_lphi
        if not np.isfinite(step):
            break
        if abs(step) <= 1e-12:
            return phi
        new_phi = float(np.exp(np.log(phi) - step))
        if not np.isfinite(new_phi) or new_phi <= 0.0:
            break
        phi = new_phi
    return phi


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
    _fit_workspace(model).pirls_reml_gamma_state = state
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
    free_mask = _free_smoothing_mask(model)
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
    free_mask = _free_smoothing_mask(model)

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


def gdi2_gamma_joint_kernel(model, y, sol, sp, *, method, need_hessian):
    """Current Gamma branch of the mgcv ``gdi2`` extended-family staging."""
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


def criterion_ml_reml_pirls_gamma_joint(model, y, log_sp, log_phi, method):
    """Joint Gamma PIRLS ML/REML outer objective."""
    from .value import (
        _pirls_laplace_logdet_term,
        _pirls_tensor_coefficient_space_logdet_term,
    )

    method = str(method).upper()
    if str(getattr(model.family, "name", "")).lower() != "gamma":
        raise NotImplementedError(
            "Joint PIRLS Gamma outer objective is implemented only for family='gamma'."
        )
    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        return np.inf

    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp, scale_reference=phi)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf
    use_exact_logdet = bool(
        can_use_simple_ml_reml_structure(model)
        and not model._has_tensor_terms()
        and bool(getattr(model.family, "supports_exact_pirls_first_derivatives", False))
    )
    try:
        if use_exact_logdet:
            kernel = _gdi1_kernel(model, y, sol, sp, method=method)
            Dp = float(sol["deviance"]) + float(kernel.bSb)
            det_term = float(kernel.K)
        else:
            penalty_quad = float(sol["penalty_quadratic"] or 0.0)
            Dp = float(sol["deviance"]) + penalty_quad
            det_term = (
                _pirls_tensor_coefficient_space_logdet_term(model, y, sol, sp, method)
                if model._has_tensor_terms()
                else _pirls_laplace_logdet_term(model, sol, sp, method)
            )
    except np.linalg.LinAlgError:
        return np.inf
    if not np.isfinite(det_term):
        return np.inf

    saturated_loglik, _, _ = _gamma_profile_objective_curvature(
        model, y, Dp, phi, mp, method=method
    )
    objective = Dp / (2.0 * phi * gamma) - saturated_loglik / gamma + det_term
    if method == "REML":
        objective -= 0.5 * mp * (np.log(2.0 * np.pi * phi) - np.log(gamma))
    return float(objective)
