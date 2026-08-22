"""Gaussian-family PIRLS joint ML/REML derivative kernels."""

import numpy as np

from .....model_state import _coef_column_offset, _fit_workspace
from ....backends import solve_pirls_given_smoothing
from ....smoothing_params import expand_smoothing_params_from_log
from ...reparam import _static_penalty_null_dim
from .common import _prior_weights
from .derivatives import (
    _MGCV_GAM_FIT3_RANK_TOL,
    _free_smoothing_mask,
    _gdi1_kernel,
    _GDI2Kernel,
)


def _gaussian_joint_kernel_state(model, y, log_sp, method, *, phi):
    """Build ``gam.fit3`` joint smoothing/scale derivative state."""
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
    _fit_workspace(model).pirls_reml_gaussian_state = state
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


def gdi2_gaussian_joint_kernel(model, y, sol, sp, *, method, need_hessian):
    """Gaussian profiled-scale branch of the joint ``gdi2`` staging."""
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


def criterion_ml_reml_pirls_gaussian_joint(model, y, log_sp, log_phi, method):
    """Joint Gaussian PIRLS ML/REML criterion from ``mgcv::gam.fit3``."""
    method = str(method).upper()
    if str(getattr(model.family, "name", "")).lower() != "gaussian":
        raise NotImplementedError(
            "Joint PIRLS Gaussian outer objective is implemented only for "
            "family='gaussian'."
        )
    if bool(getattr(model.family, "supports_closed_form_solve", False)):
        raise NotImplementedError(
            "The PIRLS Gaussian joint objective is only for noncanonical links."
        )

    phi = float(np.exp(float(log_phi)))
    gamma = float(model.score_gamma)
    if not np.isfinite(phi) or phi <= 0.0 or not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf

    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp, scale_reference=phi)
    try:
        kernel = _gdi1_kernel(model, y, sol, sp, method=method)
    except np.linalg.LinAlgError:
        return np.inf

    y_arr = np.asarray(y, dtype=np.float64)
    weights = _prior_weights(model, y_arr)
    ls = np.asarray(model.family.ls(y_arr, weights, len(y_arr), phi), dtype=np.float64)
    nobs = float(len(y_arr))
    n_true = getattr(model, "n_true_", None)
    if n_true is not None:
        n_true = float(n_true)
        if np.isfinite(n_true) and n_true > 0.0 and nobs > 0.0:
            ls *= n_true / nobs

    dp = float(sol["deviance"]) + float(kernel.bSb)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    reml_ind = 1.0 if method == "REML" else 0.0
    objective = (dp / (2.0 * phi) - float(ls[0])) / gamma + float(kernel.K)
    objective -= reml_ind * 0.5 * mp * (
        np.log(2.0 * np.pi * phi) - np.log(gamma)
    )
    return float(objective)
