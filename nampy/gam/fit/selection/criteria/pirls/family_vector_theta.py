"""Fixed-scale vector-theta criteria from mgcv gam.fit4 (scat and ziP)."""

from contextlib import contextmanager

import numpy as np

from .....model_state import _fit_workspace
from ....backends import solve_pirls_given_smoothing
from ....smoothing_params import expand_smoothing_params_from_log
from ...reparam import _coef_column_offset, _static_penalty_null_dim
from .common import _prior_weights
from .derivatives import (
    _MGCV_GAM_FIT4_RANK_TOL,
    _free_smoothing_mask,
    _gdi1_kernel,
    gdi2_theta_joint_kernel,
)


@contextmanager
def _temporary_theta(model, theta):
    previous = np.asarray(model.family.getTheta(False), dtype=float).copy()
    model.family.putTheta(theta)
    try:
        yield
    finally:
        model.family.putTheta(previous)


def _saturation(model, y):
    return model.family.ls(
        y, _prior_weights(model, y), model.family.getTheta(False), float(model.family.known_scale)
    )


def _score(model, y, sol, kernel, saturation, method):
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        return np.inf
    scale = float(model.family.known_scale)
    # gam.fit4 optimizes raw dev.resids; ziP reporting deviance subtracts a
    # theta-dependent saturated likelihood only after fitting.
    deviance = model.family.deviance(y, sol.fit_result.mu, weights=_prior_weights(model, y))
    score = (
        (float(deviance) + float(kernel.bSb)) / (2 * scale) - float(saturation["ls"])
    ) / gamma + float(kernel.K)
    if str(method).upper() in {"REML", "LAML"}:
        mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
        score -= mp / 2 * (np.log(2 * np.pi * scale) - np.log(gamma))
    return float(score)


def criterion_ml_reml_pirls_vector_theta(model, y, log_sp, method):
    method = "REML" if str(method).upper() == "LAML" else str(method).upper()
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(
        model, y, sp, scale_reference=float(model.family.known_scale)
    )
    _fit_workspace(model).pirls_outer_scale_est = float(sol.fit_result.scale)
    kernel = _gdi1_kernel(
        model, y, sol, sp, method=method, rank_tol=_MGCV_GAM_FIT4_RANK_TOL
    )
    return _score(model, y, sol, kernel, _saturation(model, y), method)


def _joint(model, y, log_sp, method):
    method = "REML" if str(method).upper() == "LAML" else str(method).upper()
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(
        model, y, sp, scale_reference=float(model.family.known_scale)
    )
    _fit_workspace(model).pirls_outer_scale_est = float(sol.fit_result.scale)
    kernel = gdi2_theta_joint_kernel(
        model, y, sol, sp, method=method, need_hessian=True
    )
    free = np.flatnonzero(_free_smoothing_mask(model))
    # bfgs uses coefficient sensitivity curvature on smoothing coordinates,
    # not the outer criterion Hessian and not ancillary theta coordinates.
    _fit_workspace(model).pirls_reml_derivative_kernel_state = {
        "dVkk": np.diag(np.asarray(kernel.gdi1.dVkk))[free]
    }
    return sol, kernel, _saturation(model, y)


def criterion_ml_reml_pirls_vector_theta_joint(model, y, log_sp, log_theta, method):
    with _temporary_theta(model, log_theta):
        sol, kernel, saturation = _joint(model, y, log_sp, method)
        return _score(model, y, sol, kernel.gdi1, saturation, method)


def criterion_gradient_ml_reml_pirls_vector_theta_joint(model, y, log_sp, log_theta, method):
    with _temporary_theta(model, log_theta):
        _, kernel, saturation = _joint(model, y, log_sp, method)
        gamma = float(model.score_gamma)
        gradient = np.asarray(kernel.Dp1, dtype=float) / (2 * gamma * float(model.family.known_scale))
        gradient += np.asarray(kernel.K1_full, dtype=float)
        ntheta = np.asarray(log_theta).size
        # gam.fit4.r subtracts theta derivatives of the saturated likelihood.
        gradient[:ntheta] -= np.asarray(saturation["lsth1"], dtype=float) / gamma
        return np.concatenate(
            (gradient[:ntheta], gradient[ntheta:][_free_smoothing_mask(model)])
        )


def criterion_hessian_ml_reml_pirls_vector_theta_joint(model, y, log_sp, log_theta, method):
    with _temporary_theta(model, log_theta):
        _, kernel, saturation = _joint(model, y, log_sp, method)
        gamma = float(model.score_gamma)
        hessian = np.asarray(kernel.Dp2, dtype=float) / (2 * gamma * float(model.family.known_scale))
        hessian += np.asarray(kernel.K2_full, dtype=float)
        ntheta = np.asarray(log_theta).size
        hessian[:ntheta, :ntheta] -= (
            np.asarray(saturation["lsth2"], dtype=float) / gamma
        )
        keep = np.concatenate(
            (np.arange(ntheta), ntheta + np.flatnonzero(_free_smoothing_mask(model)))
        )
        return hessian[np.ix_(keep, keep)]
