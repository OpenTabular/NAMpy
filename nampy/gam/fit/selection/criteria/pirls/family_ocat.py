"""Ordered-categorical PIRLS criteria and joint cutpoint derivatives."""

from contextlib import contextmanager

import numpy as np

from .....families.family_base import JointOuterStrategy
from ....backends import solve_pirls_given_smoothing
from ....smoothing_params import expand_smoothing_params_from_log
from ...reparam import _coef_column_offset, _static_penalty_null_dim
from .derivatives import (
    _MGCV_GAM_FIT4_RANK_TOL,
    _free_smoothing_mask,
    _gdi1_kernel,
    gdi2_theta_joint_kernel,
)


def is_joint_ocat_theta_model(model) -> bool:
    return (
        getattr(model.family, "joint_outer_strategy", JointOuterStrategy.NONE)
        is JointOuterStrategy.OCAT_THETA
        and bool(getattr(model.family, "estimate_theta", False))
    )


@contextmanager
def _temporary_log_theta(model, log_theta):
    previous = np.asarray(model.family.getTheta(False), dtype=np.float64).copy()
    model.family.putTheta(log_theta)
    try:
        yield
    finally:
        model.family.putTheta(previous)


def _ocat_score_from_kernel(model, y, sol, kernel, method):
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf
    score = (
        float(sol["deviance"]) + float(kernel.bSb)
    ) / (2.0 * gamma) + float(kernel.K)
    if str(method).upper() == "REML":
        mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
        score -= 0.5 * mp * (np.log(2.0 * np.pi) - np.log(gamma))
    return float(score)


def criterion_ml_reml_pirls_ocat(model, y, log_sp, method):
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    kernel = _gdi1_kernel(
        model, y, sol, sp, method=str(method).upper(), rank_tol=_MGCV_GAM_FIT4_RANK_TOL
    )
    return _ocat_score_from_kernel(model, y, sol, kernel, method)


def criterion_ml_reml_pirls_frozen_ocat(model, y, log_sp, method):
    return criterion_ml_reml_pirls_ocat(model, y, log_sp, method)


def _ocat_joint_kernel_state(model, y, log_sp, log_theta, method):
    with _temporary_log_theta(model, log_theta):
        sp = expand_smoothing_params_from_log(model, log_sp)
        sol = solve_pirls_given_smoothing(model, y, sp)
        kernel = gdi2_theta_joint_kernel(
            model, y, sol, sp, method=str(method).upper(), need_hessian=True
        )
        return sol, kernel


def criterion_ml_reml_pirls_ocat_joint(model, y, log_sp, log_theta, method):
    sol, kernel = _ocat_joint_kernel_state(model, y, log_sp, log_theta, method)
    return _ocat_score_from_kernel(model, y, sol, kernel.gdi1, method)


def criterion_gradient_ml_reml_pirls_ocat_joint(
    model, y, log_sp, log_theta, method
):
    _sol, kernel = _ocat_joint_kernel_state(model, y, log_sp, log_theta, method)
    gamma = float(model.score_gamma)
    full = np.asarray(kernel.Dp1, dtype=np.float64) / (2.0 * gamma)
    full += np.asarray(kernel.K1_full, dtype=np.float64)
    free_mask = _free_smoothing_mask(model)
    ntheta = int(np.asarray(log_theta, dtype=np.float64).size)
    return np.concatenate((full[:ntheta], full[ntheta:][free_mask]))


def criterion_hessian_ml_reml_pirls_ocat_joint(
    model, y, log_sp, log_theta, method
):
    _sol, kernel = _ocat_joint_kernel_state(model, y, log_sp, log_theta, method)
    gamma = float(model.score_gamma)
    full = np.asarray(kernel.Dp2, dtype=np.float64) / (2.0 * gamma)
    full += np.asarray(kernel.K2_full, dtype=np.float64)
    free_mask = _free_smoothing_mask(model)
    ntheta = int(np.asarray(log_theta, dtype=np.float64).size)
    keep = np.concatenate(
        (np.arange(ntheta, dtype=np.int64), ntheta + np.flatnonzero(free_mask))
    )
    return np.asarray(full[np.ix_(keep, keep)], dtype=np.float64)
