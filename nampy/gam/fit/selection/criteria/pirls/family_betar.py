"""Beta-regression PIRLS criteria and joint log-theta derivatives."""

from contextlib import contextmanager

import numpy as np

from .....families.family_base import JointOuterStrategy
from .....model_state import _coef_column_offset
from ....backends import solve_pirls_given_smoothing
from ....smoothing_params import expand_smoothing_params_from_log
from ...reparam import _static_penalty_null_dim
from .common import _prior_weights
from .derivatives import (
    _MGCV_GAM_FIT4_RANK_TOL,
    _free_smoothing_mask,
    _gdi1_kernel,
    gdi2_theta_joint_kernel,
)


def is_joint_betar_theta_model(model) -> bool:
    return (
        getattr(model.family, "joint_outer_strategy", JointOuterStrategy.NONE)
        is JointOuterStrategy.BETAR_THETA
        and bool(getattr(model.family, "estimate_theta", False))
    )


@contextmanager
def _temporary_log_theta(model, log_theta):
    previous = float(model.family.getTheta(False))
    model.family.putTheta(float(log_theta))
    try:
        yield
    finally:
        model.family.putTheta(previous)


def _betar_score_from_kernel(model, y, sol, kernel, method):
    """Evaluate ``gam.fit4``'s fixed-scale betar ML/REML expression.

    ``mgcv::betar()$ls`` is identically zero.  Therefore the saturated
    likelihood is deliberately not subtracted here; ``dev.resids`` already
    represents ``-2 log likelihood`` for this family.
    """
    gamma = float(model.score_gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf
    # ``irls_core`` retains mgcv's postprocessed reporting deviance for
    # betar. ``gam.fit4`` optimizes the raw ``dev.resids`` sum because
    # ``betar$ls`` is identically zero.
    raw_deviance = model.family.deviance(
        np.asarray(y, dtype=np.float64),
        np.asarray(sol["mu"], dtype=np.float64),
        weights=_prior_weights(model, y),
    )
    penalty_deviance = float(raw_deviance) + float(kernel.bSb)
    score = penalty_deviance / (2.0 * gamma) + float(kernel.K)
    if str(method).upper() == "REML":
        mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
        score -= 0.5 * mp * (np.log(2.0 * np.pi) - np.log(gamma))
    return float(score)


def criterion_ml_reml_pirls_betar(model, y, log_sp, method):
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    kernel = _gdi1_kernel(
        model,
        y,
        sol,
        sp,
        method=str(method).upper(),
        rank_tol=_MGCV_GAM_FIT4_RANK_TOL,
    )
    return _betar_score_from_kernel(model, y, sol, kernel, method)


def criterion_ml_reml_pirls_frozen_betar(model, y, log_sp, method):
    return criterion_ml_reml_pirls_betar(model, y, log_sp, method)


def _betar_joint_kernel_state(model, y, log_sp, log_theta, method):
    with _temporary_log_theta(model, log_theta):
        sp = expand_smoothing_params_from_log(model, log_sp)
        sol = solve_pirls_given_smoothing(model, y, sp)
        kernel = gdi2_theta_joint_kernel(
            model,
            y,
            sol,
            sp,
            method=str(method).upper(),
            need_hessian=True,
        )
        return sol, kernel


def criterion_ml_reml_pirls_betar_joint(model, y, log_sp, log_theta, method):
    with _temporary_log_theta(model, log_theta):
        sol, kernel = _betar_joint_kernel_state(model, y, log_sp, log_theta, method)
        return _betar_score_from_kernel(model, y, sol, kernel.gdi1, method)


def criterion_gradient_ml_reml_pirls_betar_joint(
    model, y, log_sp, log_theta, method
):
    with _temporary_log_theta(model, log_theta):
        _sol, kernel = _betar_joint_kernel_state(model, y, log_sp, log_theta, method)
        gamma = float(model.score_gamma)
        full = np.asarray(kernel.Dp1, dtype=np.float64) / (2.0 * gamma)
        full += np.asarray(kernel.K1_full, dtype=np.float64)
        free_mask = _free_smoothing_mask(model)
        # gdi2 stores [log(theta), log(sp...)]; the optimizer uses that order.
        return np.concatenate((full[0:1], full[1:][free_mask]))


def criterion_hessian_ml_reml_pirls_betar_joint(
    model, y, log_sp, log_theta, method
):
    with _temporary_log_theta(model, log_theta):
        _sol, kernel = _betar_joint_kernel_state(model, y, log_sp, log_theta, method)
        gamma = float(model.score_gamma)
        full = np.asarray(kernel.Dp2, dtype=np.float64) / (2.0 * gamma)
        full += np.asarray(kernel.K2_full, dtype=np.float64)
        free_mask = _free_smoothing_mask(model)
        keep = np.concatenate((np.array([0], dtype=np.int64), 1 + np.flatnonzero(free_mask)))
        return np.asarray(full[np.ix_(keep, keep)], dtype=np.float64)
