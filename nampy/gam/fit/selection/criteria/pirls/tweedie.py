"""Tweedie-specific PIRLS criteria and joint outer derivatives.

The implementation follows the ``mgcv::tw`` path in ``gam.fit4.r``.  The
generic PIRLS derivative algebra remains in :mod:`.derivatives`; this module
owns Tweedie's joint ``[theta, log(sp), log(scale)]`` parameterization.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .....families.family_base import JointOuterStrategy
from .....model_state import _coef_column_offset
from ....backends import solve_pirls_given_smoothing
from ....smoothing_params import expand_smoothing_params_from_log
from ...reparam import _static_penalty_null_dim
from .common import _prior_weights
from .derivatives import (
    _MGCV_GAM_FIT4_RANK_TOL,
    _family_ddeta_logtheta,
    _free_smoothing_mask,
    _gdi1_kernel,
    _gdi2_deviance_derivatives,
    _gdi2_ift2_state_theta,
    _gdi2_penalty_terms,
    _GDI2Kernel,
    _gdi_pk_setup,
)
from .reml_blocks import _penalty_quadratic_and_sp_derivatives

JOINT_OUTER_STRATEGY = JointOuterStrategy.TWEEDIE


@dataclass
class _TweedieJointState:
    """PIRLS and saturated-log-likelihood state for joint ``tw`` scoring."""

    kernel: _GDI2Kernel
    phi: float
    ls1: np.ndarray
    ls2: np.ndarray
    ntheta: int


def _n_true_scale_factor(model, y):
    """Return mgcv's ``n.true / n`` multiplier for Tweedie likelihood terms."""
    nobs = float(len(np.asarray(y)))
    n_true = getattr(model, "n_true_", None)
    if n_true is not None and np.isfinite(float(n_true)) and nobs > 0.0:
        return float(n_true) / nobs
    return 1.0


@contextmanager
def _temporary_tweedie_theta(family, log_theta):
    """Evaluate a joint criterion at ``log_theta`` and restore family state."""
    previous_theta = float(family.getTheta(False))
    if int(getattr(family, "n_theta", 0) or 0):
        family.putTheta(float(log_theta))
    try:
        yield
    finally:
        family.putTheta(previous_theta)


def _tweedie_joint_kernel_state(
    model, y, log_sp, log_theta, log_phi, method, *, need_hessian
):
    family = model.family
    phi = float(np.exp(float(log_phi)))
    if not np.isfinite(phi) or phi <= 0.0:
        raise ValueError("Tweedie log scale must map to finite positive scale.")

    with _temporary_tweedie_theta(family, log_theta):
        sp = expand_smoothing_params_from_log(model, log_sp)
        sol = solve_pirls_given_smoothing(
            model, y, sp, scale_reference=phi
        )
        gdi2 = _gdi2_tweedie_joint_kernel(
            model,
            y,
            sol,
            sp,
            method=str(method).upper(),
            need_hessian=need_hessian,
            phi=phi,
        )
        weights = _prior_weights(model, y)
        ls = family.ls(
            np.asarray(y, dtype=np.float64),
            weights, theta=float(family.getTheta(False)), scale=phi
        )
        ls1 = np.asarray(ls["lsth1"], dtype=np.float64).ravel()
        ls2 = np.asarray(ls["lsth2"], dtype=np.float64)
        n_true_factor = _n_true_scale_factor(model, y)
        return _TweedieJointState(
            kernel=gdi2,
            phi=phi,
            ls1=ls1 * n_true_factor,
            ls2=ls2 * n_true_factor,
            ntheta=int(getattr(family, "n_theta", 0) or 0),
        )


def criterion_ml_reml_pirls_tweedie_joint(
    model, y, log_sp, log_theta, log_phi, method
):
    """Evaluate the mgcv ``tw`` joint ML/REML/LAML criterion."""
    if getattr(model.family, "joint_outer_strategy", None) is not JOINT_OUTER_STRATEGY:
        raise NotImplementedError(
            "Joint Tweedie outer objective requires the Tweedie family strategy."
        )
    method = str(method).upper()
    phi = float(np.exp(float(log_phi)))
    gamma = float(model.score_gamma)
    if not np.isfinite(phi) or phi <= 0.0 or not np.isfinite(gamma) or gamma <= 0.0:
        return np.inf

    family = model.family
    with _temporary_tweedie_theta(family, log_theta):
        sp = expand_smoothing_params_from_log(model, log_sp)
        sol = solve_pirls_given_smoothing(model, y, sp, scale_reference=phi)
        try:
            kernel = _gdi1_kernel(model, y, sol, sp, method=method)
        except np.linalg.LinAlgError:
            return np.inf

        ls = family.ls(
            np.asarray(y, dtype=np.float64),
            _prior_weights(model, y),
            theta=float(family.getTheta(False)),
            scale=phi,
        )
        saturated_loglik = float(ls["ls"]) * _n_true_scale_factor(model, y)
        penalty_deviance = float(sol["deviance"]) + float(kernel.bSb)
        score = (penalty_deviance / (2.0 * phi) - saturated_loglik) / gamma
        score += float(kernel.K)
        if method in {"REML", "LAML"}:
            mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
            score -= 0.5 * mp * (np.log(2.0 * np.pi * phi) - np.log(gamma))
        return float(score)


def criterion_gradient_ml_reml_pirls_tweedie_joint(
    model, y, log_sp, log_theta, log_phi, method
):
    state = _tweedie_joint_kernel_state(
        model, y, log_sp, log_theta, log_phi, method, need_hessian=False
    )
    kernel = state.kernel
    gamma = float(model.score_gamma)
    ntheta = state.ntheta
    free_mask = _free_smoothing_mask(model)
    mp = float(_static_penalty_null_dim(model) + _coef_column_offset(model))
    gradient = np.asarray(kernel.Dp1, dtype=np.float64) / (
        2.0 * state.phi * gamma
    ) + np.asarray(kernel.K1_full, dtype=np.float64)
    gradient = gradient.copy()
    if ntheta:
        gradient[0] -= float(state.ls1[0]) / gamma
    grad_phi = (
        -float(kernel.Dp) / (2.0 * state.phi) - float(state.ls1[ntheta])
    ) / gamma
    if str(method).upper() == "REML":
        grad_phi -= 0.5 * mp
    return np.concatenate(
        [
            gradient[:ntheta],
            gradient[ntheta:][free_mask],
            np.array([grad_phi], dtype=np.float64),
        ]
    )


def criterion_hessian_ml_reml_pirls_tweedie_joint(
    model, y, log_sp, log_theta, log_phi, method
):
    state = _tweedie_joint_kernel_state(
        model, y, log_sp, log_theta, log_phi, method, need_hessian=True
    )
    kernel = state.kernel
    gamma = float(model.score_gamma)
    ntheta = state.ntheta
    free_mask = _free_smoothing_mask(model)
    phi = state.phi
    hessian = np.asarray(kernel.Dp2, dtype=np.float64) / (2.0 * phi * gamma)
    hessian += np.asarray(kernel.K2_full, dtype=np.float64)
    hessian = hessian.copy()
    if ntheta:
        hessian[0, 0] -= float(state.ls2[0, 0]) / gamma
    cross = -np.asarray(kernel.Dp1, dtype=np.float64) / (2.0 * phi * gamma)
    if ntheta:
        cross[0] -= float(state.ls2[0, 1]) / gamma
    hphi = float(kernel.Dp) / (2.0 * phi * gamma) - float(
        state.ls2[1, 1]
    ) / gamma

    full = np.zeros((hessian.shape[0] + 1, hessian.shape[1] + 1), dtype=np.float64)
    full[:-1, :-1] = hessian
    full[:-1, -1] = cross
    full[-1, :-1] = cross
    full[-1, -1] = hphi
    keep = np.concatenate(
        [
            np.arange(ntheta, dtype=np.int64),
            ntheta + np.flatnonzero(free_mask),
            np.array([hessian.shape[0]], dtype=np.int64),
        ]
    )
    return np.asarray(full[np.ix_(keep, keep)], dtype=np.float64)


def _gdi2_tweedie_joint_kernel(
    model, y, sol, sp, *, method, need_hessian, phi
):
    """Port the mgcv ``gdi2`` theta/sp block for ``tw``."""
    setup = _gdi_pk_setup(
        model,
        sol,
        sp,
        deriv=2,
        rank_tol=_MGCV_GAM_FIT4_RANK_TOL,
    )
    current = setup.current
    pk_state = setup.pk
    include_theta = bool(getattr(model.family, "n_theta", 0))
    ift = _gdi2_ift2_state_theta(
        model,
        y,
        sol,
        sp,
        current,
        pk_state,
        include_theta=include_theta,
    )
    dd = _family_ddeta_logtheta(
        model.family,
        y,
        np.asarray(sol["mu"], dtype=np.float64),
        _prior_weights(model, y),
        deriv=2,
    )
    D1, D2 = _gdi2_deviance_derivatives(
        current,
        ift,
        dd,
        include_theta=include_theta,
        need_hessian=need_hessian,
    )
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
    ls = model.family.ls(
        np.asarray(y, dtype=np.float64),
        _prior_weights(model, y),
        theta=float(model.family.getTheta(False)),
        scale=float(phi),
    )
    phi_curv = (
        float(Dp) / (2.0 * float(phi))
        - float(np.asarray(ls["lsth2"], dtype=np.float64)[1, 1])
        * _n_true_scale_factor(model, y)
    ) / float(model.score_gamma)
    return _GDI2Kernel(
        gdi1=gdi1,
        phi=float(phi),
        phi_curv=float(phi_curv),
        Dp=float(Dp),
        Dp1=Dp1,
        Dp2=Dp2,
        ift=ift,
        D1_full=np.asarray(D1, dtype=np.float64),
        D2_full=None if D2 is None else np.asarray(D2, dtype=np.float64),
        K1_full=np.asarray(K1, dtype=np.float64),
        K2_full=np.asarray(K2, dtype=np.float64),
        extra_name="log_theta" if include_theta else None,
        extra_value=(
            float(model.family.getTheta(False)) if include_theta else None
        ),
    )
