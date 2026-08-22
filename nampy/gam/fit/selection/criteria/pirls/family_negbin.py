"""Negative-binomial PIRLS joint ML/REML derivative kernels."""

from contextlib import contextmanager

import numpy as np

from .....families.family_base import JointOuterStrategy
from .....model_state import _fit_workspace
from ....backends import solve_pirls_given_smoothing
from ....smoothing_params import expand_smoothing_params_from_log
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


def _copy_state_vector(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).copy()


def is_joint_negbin_theta_model(model) -> bool:
    return (
        getattr(model.family, "joint_outer_strategy", JointOuterStrategy.NONE)
        is JointOuterStrategy.NEGBIN_THETA
        and bool(getattr(model.family, "estimate_theta", False))
    )


def current_joint_negbin_eval_state(model):
    result = getattr(model, "_optim_result", None)
    result_state = (
        None if result is None else getattr(result, "joint_negbin_state", None)
    )
    if isinstance(result_state, dict):
        return {
            "coef": _copy_state_vector(result_state.get("coef", None)),
            "eta": _copy_state_vector(result_state.get("eta", None)),
            "mu": _copy_state_vector(result_state.get("mu", None)),
            "theta": float(result_state.get("theta", model.family.getTheta(True))),
        }

    baseline = getattr(model, "_joint_negbin_fd_baseline_", None)
    if isinstance(baseline, dict):
        return {
            "coef": _copy_state_vector(baseline.get("coef", None)),
            "eta": _copy_state_vector(baseline.get("eta", None)),
            "mu": _copy_state_vector(baseline.get("mu", None)),
            "theta": float(baseline.get("theta", model.family.getTheta(True))),
        }

    workspace = _fit_workspace(model)
    return {
        "coef": _copy_state_vector(
            workspace.get("pirls_coef_start", workspace.get("pirls_last_coef", None))
        ),
        "eta": _copy_state_vector(
            workspace.get("pirls_eta_start", workspace.get("pirls_last_eta", None))
        ),
        "mu": _copy_state_vector(
            workspace.get("pirls_mu_start", workspace.get("pirls_last_mu", None))
        ),
        "theta": float(model.family.getTheta(True)),
    }


@contextmanager
def frozen_joint_negbin_eval_state(model, baseline_state=None):
    workspace = _fit_workspace(model)
    prev = {
        "eval_coef": _copy_state_vector(workspace.get("pirls_eval_start", None)),
        "eval_eta": _copy_state_vector(workspace.get("pirls_eval_eta_start", None)),
        "eval_mu": _copy_state_vector(workspace.get("pirls_eval_mu_start", None)),
        "lock": bool(workspace.get("pirls_lock_start", False)),
        "coef": _copy_state_vector(workspace.get("pirls_coef_start", None)),
        "eta": _copy_state_vector(workspace.get("pirls_eta_start", None)),
        "mu": _copy_state_vector(workspace.get("pirls_mu_start", None)),
        "last_coef": _copy_state_vector(workspace.get("pirls_last_coef", None)),
        "last_eta": _copy_state_vector(workspace.get("pirls_last_eta", None)),
        "last_mu": _copy_state_vector(workspace.get("pirls_last_mu", None)),
        "theta": float(model.family.getTheta(True)),
    }
    state = (
        current_joint_negbin_eval_state(model)
        if baseline_state is None
        else {
            "coef": _copy_state_vector(baseline_state.get("coef", None)),
            "eta": _copy_state_vector(baseline_state.get("eta", None)),
            "mu": _copy_state_vector(baseline_state.get("mu", None)),
            "theta": float(baseline_state.get("theta", model.family.getTheta(True))),
        }
    )
    workspace.pirls_eval_start = _copy_state_vector(state.get("coef", None))
    workspace.pirls_eval_eta_start = _copy_state_vector(state.get("eta", None))
    workspace.pirls_eval_mu_start = _copy_state_vector(state.get("mu", None))
    workspace.pirls_lock_start = True
    model.family.putTheta(float(np.log(state["theta"])))
    try:
        yield state
    finally:
        workspace.pirls_eval_start = prev["eval_coef"]
        workspace.pirls_eval_eta_start = prev["eval_eta"]
        workspace.pirls_eval_mu_start = prev["eval_mu"]
        workspace.pirls_lock_start = prev["lock"]
        workspace.pirls_coef_start = prev["coef"]
        workspace.pirls_eta_start = prev["eta"]
        workspace.pirls_mu_start = prev["mu"]
        workspace.pirls_last_coef = prev["last_coef"]
        workspace.pirls_last_eta = prev["last_eta"]
        workspace.pirls_last_mu = prev["last_mu"]
        model.family.putTheta(float(np.log(prev["theta"])))


def criterion_ml_reml_pirls_frozen_negbin(model, y, log_sp, method, baseline_state=None):
    from .value import criterion_ml_reml_pirls

    with frozen_joint_negbin_eval_state(model, baseline_state=baseline_state):
        return criterion_ml_reml_pirls(model, y, log_sp, method)


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
        gdi2 = gdi2_negbin_joint_kernel(
            model, y, sol, sp, method=str(method).upper(), need_hessian=True
        )
        # Mirror gam.fit4.r L730: ls <- family$ls(y,weights,theta,scale).
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
        _fit_workspace(model).pirls_reml_negbin_state = state
        return state
    finally:
        model.family.putTheta(prev_log_theta)
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)


def criterion_gradient_ml_reml_pirls_negbin_joint(
    model, y, log_sp, log_theta, method
):
    state = _negbin_joint_kernel_state(model, y, log_sp, log_theta, method)
    gamma = float(model.score_gamma)
    free_mask = _free_smoothing_mask(model)
    grad_full = np.asarray(state["Dp1"], dtype=np.float64) / (
        2.0 * float(state["scale_est"]) * gamma
    ) + np.asarray(state["K1"], dtype=np.float64)
    # Mirror gam.fit4.r L744: only theta component gets ls correction.
    grad_full = grad_full.copy()
    grad_full[0] -= float(state["lsth1"]) / gamma
    return np.concatenate(
        [
            np.asarray(grad_full[1:][free_mask], dtype=np.float64),
            np.array([float(grad_full[0])], dtype=np.float64),
        ]
    )


def criterion_hessian_ml_reml_pirls_negbin_joint(
    model, y, log_sp, log_theta, method
):
    state = _negbin_joint_kernel_state(model, y, log_sp, log_theta, method)
    gamma = float(model.score_gamma)
    free_mask = _free_smoothing_mask(model)
    H_full = np.asarray(state["Dp2"], dtype=np.float64) / (
        2.0 * float(state["scale_est"]) * gamma
    ) + np.asarray(state["K2"], dtype=np.float64)
    # Mirror gam.fit4.r L746-748: only theta-theta block gets ls2 correction.
    H_full = H_full.copy()
    H_full[0, 0] -= float(state["lsth2"]) / gamma
    sp_idx = 1 + np.flatnonzero(free_mask)
    keep = np.concatenate([sp_idx, np.array([0], dtype=np.int64)])
    return np.asarray(H_full[np.ix_(keep, keep)], dtype=np.float64)


def gdi2_negbin_joint_kernel(model, y, sol, sp, *, method, need_hessian):
    """Port of ``mgcv::gdi2()`` for negative-binomial ``log(theta)``."""
    setup = _gdi_pk_setup(
        model,
        sol,
        sp,
        deriv=2,
        rank_tol=_MGCV_GAM_FIT4_RANK_TOL,
    )
    current = setup.current
    pk_state = setup.pk
    ift = _gdi2_ift2_state_theta(model, y, sol, sp, current, pk_state)
    weights = _prior_weights(model, y)
    dd = _family_ddeta_logtheta(
        model.family,
        y,
        np.asarray(sol["mu"], dtype=np.float64),
        weights,
        deriv=2,
    )
    D1, D2 = _gdi2_deviance_derivatives(
        current, ift, dd, include_theta=True, need_hessian=need_hessian
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


def criterion_ml_reml_pirls_negbin_joint(model, y, log_sp, log_theta, method):
    """Joint ``(log_sp, log_theta)`` negative-binomial PIRLS objective."""
    from .value import criterion_ml_reml_pirls

    if str(getattr(model.family, "name", "")).lower() != "negbin":
        raise NotImplementedError(
            "Joint PIRLS NegBin outer objective is implemented only for family='negbin'."
        )
    theta = float(np.exp(float(log_theta)))
    if not np.isfinite(theta) or theta <= 0.0:
        return np.inf
    prev_log_theta = float(model.family.getTheta(False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    try:
        model.family.putTheta(float(log_theta))
        model.family._disable_theta_efs = True
        return criterion_ml_reml_pirls(model, y, log_sp, method)
    finally:
        model.family.putTheta(prev_log_theta)
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)
