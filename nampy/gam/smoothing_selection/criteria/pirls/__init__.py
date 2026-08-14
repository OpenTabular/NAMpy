"""PIRLS smoothing criteria and exact derivative kernels."""

import numpy as np

from .derivatives import (
    _gdi1_ift1_state,
    _gdi1_kernel,
    _gdi2_joint_kernel,
    _gdi_pk_setup,
    _negbin_ddeta_logtheta,
    _prior_weights,
    criterion_gradient_gcv_ubre_pirls_exact,
    criterion_gradient_ml_reml_pirls_exact,
    criterion_gradient_ml_reml_pirls_gamma_joint,
    criterion_gradient_ml_reml_pirls_gaussian_joint,
    criterion_gradient_ml_reml_pirls_negbin_joint,
    criterion_hessian_gcv_ubre_pirls_exact,
    criterion_hessian_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_gamma_joint,
    criterion_hessian_ml_reml_pirls_gaussian_joint,
    criterion_hessian_ml_reml_pirls_negbin_joint,
)
from .value import (
    _current_joint_negbin_eval_state,
    _gamma_profile_objective_curvature,
    _is_joint_negbin_theta_model,
    _pirls_ml_reml_objective_from_solution,
    _solve_gamma_profile_scale,
    criterion_ml_reml_pirls,
    criterion_ml_reml_pirls_dynamic,
    criterion_ml_reml_pirls_frozen_negbin,
    criterion_ml_reml_pirls_gamma_joint,
    criterion_ml_reml_pirls_gaussian_joint,
    criterion_ml_reml_pirls_negbin_joint,
    expand_smoothing_params_from_log,
    solve_pirls_given_smoothing,
)


def criterion_gcv_pirls(model, y, log_sp):
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    n = model.n_samples_
    den = 1.0 - model.score_gamma * sol["trace_H"] / n
    if not np.isfinite(den) or den == 0.0:
        return np.inf
    return (sol["deviance"] / n) / (den**2)


def criterion_ubre_pirls(model, y, log_sp):
    sp = expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    scale = model.family.known_scale
    if scale is None:
        raise ValueError(
            f"UBRE/AIC requested for family={model.family.name!r}, "
            "but the family does not have known scale."
        )
    n = model.n_samples_
    edf = sol["trace_H"]
    return (sol["deviance"] / n) - scale + (2.0 * model.score_gamma * scale * edf / n)

__all__ = [
    "_current_joint_negbin_eval_state",
    "_gamma_profile_objective_curvature",
    "_gdi1_ift1_state",
    "_gdi1_kernel",
    "_gdi2_joint_kernel",
    "_gdi_pk_setup",
    "_is_joint_negbin_theta_model",
    "_negbin_ddeta_logtheta",
    "_pirls_ml_reml_objective_from_solution",
    "_prior_weights",
    "_solve_gamma_profile_scale",
    "criterion_gcv_pirls",
    "criterion_gradient_gcv_ubre_pirls_exact",
    "criterion_gradient_ml_reml_pirls_exact",
    "criterion_gradient_ml_reml_pirls_gaussian_joint",
    "criterion_gradient_ml_reml_pirls_gamma_joint",
    "criterion_gradient_ml_reml_pirls_negbin_joint",
    "criterion_hessian_gcv_ubre_pirls_exact",
    "criterion_hessian_ml_reml_pirls_exact",
    "criterion_hessian_ml_reml_pirls_gaussian_joint",
    "criterion_hessian_ml_reml_pirls_gamma_joint",
    "criterion_hessian_ml_reml_pirls_negbin_joint",
    "criterion_ml_reml_pirls",
    "criterion_ml_reml_pirls_dynamic",
    "criterion_ml_reml_pirls_frozen_negbin",
    "criterion_ml_reml_pirls_gaussian_joint",
    "criterion_ml_reml_pirls_gamma_joint",
    "criterion_ml_reml_pirls_negbin_joint",
    "criterion_ubre_pirls",
    "expand_smoothing_params_from_log",
    "solve_pirls_given_smoothing",
]
