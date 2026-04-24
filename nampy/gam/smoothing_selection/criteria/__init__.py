"""
Smoothing-parameter selection criteria (REML/GCV/PIRLS).

Submodules: ``penalty``, ``laplace``, ``gaussian``, ``gaussian_dyn``, ``gaussian_reml_algebra``,
``pirls``, ``ml_reml``, ``pirls_reml_derivative_blocks``, ``pirls_deriv``, ``dispatch``.
"""

from ..reparam import _stable_penalty_logdet_derivatives, _static_penalty_null_dim
from .dispatch import (
    criterion_gradient,
    criterion_gradient_numerical,
    criterion_hessian,
    criterion_hessian_numerical,
    criterion_infinite_sp_signal,
    criterion_value,
)
from .gaussian import (
    criterion_gcv_gaussian,
    criterion_ml_reml_exact,
    criterion_ml_reml_exact_dynamic,
    gcv_score_gaussian,
)
from .gaussian_dyn import (
    _gaussian_dynamic_reml_derivative_terms,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_hessian_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_profiled,
)
from .gaussian_reml_algebra import (
    deviance_method_scale_estimate,
    gaussian_reml_laplace_score,
    gaussian_reml_saturation_terms_wrt_variance,
    gaussian_reml_score_from_saturated_part,
    gaussian_reml_weighted_degrees_and_log_weight_term,
    gaussian_weighted_residual_sum_squares,
    pearson_method_scale_estimate,
    prior_weights_diagonal_from_fit,
    profiled_gaussian_reml_variance,
    quadratic_form_penalty,
)
from .laplace import _penalty_derivative_matrices
from .ml_reml import criterion_ml_reml, resolve_ml_reml_scoring_backend
from .ncv import (
    criterion_gradient_ncv,
    criterion_gradient_ncv_negbin_joint,
    criterion_ncv,
    criterion_ncv_negbin_joint,
)
from .pirls import (
    _pirls_ml_reml_objective_from_solution,
    criterion_gcv_pirls,
    criterion_ml_reml_pirls,
    criterion_ml_reml_pirls_gamma_joint,
    criterion_ml_reml_pirls_negbin_joint,
    criterion_ubre_pirls,
)
from .pirls_deriv import (
    criterion_gradient_ml_reml_pirls_exact,
    criterion_gradient_ml_reml_pirls_gamma_joint,
    criterion_gradient_ml_reml_pirls_negbin_joint,
    criterion_hessian_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_gamma_joint,
    criterion_hessian_ml_reml_pirls_negbin_joint,
)

__all__ = [
    "_gaussian_dynamic_reml_derivative_terms",
    "_pirls_ml_reml_objective_from_solution",
    "_penalty_derivative_matrices",
    "_stable_penalty_logdet_derivatives",
    "_static_penalty_null_dim",
    "criterion_gcv_gaussian",
    "criterion_gcv_pirls",
    "criterion_gradient",
    "criterion_gradient_ml_reml_gaussian_dynamic_joint",
    "criterion_gradient_ncv",
    "criterion_gradient_ncv_negbin_joint",
    "criterion_gradient_ml_reml_pirls_gamma_joint",
    "criterion_gradient_ml_reml_pirls_negbin_joint",
    "criterion_gradient_ml_reml_pirls_exact",
    "criterion_gradient_numerical",
    "criterion_hessian",
    "criterion_hessian_ml_reml_gaussian_dynamic_joint",
    "criterion_hessian_ml_reml_pirls_gamma_joint",
    "criterion_hessian_ml_reml_pirls_negbin_joint",
    "criterion_hessian_ml_reml_pirls_exact",
    "criterion_hessian_numerical",
    "criterion_infinite_sp_signal",
    "criterion_ml_reml",
    "criterion_ml_reml_exact",
    "criterion_ml_reml_exact_dynamic",
    "criterion_ml_reml_gaussian_dynamic_joint",
    "criterion_ml_reml_gaussian_dynamic_profiled",
    "criterion_ncv",
    "criterion_ncv_negbin_joint",
    "criterion_ml_reml_pirls_gamma_joint",
    "criterion_ml_reml_pirls_negbin_joint",
    "criterion_ml_reml_pirls",
    "criterion_ubre_pirls",
    "criterion_value",
    "gcv_score_gaussian",
    "deviance_method_scale_estimate",
    "gaussian_reml_laplace_score",
    "gaussian_reml_saturation_terms_wrt_variance",
    "gaussian_reml_score_from_saturated_part",
    "gaussian_reml_weighted_degrees_and_log_weight_term",
    "gaussian_weighted_residual_sum_squares",
    "pearson_method_scale_estimate",
    "prior_weights_diagonal_from_fit",
    "profiled_gaussian_reml_variance",
    "quadratic_form_penalty",
    "resolve_ml_reml_scoring_backend",
]
