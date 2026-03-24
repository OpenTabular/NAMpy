"""
Smoothing-parameter selection criteria (REML/GCV/PIRLS).

Submodules: ``penalty``, ``laplace``, ``gaussian``, ``gaussian_dyn``, ``gaussian_reml_algebra``,
``pirls``, ``ml_reml``, ``pirls_reml_derivative_blocks``, ``pirls_deriv``, ``dispatch``.
"""

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
    criterion_ml_reml_gaussian_exact_joint,
    gcv_score_gaussian,
)
from .gaussian_dyn import (
    _gaussian_dynamic_reml_derivative_terms,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_joint,
)
from .gaussian_grad import criterion_gradient_ml_reml_exact
from .laplace import _penalty_derivative_matrices
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
from .ml_reml import criterion_ml_reml, resolve_ml_reml_scoring_backend
from .penalty import _static_penalty_null_dim
from .pirls import (
    criterion_gcv_pirls,
    criterion_ml_reml_pirls,
    criterion_ubre_pirls,
)
from .pirls_deriv import (
    criterion_gradient_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_exact,
)

__all__ = [
    "_gaussian_dynamic_reml_derivative_terms",
    "_penalty_derivative_matrices",
    "_static_penalty_null_dim",
    "criterion_gcv_gaussian",
    "criterion_gcv_pirls",
    "criterion_gradient",
    "criterion_gradient_ml_reml_gaussian_dynamic_joint",
    "criterion_gradient_ml_reml_exact",
    "criterion_gradient_ml_reml_pirls_exact",
    "criterion_gradient_numerical",
    "criterion_hessian",
    "criterion_hessian_ml_reml_pirls_exact",
    "criterion_hessian_numerical",
    "criterion_infinite_sp_signal",
    "criterion_ml_reml",
    "criterion_ml_reml_exact",
    "criterion_ml_reml_exact_dynamic",
    "criterion_ml_reml_gaussian_dynamic_joint",
    "criterion_ml_reml_gaussian_exact_joint",
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
