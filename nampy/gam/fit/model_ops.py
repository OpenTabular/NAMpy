"""Compatibility facade for fit helpers.

Legacy imports still come here. Real implementations now live in smaller fit
modules grouped by concern.
"""

from .capabilities import (
    can_use_exact_gaussian_ml_reml,
    can_use_simple_ml_reml_structure,
    n_free_smoothing_params,
    needs_exact_gaussian_reparameterization,
    raise_ml_reml_backend_error,
    resolve_ml_reml_scoring_backend,
    resolve_smoothing_method,
    supports_smoothing_method,
    uses_closed_form_solver,
    uses_pirls_solver,
)
from .criterion_ops import criterion_gradient, criterion_hessian, criterion_value
from .design_ops import (
    build_gaussian_reparameterized_system,
    build_penalty_reparameterized_system,
    compile_designs,
)
from .penalty_ops import assemble_penalty_matrix, one_penalty_per_term_matrices
from .result_ops import (
    build_fit_result,
    build_gam_result,
    copy_fit_result,
    sync_gam_result,
)
from .smoothing_params import (
    expand_smoothing_params_from_log,
    optimize_smoothing_params,
    resolve_min_sp,
    resolve_smoothing_params,
)
from .solve_ops import solve_gaussian_given_smoothing, solve_pirls_given_smoothing

__all__ = [
    "uses_closed_form_solver",
    "uses_pirls_solver",
    "can_use_exact_gaussian_ml_reml",
    "can_use_simple_ml_reml_structure",
    "needs_exact_gaussian_reparameterization",
    "resolve_ml_reml_scoring_backend",
    "raise_ml_reml_backend_error",
    "supports_smoothing_method",
    "resolve_smoothing_method",
    "resolve_min_sp",
    "resolve_smoothing_params",
    "n_free_smoothing_params",
    "expand_smoothing_params_from_log",
    "compile_designs",
    "one_penalty_per_term_matrices",
    "assemble_penalty_matrix",
    "build_gaussian_reparameterized_system",
    "build_penalty_reparameterized_system",
    "solve_gaussian_given_smoothing",
    "solve_pirls_given_smoothing",
    "criterion_value",
    "criterion_gradient",
    "criterion_hessian",
    "optimize_smoothing_params",
    "build_fit_result",
    "copy_fit_result",
    "build_gam_result",
    "sync_gam_result",
]
