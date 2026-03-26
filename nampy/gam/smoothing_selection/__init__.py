"""
Smoothing-parameter criteria and outer optimization.

Layout
------
``criteria/`` — REML/GCV/PIRLS scores and derivatives (see ``criteria.__doc__``).
``optimize/`` — L-BFGS-B / Newton / P-IRLS outer Newton (see ``optimize.__doc__``).
``reparam.py`` — basis/penalty reparameterization for ML/REML structure.
"""

from .criteria import (
    gcv_score_gaussian,
    criterion_gcv_gaussian,
    criterion_ml_reml_exact,
    criterion_ml_reml,
    criterion_gcv_pirls,
    criterion_ubre_pirls,
    criterion_gradient,
    criterion_hessian,
    resolve_ml_reml_scoring_backend,
    criterion_value,
)
from .optimize import (
    supports_smoothing_method,
    resolve_smoothing_method,
    n_free_smoothing_params,
    expand_smoothing_params_from_log,
    optimize_smoothing_params,
)
from .reparam import (
    reparameterize_smooth,
    can_use_simple_ml_reml_structure,
    can_use_exact_gaussian_ml_reml,
    build_penalty_reparameterized_system,
    build_gaussian_reparameterized_system,
)

__all__ = [
    "gcv_score_gaussian",
    "criterion_gcv_gaussian",
    "criterion_ml_reml_exact",
    "criterion_ml_reml",
    "criterion_gcv_pirls",
    "criterion_ubre_pirls",
    "criterion_gradient",
    "criterion_hessian",
    "resolve_ml_reml_scoring_backend",
    "criterion_value",
    "supports_smoothing_method",
    "resolve_smoothing_method",
    "n_free_smoothing_params",
    "expand_smoothing_params_from_log",
    "optimize_smoothing_params",
    "reparameterize_smooth",
    "can_use_simple_ml_reml_structure",
    "can_use_exact_gaussian_ml_reml",
    "build_penalty_reparameterized_system",
    "build_gaussian_reparameterized_system",
]
