"""
Smoothing-parameter criteria and outer optimization.

Layout
------
``criteria/`` — REML/GCV/PIRLS scores and derivatives (see ``criteria.__doc__``).
``optimize/`` — L-BFGS-B / Newton / P-IRLS outer Newton (see ``optimize.__doc__``).
``reparam.py`` — basis/penalty reparameterization for ML/REML structure.
"""

from .criteria import (
    criterion_gcv_gaussian,
    criterion_gcv_pirls,
    criterion_gradient,
    criterion_hessian,
    criterion_ml_reml,
    criterion_ml_reml_exact,
    criterion_ubre_pirls,
    criterion_value,
    gcv_score_gaussian,
    resolve_ml_reml_scoring_backend,
)
from .optimize import (
    expand_smoothing_params_from_log,
    n_free_smoothing_params,
    optimize_smoothing_params,
    resolve_smoothing_method,
    supports_smoothing_method,
)
from .postfit import gam_vcomp, one_se_rule, sp_vcov
from .reparam import (
    SlBlock,
    build_gaussian_reparameterized_system,
    build_penalty_reparameterized_system,
    can_use_exact_gaussian_ml_reml,
    can_use_simple_ml_reml_structure,
    reparameterize_smooth,
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
    "sp_vcov",
    "gam_vcomp",
    "one_se_rule",
    "SlBlock",
    "reparameterize_smooth",
    "can_use_simple_ml_reml_structure",
    "can_use_exact_gaussian_ml_reml",
    "build_penalty_reparameterized_system",
    "build_gaussian_reparameterized_system",
]
