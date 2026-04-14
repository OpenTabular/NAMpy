"""Smoothing-parameter objective and optimizer facade."""

from ..smoothing_selection import (
    criterion_gcv_gaussian,
    criterion_gcv_pirls,
    criterion_gradient,
    criterion_hessian,
    criterion_ml_reml,
    criterion_ml_reml_exact,
    criterion_ubre_pirls,
    criterion_value,
    gam_vcomp,
    gcv_score_gaussian,
    n_free_smoothing_params,
    one_se_rule,
    optimize_smoothing_params,
    resolve_ml_reml_scoring_backend,
    resolve_smoothing_method,
    sp_vcov,
    supports_smoothing_method,
)
from ..smoothing_selection.criteria import (
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_pirls,
)
from ..smoothing_selection.optimize import criterion_infinite_sp_signal
from ..smoothing_selection.postfit import optimizer_endpoint_diagnostics

__all__ = [
    "criterion_gradient",
    "criterion_hessian",
    "criterion_infinite_sp_signal",
    "criterion_gcv_gaussian",
    "criterion_gcv_pirls",
    "criterion_ubre_pirls",
    "criterion_ml_reml",
    "criterion_ml_reml_exact",
    "criterion_ml_reml_gaussian_dynamic_joint",
    "criterion_ml_reml_pirls",
    "criterion_value",
    "gcv_score_gaussian",
    "gam_vcomp",
    "n_free_smoothing_params",
    "one_se_rule",
    "optimize_smoothing_params",
    "optimizer_endpoint_diagnostics",
    "resolve_ml_reml_scoring_backend",
    "resolve_smoothing_method",
    "sp_vcov",
    "supports_smoothing_method",
]
