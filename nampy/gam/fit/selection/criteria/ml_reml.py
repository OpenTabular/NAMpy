"""
ML/REML/LAML criterion values: backend selection and evaluation.

Chooses between the Gaussian exact, Gaussian dynamic, P-IRLS Laplace, and
P-IRLS dynamic backends based on the model family and design structure,
then evaluates the appropriate criterion value.

References
----------
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood
  estimation of semiparametric generalized linear models.
  https://doi.org/10.1111/j.1467-9868.2010.00749.x
- Wood, S. N., Pya, N., and Säfken, B. (2016). Smoothing Parameter and Model Selection
  for General Smooth Models.
  https://doi.org/10.1080/01621459.2016.1180986
"""

import numpy as np

from ...backends import GENERAL_FAMILY_BACKEND
from ...capabilities import has_transformed_observations, raise_ml_reml_backend_error
from ...solvers.general_family.fixed_smoothing import (
    criterion_ml_reml_general_family,
)
from ..reparam import (
    can_use_exact_gaussian_ml_reml,
    can_use_simple_ml_reml_structure,
)
from .gaussian import criterion_ml_reml_exact, criterion_ml_reml_exact_dynamic
from .pirls import criterion_ml_reml_pirls, criterion_ml_reml_pirls_dynamic


def resolve_ml_reml_scoring_backend(model, method="reml"):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
        raise ValueError("method must be one of {'ml', 'reml', 'laml'}")

    if bool(
        getattr(model.family, "supports_closed_form_solve", False)
    ) and can_use_exact_gaussian_ml_reml(model) and not has_transformed_observations(model):
        return "gaussian_exact"

    if bool(getattr(model.family, "supports_closed_form_solve", False)):
        return "gaussian_dynamic"

    if bool(
        getattr(model.family, "supports_pirls", False)
    ) and can_use_simple_ml_reml_structure(model):
        return "pirls_laplace"

    if bool(getattr(model.family, "supports_pirls", False)):
        return "pirls_laplace_dynamic"
    if str(getattr(model.family, "family_class", "")).lower() == "general":
        return GENERAL_FAMILY_BACKEND

    return None


def criterion_ml_reml(model, y, log_sp, method):
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend == "gaussian_exact":
        score_exact = criterion_ml_reml_exact(model, y, log_sp, method.upper())
        if np.isfinite(float(score_exact)):
            return float(score_exact)
        return criterion_ml_reml_exact_dynamic(model, y, log_sp, method.upper())
    if backend == "gaussian_dynamic":
        return criterion_ml_reml_exact_dynamic(model, y, log_sp, method.upper())
    if backend == "pirls_laplace":
        branch_method = "REML" if str(method).lower() in {"reml", "laml"} else "ML"
        return criterion_ml_reml_pirls(model, y, log_sp, branch_method)
    if backend == "pirls_laplace_dynamic":
        branch_method = "REML" if str(method).lower() in {"reml", "laml"} else "ML"
        return criterion_ml_reml_pirls_dynamic(model, y, log_sp, branch_method)
    if backend == GENERAL_FAMILY_BACKEND:
        return criterion_ml_reml_general_family(model, y, log_sp, method)
    raise_ml_reml_backend_error(model, method)
    raise AssertionError("unreachable")
