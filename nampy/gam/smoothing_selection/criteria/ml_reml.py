"""
ML/REML/LAML criterion values: backend selection and evaluation.

Chooses between the Gaussian exact, Gaussian dynamic, P-IRLS Laplace, and
P-IRLS dynamic backends based on the model family and design structure,
then evaluates the appropriate criterion value.
"""

import numpy as np

from .gaussian import criterion_ml_reml_exact, criterion_ml_reml_exact_dynamic
from .pirls import criterion_ml_reml_pirls, criterion_ml_reml_pirls_dynamic


def _model_has_random_effect_smooth(model) -> bool:
    """``bs=\"re\"`` designs: col(1) ⊆ col(Z) breaks the Laplace ``Z'Z+\\lambda`` REML path."""
    for tb in getattr(model, "term_blocks_", None) or []:
        if str(getattr(tb, "term_type", "")).lower() == "random_effect":
            return True
    return False


def resolve_ml_reml_scoring_backend(model, method="reml"):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
        raise ValueError("method must be one of {'ml', 'reml', 'laml'}")

    if (
        bool(getattr(model.family, "supports_closed_form_solve", False))
        and model._can_use_exact_gaussian_ml_reml()
    ):
        return "gaussian_exact"

    if bool(getattr(model.family, "supports_closed_form_solve", False)):
        return "gaussian_dynamic"

    if (
        bool(getattr(model.family, "supports_pirls", False))
        and model._can_use_simple_ml_reml_structure()
    ):
        return "pirls_laplace"

    if bool(getattr(model.family, "supports_pirls", False)):
        return "pirls_laplace_dynamic"

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
    model._raise_ml_reml_backend_error(method)
    raise AssertionError("unreachable")
