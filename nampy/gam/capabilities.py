"""Task-facing discovery composed from statistical operation owners."""

from types import MappingProxyType

from ..contracts import CapabilityStatus
from ..diagnostics.fisher import FisherDiagnostics
from .families.registry import family_spec_for_instance
from .fisher import validate_fisher_family
from .model_state import _cov_bayes


def estimator_capabilities(estimator, *, task):
    """Read declarations and fitted metadata; never fit or evaluate new rows."""
    fitted = getattr(estimator, "gam_", None)
    ready = (
        fitted is not None
        or (task == "cox" and hasattr(estimator, "coef_"))
        or (task in {"quantile", "expectile", "calibrated_quantile"} and hasattr(estimator, "design_"))
    )

    def available(*conditions):
        requirements = (() if ready else ("fitted_model",)) + conditions
        return CapabilityStatus(
            "conditional" if requirements else "supported", conditions=requirements
        )

    no = CapabilityStatus(
        "unsupported", "This modeling task does not declare this operation."
    )
    result = {
        "predict_link": available(),
        "predict_components": available(),
        "predict_parameters": no,
        "predict_distribution": no,
        "predict_quantiles": no,
        "uncertainty.coefficients": available("available_covariance"),
        "uncertainty.quantile_function": no,
        "survival.curves": no,
        "survival.hazard": no,
        "survival.intervals": no,
        "survival.bootstrap": no,
    }
    covariance = (
        getattr(estimator, "covariance_", None)
        if task == "cox"
        else (_cov_bayes(fitted) if fitted is not None else None)
    )
    covariance_conditions = (
        () if ready and covariance is not None else ("available_covariance",)
    )
    result["uncertainty.coefficients"] = available(*covariance_conditions)
    if task in {"survival", "cox"}:
        result.update(
            {
                "predict_distribution": available(),
                "survival.curves": available(),
                "survival.hazard": no if task == "cox" else available(),
                "survival.intervals": available(*covariance_conditions),
                "survival.bootstrap": available("valid_subject_resampling"),
            }
        )
        if ready and task == "survival":
            from .inference.survival import validate_coefficient_coordinates

            try:
                validate_coefficient_coordinates(fitted)
            except NotImplementedError as error:
                result["survival.intervals"] = CapabilityStatus("unsupported", str(error))
        family = getattr(fitted, "family", None)
    elif task in {"quantile", "expectile", "calibrated_quantile"}:
        family = None
        result["uncertainty.coefficients"] = no
        if task == "calibrated_quantile":
            conditions = (() if ready and getattr(estimator, "posterior_covariance_", None) is not None
                          else ("calibrated_covariance",))
            result["uncertainty.coefficients"] = available(*conditions)
            result["uncertainty.quantile_function"] = available(*conditions)
    else:
        family = fitted.family if fitted is not None else estimator._build_gam().family
        if task == "distributional":
            result["predict_parameters"] = available()
            spec = family_spec_for_instance(family)
            if spec is not None and spec.supports_distribution_prediction:
                conditions = tuple(
                    getattr(
                        family,
                        "prediction_context_names",
                        getattr(family, "context_names", ()),
                    )
                )
                result["predict_distribution"] = available(*conditions)
                law = spec.law_factory() if spec.law_factory is not None else None
                if law is not None:
                    for operation in law.spec.operations:
                        result[f"distribution.{operation}"] = available(*conditions)
                    if "quantile" in law.spec.operations:
                        result["predict_quantiles"] = available(*conditions)
    for method in ("exact", "score_outer_product", "negative_hessian"):
        key = f"fisher.{method}"
        if family is None or task in {"survival", "cox"}:
            result[key] = CapabilityStatus(
                "unsupported",
                "This task has no declared Fisher observation experiment.",
            )
            continue
        try:
            validate_fisher_family(family, FisherDiagnostics(estimator=method))
        except NotImplementedError as error:
            result[key] = CapabilityStatus("unsupported", str(error))
        else:
            conditions = ()
            if getattr(estimator, "fisher_diagnostics", False):
                conditions += ("compatible_diagnostic_configuration",)
            if ready and getattr(fitted, "_fisher_offset_required_", False):
                conditions += ("offset",)
            if ready and getattr(family, "_trial_counts", None) is not None:
                conditions += ("trials",)
            result[key] = available(*conditions)
    return MappingProxyType(result)
