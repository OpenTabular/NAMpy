"""Discovery from neural architecture, family and Fisher validators."""

from types import MappingProxyType

from ..contracts import CapabilityStatus
from ..diagnostics.fisher import FisherDiagnostics
from .families.registry import family_spec_for_instance, resolve_family
from .fisher import validate_fisher_model
from .fisher_likelihood import declared_fisher_family
from .objectives.quantile import Quantile


def estimator_capabilities(estimator):
    """Inspect configuration/state without forwarding, sampling or differentiation."""
    module = getattr(estimator, "model", None)
    ready = (
        module is not None
        and getattr(estimator, "fitted_preprocessing_", None) is not None
    )
    objective = getattr(estimator, "objective", None)
    task = getattr(estimator, "_estimator_type", None)
    survival = task == "survival"
    family = getattr(estimator, "family_", None)
    specification = family_spec_for_instance(family) if family is not None else None
    distributional = (
        getattr(estimator, "_prediction_kind", None) == "distributional"
        and not survival
    )
    if specification is None and distributional:
        name = getattr(objective, "family", None)
        if name is None:
            from .registry import get_architecture

            architecture = getattr(type(estimator), "architecture_name", None)
            name = (
                get_architecture(architecture).default_family
                if architecture
                else "normal"
            )
        specification = resolve_family(name)
    quantile = specification is not None and specification.distribution is Quantile

    def available(*conditions):
        requirements = (() if ready else ("fitted_model",)) + conditions
        return CapabilityStatus(
            "conditional" if requirements else "supported", conditions=requirements
        )

    no = CapabilityStatus("unsupported", "This task has no declared operation.")
    result = {
        "predict_link": available(),
        "predict_components": available(),
        "predict_parameters": available() if distributional and not quantile else no,
        "predict_distribution": no,
        "predict_quantiles": no,
        "uncertainty.coefficients": no,
        "survival.curves": available() if survival else no,
        "survival.hazard": no,
        "survival.intervals": no,
        "survival.bootstrap": available("valid_subject_resampling") if survival else no,
    }
    if survival:
        likelihood = getattr(
            estimator, "likelihood_", getattr(objective, "likelihood", None)
        )
        result["predict_distribution"] = available()
        result["survival.hazard"] = no if likelihood == "cox" else available()
    elif quantile:
        result["predict_quantiles"] = available("explicitly_fitted_levels")
    elif specification is not None and specification.supports_distribution_prediction:
        result["predict_distribution"] = available()
        law = (
            specification.law_factory()
            if specification.law_factory is not None
            else None
        )
        if law is not None:
            for operation in law.spec.operations:
                result[f"distribution.{operation}"] = available()
            if "quantile" in law.spec.operations:
                result["predict_quantiles"] = available()
    for method in ("exact", "score_outer_product", "negative_hessian"):
        status = available()
        if survival or quantile or getattr(objective, "loss_fct", None) is not None:
            status = CapabilityStatus(
                "unsupported",
                "No normalized independent-response Fisher likelihood is declared.",
            )
        elif ready:
            config = FisherDiagnostics(estimator=method, regression_scale=1.0)
            try:
                validate_fisher_model(module, config)
            except NotImplementedError as error:
                status = CapabilityStatus("unsupported", str(error))
        else:
            if specification is not None:
                try:
                    declared_fisher_family(
                        specification.distribution,
                        method,
                        robustness=(
                            getattr(objective, "distributional_kwargs", None) or {}
                        ).get("rob"),
                    )
                except NotImplementedError as error:
                    status = CapabilityStatus("unsupported", str(error))
            if method == "negative_hessian":
                if (
                    estimator.base_model.__dict__.get("supports_fisher_hessian")
                    is not True
                ):
                    status = CapabilityStatus(
                        "unsupported",
                        "Architecture does not declare smooth Hessian Fisher support.",
                    )
        if (
            status.state != "unsupported"
            and not distributional
            and task != "classifier"
        ):
            status = available("regression_scale")
        if status.state != "unsupported" and getattr(
            estimator, "fisher_offset_required_", False
        ):
            status = CapabilityStatus(
                "conditional", conditions=(*status.conditions, "offset")
            )
        result[f"fisher.{method}"] = status
    return MappingProxyType(result)
