"""Registered neural estimator classes and stable public-class generation."""

from __future__ import annotations

from dataclasses import dataclass

from ..neural.registry import get_architecture
from ._survival import NeuralSurvival
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


def _registered_parameters(
    estimator,
    task,
    architecture_config,
    objective,
    training,
    preprocessing,
    fisher_diagnostics,
):
    from ..neural.configuration import (
        ObjectiveConfig,
        PreprocessingConfig,
        SurvivalObjectiveConfig,
    )

    spec = get_architecture(estimator.architecture_name)
    spec.require(task)
    if architecture_config is None:
        architecture_config = spec.config(**dict(spec.objective_defaults.get(task, {})))
    if objective is None:
        objective = (
            SurvivalObjectiveConfig()
            if task == "survival"
            else ObjectiveConfig(
                family=(spec.default_family or "normal")
                if task == "distributional"
                else None,
                distributional_kwargs=(dict(spec.distributional_defaults) or None)
                if task == "distributional"
                else None,
            )
        )
    if task == "distributional":
        if (
            spec.fixed_family is not None
            and objective.family is not None
            and objective.family != spec.fixed_family
        ):
            raise ValueError(f"{spec.name} requires family={spec.fixed_family!r}.")
        validate = getattr(spec.module, "validate_distributional_kwargs", None)
        if validate is not None:
            validate(objective.distributional_kwargs)
    if preprocessing is None:
        defaults = dict(spec.preprocessor_defaults)
        if task == "classification":
            defaults["task"] = "classification"
        preprocessing = PreprocessingConfig(**defaults)
    return {
        "model": spec.module,
        "config_class": spec.config,
        "architecture_config": architecture_config,
        "objective": objective,
        "training": training,
        "preprocessing": preprocessing,
        "fisher_diagnostics": fisher_diagnostics,
    }


class RegisteredNeuralRegressor(NeuralRegressor):
    architecture_name: str

    def __init__(
        self,
        *,
        architecture_config=None,
        objective=None,
        training=None,
        preprocessing=None,
        fisher_diagnostics=False,
    ):
        super().__init__(
            **_registered_parameters(
                self,
                "regression",
                architecture_config,
                objective,
                training,
                preprocessing,
                fisher_diagnostics,
            )
        )


class RegisteredNeuralClassifier(NeuralClassifier):
    architecture_name: str

    def __init__(
        self,
        *,
        architecture_config=None,
        objective=None,
        training=None,
        preprocessing=None,
        fisher_diagnostics=False,
    ):
        super().__init__(
            **_registered_parameters(
                self,
                "classification",
                architecture_config,
                objective,
                training,
                preprocessing,
                fisher_diagnostics,
            )
        )


class RegisteredNeuralLSS(NeuralLSS):
    architecture_name: str

    def __init__(
        self,
        *,
        architecture_config=None,
        objective=None,
        training=None,
        preprocessing=None,
        fisher_diagnostics=False,
    ):
        super().__init__(
            **_registered_parameters(
                self,
                "distributional",
                architecture_config,
                objective,
                training,
                preprocessing,
                fisher_diagnostics,
            )
        )


class RegisteredNeuralSurvival(NeuralSurvival):
    architecture_name: str

    def __init__(
        self,
        *,
        architecture_config=None,
        objective=None,
        training=None,
        preprocessing=None,
        fisher_diagnostics=False,
    ):
        super().__init__(
            **_registered_parameters(
                self,
                "survival",
                architecture_config,
                objective,
                training,
                preprocessing,
                fisher_diagnostics,
            )
        )


_BASES = {
    "regression": RegisteredNeuralRegressor,
    "classification": RegisteredNeuralClassifier,
    "distributional": RegisteredNeuralLSS,
    "survival": RegisteredNeuralSurvival,
}


def registered_estimator_class(
    class_name: str,
    *,
    architecture: str,
    objective: str,
    module_name: str,
    doc: str | None = None,
    mixin: type | None = None,
):
    """Build a stable module-level estimator class from one architecture spec.

    Parameters
    ----------
    mixin : type or None
        Optional estimator-layer methods composed before the task adapter.
    """
    spec = get_architecture(architecture)
    spec.require(objective)
    bases = (_BASES[objective],) if mixin is None else (mixin, _BASES[objective])
    return type(
        class_name,
        bases,
        {
            "architecture_name": spec.name,
            "__module__": module_name,
            "__doc__": doc
            or f"{spec.name} neural estimator for the {objective} objective.",
        },
    )


@dataclass(frozen=True)
class NeuralEstimatorFamily:
    """Public estimator classes generated from one architecture declaration."""

    regressor: type | None = None
    classifier: type | None = None
    lss: type | None = None
    survival: type | None = None


def estimator_family(
    architecture: str, *, module_name: str, mixin: type | None = None
) -> NeuralEstimatorFamily:
    """Generate every estimator surface supported by an architecture spec.

    Parameters
    ----------
    mixin : type or None
        Optional estimator-layer methods shared by the generated task adapters.
    """
    spec = get_architecture(architecture)
    generated = {}
    names = {
        "regression": ("regressor", f"{spec.estimator_prefix}Regressor"),
        "classification": ("classifier", f"{spec.estimator_prefix}Classifier"),
        "distributional": ("lss", f"{spec.estimator_prefix}LSS"),
        "survival": ("survival", f"{spec.estimator_prefix}Survival"),
    }
    for objective, (field_name, class_name) in names.items():
        if spec.supports(objective):
            generated[field_name] = registered_estimator_class(
                class_name,
                architecture=spec.name,
                objective=objective,
                module_name=module_name,
                mixin=mixin,
            )
    return NeuralEstimatorFamily(**generated)


__all__ = [
    "RegisteredNeuralClassifier",
    "RegisteredNeuralLSS",
    "RegisteredNeuralRegressor",
    "NeuralEstimatorFamily",
    "estimator_family",
    "registered_estimator_class",
]
