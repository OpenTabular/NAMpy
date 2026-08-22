"""Registered neural estimator classes and stable public-class generation."""

from __future__ import annotations

from dataclasses import dataclass

from ..neural.registry import get_architecture
from .classifier import NeuralClassifier
from .lss import NeuralLSS
from .regressor import NeuralRegressor


def _with_architecture_defaults(spec, kwargs, *, objective):
    resolved = dict(kwargs)
    for name, value in spec.objective_defaults.get(objective, {}).items():
        resolved.setdefault(name, value)
    for name, value in spec.preprocessor_defaults.items():
        if name not in resolved and f"preprocessor__{name}" not in resolved:
            resolved[name] = value
    return resolved


class RegisteredNeuralRegressor(NeuralRegressor):
    architecture_name: str

    def __init__(self, **kwargs):
        spec = get_architecture(self.architecture_name)
        spec.require("regression")
        self.architecture_name = spec.name
        super().__init__(
            model=spec.module,
            config=spec.config,
            **_with_architecture_defaults(spec, kwargs, objective="regression"),
        )


class RegisteredNeuralClassifier(NeuralClassifier):
    architecture_name: str

    def __init__(self, **kwargs):
        spec = get_architecture(self.architecture_name)
        spec.require("classification")
        self.architecture_name = spec.name
        super().__init__(
            model=spec.module,
            config=spec.config,
            **_with_architecture_defaults(spec, kwargs, objective="classification"),
        )


class RegisteredNeuralLSS(NeuralLSS):
    architecture_name: str

    def __init__(self, **kwargs):
        spec = get_architecture(self.architecture_name)
        spec.require("distributional")
        family = kwargs.pop("family", spec.default_family)
        if spec.fixed_family is not None and str(family).lower() != spec.fixed_family:
            raise ValueError(
                f"{spec.name} only supports family={spec.fixed_family!r}."
            )
        distributional_kwargs = kwargs.pop("distributional_kwargs", None)
        if distributional_kwargs is None:
            distributional_kwargs = dict(spec.distributional_defaults) or None
        elif spec.fixed_family == "quantile":
            if distributional_kwargs.get("enforce_monotonic", False):
                raise ValueError(
                    "QNAMLSS applies monotonicity in the model; set "
                    "enforce_monotonic=False on the distribution family."
                )
        self.architecture_name = spec.name
        kwargs = _with_architecture_defaults(
            spec, kwargs, objective="distributional"
        )
        super().__init__(
            model=spec.module,
            config=spec.config,
            family=family,
            distributional_kwargs=distributional_kwargs,
            **kwargs,
        )


_BASES = {
    "regression": RegisteredNeuralRegressor,
    "classification": RegisteredNeuralClassifier,
    "distributional": RegisteredNeuralLSS,
}


def registered_estimator_class(
    class_name: str,
    *,
    architecture: str,
    objective: str,
    module_name: str,
    doc: str | None = None,
):
    """Build a stable module-level estimator class from one architecture spec."""
    spec = get_architecture(architecture)
    spec.require(objective)
    mixin = spec.estimator_mixin
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


def estimator_family(architecture: str, *, module_name: str) -> NeuralEstimatorFamily:
    """Generate every estimator surface supported by an architecture spec."""
    spec = get_architecture(architecture)
    generated = {}
    names = {
        "regression": ("regressor", f"{spec.estimator_prefix}Regressor"),
        "classification": ("classifier", f"{spec.estimator_prefix}Classifier"),
        "distributional": ("lss", f"{spec.estimator_prefix}LSS"),
    }
    for objective, (field_name, class_name) in names.items():
        if spec.supports(objective):
            generated[field_name] = registered_estimator_class(
                class_name,
                architecture=spec.name,
                objective=objective,
                module_name=module_name,
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
