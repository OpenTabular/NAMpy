"""Declarative registry for neural forward architectures."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _load_symbol(path: str):
    module_name, symbol_name = path.split(":", 1)
    return getattr(import_module(module_name), symbol_name)


@dataclass(frozen=True)
class NeuralArchitecture:
    """One architecture definition, independent of its training objective."""

    name: str
    estimator_prefix: str
    module_path: str
    config_path: str
    capabilities: frozenset[str]
    default_family: str = "normal"
    fixed_family: str | None = None
    distributional_defaults: Mapping[str, Any] = field(default_factory=dict)
    objective_defaults: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    preprocessor_defaults: Mapping[str, Any] = field(default_factory=dict)
    input_requirements: Mapping[str, Any] = field(default_factory=dict)
    estimator_mixin_path: str | None = None

    def __post_init__(self) -> None:
        name = self.name.strip().lower()
        if not name:
            raise ValueError("Architecture name cannot be empty.")
        if not self.estimator_prefix.strip():
            raise ValueError("estimator_prefix cannot be empty.")
        if not self.capabilities:
            raise ValueError(f"Architecture {name!r} must declare capabilities.")
        allowed = {
            "regression",
            "classification",
            "distributional",
            "additive_components",
            "interactions",
            "masked_pretraining",
            "fixed_linear_design",
            "native_training",
            "interaction_selection",
            "local_term_importance",
        }
        unknown = self.capabilities - allowed
        if unknown:
            raise ValueError(
                f"Architecture {name!r} has unknown capabilities {sorted(unknown)}."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "distributional_defaults",
            MappingProxyType(dict(self.distributional_defaults)),
        )
        objective_defaults = {
            str(objective): MappingProxyType(dict(defaults))
            for objective, defaults in self.objective_defaults.items()
        }
        object.__setattr__(
            self,
            "objective_defaults",
            MappingProxyType(objective_defaults),
        )
        object.__setattr__(
            self,
            "preprocessor_defaults",
            MappingProxyType(dict(self.preprocessor_defaults)),
        )
        object.__setattr__(
            self,
            "input_requirements",
            MappingProxyType(dict(self.input_requirements)),
        )

    @property
    def module(self):
        """Load and return the torch architecture class."""
        return _load_symbol(self.module_path)

    @property
    def config(self):
        """Load and return the architecture configuration dataclass."""
        return _load_symbol(self.config_path)

    @property
    def estimator_mixin(self):
        """Optional sklearn-estimator lifecycle mixin for staged architectures."""
        if self.estimator_mixin_path is None:
            return None
        return _load_symbol(self.estimator_mixin_path)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, capability: str) -> None:
        if not self.supports(capability):
            raise TypeError(
                f"Architecture {self.name!r} does not support {capability!r}."
            )


_REGISTRY: dict[str, NeuralArchitecture] = {}


def register_architecture(spec: NeuralArchitecture) -> NeuralArchitecture:
    """Register one architecture and reject ambiguous duplicate names."""
    if spec.name in _REGISTRY:
        raise ValueError(f"Architecture {spec.name!r} is already registered.")
    _REGISTRY[spec.name] = spec
    return spec


def get_architecture(name: str) -> NeuralArchitecture:
    try:
        return _REGISTRY[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown neural architecture {name!r}; available: "
            f"{', '.join(sorted(_REGISTRY))}."
        ) from exc


def architectures() -> Mapping[str, NeuralArchitecture]:
    """Return a read-only view of registered architectures."""
    return MappingProxyType(_REGISTRY)


_STANDARD = frozenset(
    {
        "regression",
        "classification",
        "distributional",
        "additive_components",
    }
)
_WITH_INTERACTIONS = _STANDARD | {"interactions"}


def _register_builtins() -> None:
    definitions = (
        ("linreg", "LinReg", "LinReg", "DefaultLinRegConfig", _STANDARD),
        ("nam", "NAM", "NAM", "DefaultNAMConfig", _WITH_INTERACTIONS),
        (
            "sian",
            "SIAN",
            "SIAN",
            "DefaultSIANConfig",
            _WITH_INTERACTIONS | {"interaction_selection"},
        ),
        ("snam", "SNAM", "SNAM", "DefaultSNAMConfig", _WITH_INTERACTIONS),
        (
            "gpnam",
            "GPNAM",
            "GPNAM",
            "DefaultGPNAMConfig",
            _WITH_INTERACTIONS | {"fixed_linear_design"},
        ),
        (
            "igann",
            "IGANN",
            "IGANN",
            "DefaultIGANNConfig",
            frozenset(
                {
                    "regression",
                    "classification",
                    "distributional",
                    "additive_components",
                    "native_training",
                }
            ),
        ),
        ("nbm", "NBM", "NBM", "DefaultNBMConfig", _WITH_INTERACTIONS),
        (
            "nbm_spam",
            "NBMSPAM",
            "NBMSPAM",
            "DefaultNBMSPAMConfig",
            _WITH_INTERACTIONS,
        ),
        (
            "spam",
            "SPAM",
            "SPAM",
            "DefaultSPAMConfig",
            _WITH_INTERACTIONS | {"local_term_importance"},
        ),
        ("natt", "NATT", "NATT", "DefaultNATTConfig", _WITH_INTERACTIONS),
        (
            "namformer",
            "NAMformer",
            "NAMformer",
            "DefaultNAMformerConfig",
            _WITH_INTERACTIONS,
        ),
        (
            "treenam",
            "TreeNAM",
            "TreeNAM",
            "DefaultTreeNAMConfig",
            _WITH_INTERACTIONS,
        ),
        (
            "ensemble_treenam",
            "EnsembleTreeNAM",
            "EnsembleTreeNAM",
            "DefaultEnsembleTreeNAMConfig",
            _WITH_INTERACTIONS,
        ),
        (
            "nodegam",
            "NodeGAM",
            "NodeGAM",
            "DefaultNodeGAMConfig",
            _WITH_INTERACTIONS | {"masked_pretraining"},
        ),
        (
            "qnam",
            "QNAM",
            "QNAM",
            "DefaultQNAMConfig",
            frozenset({"distributional", "additive_components", "interactions"}),
        ),
        (
            "spline_nam",
            "SplineNAM",
            "SplineNAM",
            "DefaultSplineNAMConfig",
            frozenset({"regression", "additive_components", "interactions"}),
        ),
    )
    for name, estimator_prefix, module_name, config_name, capabilities in definitions:
        kwargs = {}
        if name == "nam":
            kwargs = {
                "preprocessor_defaults": {
                    "numerical_method": "none",
                    "categorical_method": "one-hot",
                    "scaling": "minmax",
                    "dtype": np.float32,
                },
                "input_requirements": {
                    "features": "one network per PreTab feature block",
                    "categorical": "one-hot columns stay grouped by source feature",
                },
            }
        elif name == "qnam":
            kwargs = {
                "default_family": "quantile",
                "fixed_family": "quantile",
                "distributional_defaults": {
                    "quantiles": [0.25, 0.5, 0.75],
                    "enforce_monotonic": False,
                },
            }
        elif name in {"nbm", "nbm_spam", "spam"}:
            kwargs = {
                "preprocessor_defaults": {
                    "numerical_method": "none",
                    "categorical_method": "one-hot",
                    "scaling": "minmax",
                    "dtype": np.float32,
                },
                "input_requirements": {
                    "concepts": "PreTab feature blocks are flattened to scalar concepts",
                    "categorical": "grouped one-hot blocks are flattened by the architecture",
                },
            }
        elif name == "gpnam":
            kwargs = {
                "preprocessor_defaults": {
                    "numerical_method": "none",
                    "categorical_method": "one-hot",
                    "scaling": None,
                },
                "input_requirements": {
                    "numerical": "one scalar column per source feature",
                    "categorical": "one-hot encoded",
                },
            }
        elif name == "igann":
            kwargs = {
                "objective_defaults": {
                    "distributional": {"n_estimators": 100},
                },
                "preprocessor_defaults": {
                    "numerical_method": "none",
                    "categorical_method": "int",
                    "scaling": None,
                },
                "input_requirements": {
                    "numerical": "one scalar column per source feature",
                    "categorical": "integer encoded; IGANN drops the reference level",
                    "classification": "binary targets only",
                },
            }
        elif name == "sian":
            kwargs = {
                "estimator_mixin_path": "nampy.models.sian:_SIANSelectionMixin",
                "preprocessor_defaults": {
                    "numerical_method": "standardization",
                    "categorical_method": "one-hot",
                    "scaling": None,
                },
                "input_requirements": {
                    "interaction_selection": (
                        "logical source features are grouped across transformed columns"
                    ),
                },
            }
        elif name == "spline_nam":
            kwargs = {
                "preprocessor_defaults": {
                    "numerical_method": "minmax",
                    "categorical_method": "int",
                    "scaling": None,
                },
                "input_requirements": {
                    "features": "one scalar transformed column per source feature",
                },
            }
        register_architecture(
            NeuralArchitecture(
                name=name,
                estimator_prefix=estimator_prefix,
                module_path=f"nampy.neural.architectures.{name}:{module_name}",
                config_path=(
                    f"nampy.neural.configs.{name}_config:{config_name}"
                ),
                capabilities=capabilities,
                **kwargs,
            )
        )


_register_builtins()


__all__ = [
    "NeuralArchitecture",
    "architectures",
    "get_architecture",
    "register_architecture",
]
