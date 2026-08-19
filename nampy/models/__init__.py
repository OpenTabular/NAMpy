"""Lazy public estimator exports for the GAM and neural backends."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORT_MODULES = {
    "GAMClassifier": ".gam",
    "GAMRegressor": ".gam",
    "NAMClassifier": ".nam",
    "NAMLSS": ".nam",
    "NAMRegressor": ".nam",
    "NeuralClassifier": ".classifier",
    "NeuralLSS": ".lss",
    "NeuralRegressor": ".regressor",
    "LinRegClassifier": ".linreg",
    "LinRegLSS": ".linreg",
    "LinRegRegressor": ".linreg",
    "QNAMLSS": ".qnam",
    "GPNAMClassifier": ".gpnam",
    "GPNAMLSS": ".gpnam",
    "GPNAMRegressor": ".gpnam",
    "NBMRegressor": ".nbm",
    "NBMClassifier": ".nbm",
    "NBMLSS": ".nbm",
    "NATTRegressor": ".natt",
    "NATTClassifier": ".natt",
    "NATTLSS": ".natt",
    "NAMformerClassifier": ".namformer",
    "NAMformerLSS": ".namformer",
    "NAMformerRegressor": ".namformer",
    "TreeNAMRegressor": ".treenam",
    "TreeNAMClassifier": ".treenam",
    "TreeNAMLSS": ".treenam",
    "EnsembleTreeNAMRegressor": ".ensemble_treenam",
    "EnsembleTreeNAMClassifier": ".ensemble_treenam",
    "EnsembleTreeNAMLSS": ".ensemble_treenam",
    "SplineNAMRegressor": ".spline_nam",
    "SNAMRegressor": ".snam",
    "SNAMClassifier": ".snam",
    "SNAMLSS": ".snam",
    "NodeGAMRegressor": ".nodegam",
    "NodeGAMClassifier": ".nodegam",
    "NodeGAMLSS": ".nodegam",
}

if TYPE_CHECKING:
    from .classifier import NeuralClassifier
    from .ensemble_treenam import (
        EnsembleTreeNAMClassifier,
        EnsembleTreeNAMLSS,
        EnsembleTreeNAMRegressor,
    )
    from .gam import GAMClassifier, GAMRegressor
    from .gpnam import GPNAMLSS, GPNAMClassifier, GPNAMRegressor
    from .linreg import LinRegClassifier, LinRegLSS, LinRegRegressor
    from .lss import NeuralLSS
    from .nam import NAMLSS, NAMClassifier, NAMRegressor
    from .namformer import NAMformerClassifier, NAMformerLSS, NAMformerRegressor
    from .natt import NATTLSS, NATTClassifier, NATTRegressor
    from .nbm import NBMLSS, NBMClassifier, NBMRegressor
    from .nodegam import NodeGAMClassifier, NodeGAMLSS, NodeGAMRegressor
    from .qnam import QNAMLSS
    from .regressor import NeuralRegressor
    from .snam import SNAMLSS, SNAMClassifier, SNAMRegressor
    from .spline_nam import SplineNAMRegressor
    from .treenam import TreeNAMClassifier, TreeNAMLSS, TreeNAMRegressor


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = list(_EXPORT_MODULES)
