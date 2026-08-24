"""Lazy public estimator exports for the GAM and neural backends."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORT_MODULES = {
    "GAMClassifier": ".gam",
    "GAMRegressor": ".gam",
    "GAMLSS": ".gamlss",
    "NAMClassifier": ".nam",
    "NAMLSS": ".nam",
    "NAMRegressor": ".nam",
    "SIANClassifier": ".sian",
    "SIANLSS": ".sian",
    "SIANRegressor": ".sian",
    "NeuralClassifier": ".classifier",
    "NeuralLSS": ".lss",
    "NeuralRegressor": ".regressor",
    "NeuralEnsemble": ".ensemble",
    "NeuralEstimatorFamily": "._registered",
    "estimator_family": "._registered",
    "LinRegClassifier": ".linreg",
    "LinRegLSS": ".linreg",
    "LinRegRegressor": ".linreg",
    "QNAMLSS": ".qnam",
    "GPNAMClassifier": ".gpnam",
    "GPNAMLSS": ".gpnam",
    "GPNAMRegressor": ".gpnam",
    "IGANNClassifier": ".igann",
    "IGANNLSS": ".igann",
    "IGANNRegressor": ".igann",
    "NBMRegressor": ".nbm",
    "NBMClassifier": ".nbm",
    "NBMLSS": ".nbm",
    "NBMSPAMRegressor": ".nbm_spam",
    "NBMSPAMClassifier": ".nbm_spam",
    "NBMSPAMLSS": ".nbm_spam",
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
    "SPAMRegressor": ".spam",
    "SPAMClassifier": ".spam",
    "SPAMLSS": ".spam",
}

if TYPE_CHECKING:
    from ._registered import NeuralEstimatorFamily, estimator_family
    from .classifier import NeuralClassifier
    from .ensemble import NeuralEnsemble
    from .ensemble_treenam import (
        EnsembleTreeNAMClassifier,
        EnsembleTreeNAMLSS,
        EnsembleTreeNAMRegressor,
    )
    from .gam import GAMClassifier, GAMRegressor
    from .gamlss import GAMLSS
    from .gpnam import GPNAMLSS, GPNAMClassifier, GPNAMRegressor
    from .igann import IGANNLSS, IGANNClassifier, IGANNRegressor
    from .linreg import LinRegClassifier, LinRegLSS, LinRegRegressor
    from .lss import NeuralLSS
    from .nam import NAMLSS, NAMClassifier, NAMRegressor
    from .namformer import NAMformerClassifier, NAMformerLSS, NAMformerRegressor
    from .natt import NATTLSS, NATTClassifier, NATTRegressor
    from .nbm import NBMLSS, NBMClassifier, NBMRegressor
    from .nbm_spam import NBMSPAMLSS, NBMSPAMClassifier, NBMSPAMRegressor
    from .nodegam import NodeGAMClassifier, NodeGAMLSS, NodeGAMRegressor
    from .qnam import QNAMLSS
    from .regressor import NeuralRegressor
    from .sian import SIANLSS, SIANClassifier, SIANRegressor
    from .snam import SNAMLSS, SNAMClassifier, SNAMRegressor
    from .spam import SPAMLSS, SPAMClassifier, SPAMRegressor
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
