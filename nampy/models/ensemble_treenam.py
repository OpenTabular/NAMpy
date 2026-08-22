"""Public estimator family generated from the ensemble TreeNAM declaration."""

from ._registered import estimator_family

_family = estimator_family("ensemble_treenam", module_name=__name__)
EnsembleTreeNAMRegressor = _family.regressor
EnsembleTreeNAMClassifier = _family.classifier
EnsembleTreeNAMLSS = _family.lss

__all__ = [
    "EnsembleTreeNAMRegressor",
    "EnsembleTreeNAMClassifier",
    "EnsembleTreeNAMLSS",
]
