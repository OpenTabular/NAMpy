"""Public estimator family generated from the TreeNAM declaration."""

from ._registered import estimator_family

_family = estimator_family("treenam", module_name=__name__)
TreeNAMRegressor = _family.regressor
TreeNAMClassifier = _family.classifier
TreeNAMLSS = _family.lss

__all__ = ["TreeNAMRegressor", "TreeNAMClassifier", "TreeNAMLSS"]
