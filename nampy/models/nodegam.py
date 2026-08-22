"""Public estimator family generated from the NodeGAM declaration."""

from ._registered import estimator_family

_family = estimator_family("nodegam", module_name=__name__)
NodeGAMRegressor = _family.regressor
NodeGAMClassifier = _family.classifier
NodeGAMLSS = _family.lss

__all__ = ["NodeGAMRegressor", "NodeGAMClassifier", "NodeGAMLSS"]
