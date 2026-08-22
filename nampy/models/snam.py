"""Public estimator family generated from the sparse NAM declaration."""

from ._registered import estimator_family

_family = estimator_family("snam", module_name=__name__)
SNAMRegressor = _family.regressor
SNAMClassifier = _family.classifier
SNAMLSS = _family.lss

__all__ = ["SNAMRegressor", "SNAMClassifier", "SNAMLSS"]
