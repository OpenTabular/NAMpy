"""Public estimator family generated from the NAM architecture declaration."""

from ._registered import estimator_family

_family = estimator_family("nam", module_name=__name__)
NAMRegressor = _family.regressor
NAMClassifier = _family.classifier
NAMLSS = _family.lss

__all__ = ["NAMRegressor", "NAMClassifier", "NAMLSS"]
