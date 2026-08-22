"""Public estimator family generated from the GP-NAM declaration."""

from ._registered import estimator_family

_family = estimator_family("gpnam", module_name=__name__)
GPNAMRegressor = _family.regressor
GPNAMClassifier = _family.classifier
GPNAMLSS = _family.lss

__all__ = ["GPNAMRegressor", "GPNAMClassifier", "GPNAMLSS"]
