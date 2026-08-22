"""Public estimator family generated from the NBM declaration."""

from ._registered import estimator_family

_family = estimator_family("nbm", module_name=__name__)
NBMRegressor = _family.regressor
NBMClassifier = _family.classifier
NBMLSS = _family.lss

__all__ = ["NBMRegressor", "NBMClassifier", "NBMLSS"]
