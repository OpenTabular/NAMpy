"""Public estimator family generated from the SPAM declaration."""

from ._registered import estimator_family

_family = estimator_family("spam", module_name=__name__)
SPAMRegressor = _family.regressor
SPAMClassifier = _family.classifier
SPAMLSS = _family.lss

__all__ = ["SPAMRegressor", "SPAMClassifier", "SPAMLSS"]
