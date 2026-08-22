"""Public estimator family generated from the NATT declaration."""

from ._registered import estimator_family

_family = estimator_family("natt", module_name=__name__)
NATTRegressor = _family.regressor
NATTClassifier = _family.classifier
NATTLSS = _family.lss

__all__ = ["NATTRegressor", "NATTClassifier", "NATTLSS"]
