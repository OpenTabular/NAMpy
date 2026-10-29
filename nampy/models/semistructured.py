"""Registered identifiable spline/neural estimator family."""

from ._registered import estimator_family

_family = estimator_family("semistructured", module_name=__name__)
SemiStructuredRegressor = _family.regressor
SemiStructuredLSS = _family.lss
SemiStructuredClassifier = _family.classifier

__all__ = ["SemiStructuredRegressor", "SemiStructuredLSS", "SemiStructuredClassifier"]
