"""Public estimator family generated from the IGANN declaration."""

from ._registered import estimator_family

_family = estimator_family("igann", module_name=__name__)
IGANNRegressor = _family.regressor
IGANNClassifier = _family.classifier
IGANNLSS = _family.lss

__all__ = ["IGANNRegressor", "IGANNClassifier", "IGANNLSS"]
