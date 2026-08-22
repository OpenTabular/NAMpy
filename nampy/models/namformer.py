"""Public estimator family generated from the NAMformer declaration."""

from ._registered import estimator_family

_family = estimator_family("namformer", module_name=__name__)
NAMformerRegressor = _family.regressor
NAMformerClassifier = _family.classifier
NAMformerLSS = _family.lss

__all__ = ["NAMformerRegressor", "NAMformerClassifier", "NAMformerLSS"]
