"""Public estimator family generated from the neural linear declaration."""

from ._registered import estimator_family

_family = estimator_family("linreg", module_name=__name__)
LinRegRegressor = _family.regressor
LinRegClassifier = _family.classifier
LinRegLSS = _family.lss

__all__ = ["LinRegRegressor", "LinRegClassifier", "LinRegLSS"]
