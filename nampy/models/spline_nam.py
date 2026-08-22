"""Public estimator family generated from the SplineNAM declaration."""

from ._registered import estimator_family

_family = estimator_family("spline_nam", module_name=__name__)
SplineNAMRegressor = _family.regressor

__all__ = ["SplineNAMRegressor"]
