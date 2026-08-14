from ..basemodels.spline_nam import SplineNAM
from ..configs.spline_nam_config import DefaultSplineNAMConfig
from .sklearn_regressor import SklearnBaseRegressor


class SplineNAMRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        kwargs.setdefault("numerical_preprocessing", "minmax")
        kwargs.setdefault("categorical_preprocessing", "int")
        kwargs.setdefault("n_knots", DefaultSplineNAMConfig.n_knots)
        super().__init__(model=SplineNAM, config=DefaultSplineNAMConfig, **kwargs)
