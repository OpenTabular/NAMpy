from ..basemodels.spline_nam import SplineNAM
from ..configs.spline_nam_config import DefaultSplineNAMConfig
from .sklearn_regressor import SklearnBaseRegressor


class SplineNAMRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=SplineNAM, config=DefaultSplineNAMConfig, **kwargs)
