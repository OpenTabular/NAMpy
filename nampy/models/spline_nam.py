from ..basemodels.spline_nam import SplineNAM
from ..configs.spline_nam_config import DefaultSplineNAMConfig
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from .sklearn_regressor import SklearnBaseRegressor


class SplineNAMRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        if "n_knots" in kwargs and "spline_n_knots" not in kwargs:
            kwargs["spline_n_knots"] = kwargs["n_knots"]
        super().__init__(model=SplineNAM, config=DefaultSplineNAMConfig, **kwargs)


class SplineNAMClassifier(SklearnBaseClassifier):
    def __init__(self, **kwargs):
        if "n_knots" in kwargs and "spline_n_knots" not in kwargs:
            kwargs["spline_n_knots"] = kwargs["n_knots"]
        super().__init__(model=SplineNAM, config=DefaultSplineNAMConfig, **kwargs)


class SplineNAMLSS(SklearnBaseLSS):
    def __init__(self, **kwargs):
        if "n_knots" in kwargs and "spline_n_knots" not in kwargs:
            kwargs["spline_n_knots"] = kwargs["n_knots"]
        super().__init__(model=SplineNAM, config=DefaultSplineNAMConfig, **kwargs)
