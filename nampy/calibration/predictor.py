"""Post-fit predictor snapshots; fitting and calibration populations stay separate."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import is_classifier, is_regressor

from ..distributions import DistributionPrediction
from ..persistence import load_estimator, save_estimator, snapshot_estimator
from .conformal import ConformalCalibration, _level, _vector, calibrate_conformal
from .grouped import GroupedConformalCalibration, calibrate_grouped


def _predict(predictors, route, X, options, level):
    if route == "quantiles":
        return predictors[0].predict_quantiles(X, [(1-level)/2, (1+level)/2], **options)
    if route == "quantile_pair":
        return np.column_stack([_vector(model.predict(X, **options), "quantile prediction")
                                for model in predictors])
    method = {"response": "predict", "point": "predict_point", "distribution": "predict_distribution"}[route]
    prediction = getattr(predictors[0], method)(X, **options)
    if route == "distribution" and not isinstance(prediction, DistributionPrediction):
        raise NotImplementedError("Calibration requires scalar response distributions; survival predictions are unsupported.")
    return prediction


@dataclass(frozen=True)
class ConformalPredictor:
    """A post-fit snapshot plus calibration state, without fit/set_params methods.

    Source-estimator refits cannot change this predictor. Recalibrate explicitly
    to use a different fit or coverage level. Prediction kwargs (notably offsets
    and distribution context) belong to each supplied dataset and are never
    carried over from the calibration cohort. No calibration observations or
    execution machinery are retained.
    """

    _predictors: tuple
    _route: str
    calibration: ConformalCalibration | GroupedConformalCalibration

    def predict_interval(self, X, *, density_ratio=None, groups=None, **predict_kwargs):
        prediction = _predict(self._predictors, self._route, X, predict_kwargs, self.calibration.level)
        if isinstance(self.calibration, GroupedConformalCalibration):
            if density_ratio is not None or groups is None:
                raise ValueError("Grouped prediction requires group labels and does not support density ratios.")
            result = self.calibration.interval(prediction, groups=groups)
        else:
            if groups is not None:
                raise ValueError("Group labels require grouped calibration.")
            result = self.calibration.interval(prediction, density_ratio=density_ratio)
        if len(result.lower) != len(X):
            raise ValueError("Predictions must preserve supplied subject rows.")
        return result

    def _artifact_state(self):
        return dict(self.__dict__)

    def save_model(self, path):
        """Persist fitted snapshots and scores through NAMpy's trusted artifacts."""
        return save_estimator(self, path)

    @classmethod
    def load_model(cls, path):
        return load_estimator(path, cls)


def calibrate_regressor(estimator, X, y, *, method="absolute", level=.9,
                        prediction="response", upper_estimator=None, predict_kwargs=None,
                        density_ratio=None, weight_source="known", groups=None, group_mode="cluster"):
    """Calibrate a snapshot of an already fitted model on separate held-out data.

    prediction='response' uses scalar regression predict; 'point' uses the
    family-defined predict_point; 'quantiles' uses explicitly supported quantile
    predictions without a density; 'distribution' uses the shared distribution
    contract. With method='quantile', supply either a distribution route or
    quantile predictions or lower/upper scalar regressors via estimator/upper_estimator. Pair models
    should estimate the central quantiles for level; levels are caller-specified
    by those models, never inferred from parameter arrays. Both pair models
    receive the same explicit prediction kwargs.

    No fitting/tuning/splitting occurs. Callers must ensure calibration was not
    used to train or select these models. density_ratio supplies covariate-shift
    ratios; groups selects fixed-stratum or complete-cluster calibration. Case
    weights, combined weighting/grouping, classification, survival (use its
    dedicated lower-bound API), time series and vector responses are unsupported.
    """
    level = _level(level)
    if groups is not None and (density_ratio is not None or weight_source != "known"):
        raise NotImplementedError("Combined density-ratio and grouped calibration is unsupported.")
    if groups is None and group_mode != "cluster":
        raise ValueError("group_mode requires calibration group labels.")
    if prediction not in {"response", "point", "distribution", "quantiles"}:
        raise ValueError("prediction must be response, point, distribution or quantiles.")
    if method not in {"absolute", "quantile"}:
        raise NotImplementedError("Only absolute-residual and quantile split conformal are supported.")
    if upper_estimator is not None and (method != "quantile" or prediction != "response"):
        raise ValueError("upper_estimator requires quantile mode with response predictions.")
    if method == "absolute" and prediction == "quantiles":
        raise ValueError("Quantiles predictions require quantile calibration.")
    if method == "quantile" and prediction not in {"distribution", "quantiles"} and upper_estimator is None:
        raise ValueError("Quantile calibration requires quantile predictions, a distribution or explicit lower/upper regressors.")
    models = (estimator,) if upper_estimator is None else (estimator, upper_estimator)
    for model in models:
        role = getattr(model, "_estimator_type", None)
        if role == "survival" or is_classifier(model) or (prediction == "response" and not is_regressor(model)):
            raise NotImplementedError("Response calibration requires scalar regressors; classification and survival are unsupported.")
        if not callable(getattr(model, {"response": "predict", "point": "predict_point", "distribution": "predict_distribution", "quantiles": "predict_quantiles"}[prediction], None)):
            raise NotImplementedError(f"The estimator does not support {prediction} predictions.")
    # Snapshot before evaluation. Existing backend artifact owners remove runtime
    # state, and failures publish no partial calibration or changes to the source.
    predictors = tuple(snapshot_estimator(model) for model in models)
    route = prediction if upper_estimator is None else "quantile_pair"
    values = _predict(predictors, route, X, dict(predict_kwargs or {}), level)
    if len(y) != len(X):
        raise ValueError("Calibration features and responses must have identical row counts.")
    calibration = (calibrate_grouped(y, values, groups, mode=group_mode, method=method, level=level)
                   if groups is not None else calibrate_conformal(y, values, method=method, level=level,
                       density_ratio=density_ratio, weight_source=weight_source))
    return ConformalPredictor(predictors, route, calibration)
