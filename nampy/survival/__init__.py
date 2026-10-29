"""Shared survival targets, predictions, evaluation and cohort inference."""

from importlib import import_module

_EXPORT_MODULES = {
    "SurvivalBootstrapError": ".bootstrap",
    "SurvivalBootstrapFailure": ".bootstrap",
    "SurvivalBootstrapResult": ".bootstrap",
    "bootstrap_survival": ".bootstrap",
    "PiecewiseExponentialDesign": ".contracts",
    "SurvivalComponents": ".contracts",
    "SurvivalTarget": ".contracts",
    "piecewise_exponential_data": ".data",
    "BrierScore": ".evaluation",
    "CensoringDistribution": ".evaluation",
    "ConcordanceResult": ".evaluation",
    "SurvivalCurve": ".evaluation",
    "brier_score": ".evaluation",
    "concordance_index": ".evaluation",
    "survival_curve": ".evaluation",
    "LongitudinalSurvivalTarget": ".longitudinal",
    "SurvivalTrajectory": ".longitudinal",
    "predict_trajectory": ".longitudinal",
    "SurvivalPrediction": ".prediction",
    "survival_from_hazard": ".prediction",
    "SurvivalCurveInterval": ".uncertainty",
    "SurvivalEffectInterval": ".uncertainty",
    "SurvivalEffectSamples": ".uncertainty",
    "combine_survival_predictions": ".uncertainty",
    "prepare_survival_curve_interval": ".uncertainty",
    "prepare_survival_effect_interval": ".uncertainty",
    "prepare_survival_hazard_interval": ".uncertainty",
    "prepare_survival_rmst_interval": ".uncertainty",
}


def __getattr__(name):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = list(_EXPORT_MODULES)
