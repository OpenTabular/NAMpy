"""Shared postfit contributions, effects, model checks and Fisher diagnostics."""

from importlib import import_module

_EXPORT_MODULES = {
    "CentileCalibration": ".distribution",
    "CentileCurves": ".distribution",
    "QuantileResiduals": ".distribution",
    "WormPanel": ".distribution",
    "WormPlotData": ".distribution",
    "centile_calibration": ".distribution",
    "prepare_centiles": ".distribution",
    "prepare_worm_plot": ".distribution",
    "quantile_residuals": ".distribution",
    "center_additive_prediction": ".contributions",
    "explain_additive_prediction": ".contributions",
    "term_importance_table": ".contributions",
    "DistributionalEffects": ".effects",
    "combine_distributional_effects": ".effects",
    "prepare_distributional_effects": ".effects",
    "FisherDiagnostics": ".fisher",
    "FisherGroupSummary": ".fisher",
    "FisherInformationResult": ".fisher",
    "FisherSnapshot": ".fisher",
    "SurvivalEffects": ".survival",
    "prepare_survival_effects": ".survival",
    "SurvivalResiduals": ".survival",
    "survival_residuals": ".survival",
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
