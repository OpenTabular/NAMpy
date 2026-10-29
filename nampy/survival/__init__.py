"""Shared survival contracts, predictions, diagnostics and evaluation."""

from .bootstrap import (
    SurvivalBootstrapError,
    SurvivalBootstrapFailure,
    SurvivalBootstrapResult,
    bootstrap_survival,
)
from .contracts import PiecewiseExponentialDesign, SurvivalComponents, SurvivalTarget
from .data import piecewise_exponential_data
from .diagnostics import SurvivalResiduals, survival_residuals
from .evaluation import (
    BrierScore,
    CensoringDistribution,
    ConcordanceResult,
    SurvivalCurve,
    brier_score,
    concordance_index,
    survival_curve,
)
from .interpretation import SurvivalEffects, prepare_survival_effects
from .longitudinal import (
    LongitudinalSurvivalTarget,
    SurvivalTrajectory,
    predict_trajectory,
)
from .prediction import SurvivalPrediction, survival_from_hazard
from .uncertainty import (
    SurvivalCurveInterval,
    SurvivalEffectInterval,
    SurvivalEffectSamples,
    combine_survival_predictions,
    prepare_survival_curve_interval,
    prepare_survival_effect_interval,
    prepare_survival_hazard_interval,
    prepare_survival_rmst_interval,
)

__all__ = [
    "BrierScore",
    "SurvivalBootstrapError",
    "SurvivalBootstrapFailure",
    "SurvivalBootstrapResult",
    "bootstrap_survival",
    "CensoringDistribution",
    "ConcordanceResult",
    "PiecewiseExponentialDesign",
    "SurvivalComponents",
    "SurvivalCurve",
    "SurvivalPrediction",
    "SurvivalEffects",
    "SurvivalCurveInterval",
    "SurvivalEffectInterval",
    "SurvivalEffectSamples",
    "SurvivalResiduals",
    "SurvivalTarget",
    "LongitudinalSurvivalTarget",
    "SurvivalTrajectory",
    "predict_trajectory",
    "brier_score",
    "combine_survival_predictions",
    "concordance_index",
    "piecewise_exponential_data",
    "prepare_survival_curve_interval",
    "prepare_survival_effect_interval",
    "prepare_survival_effects",
    "prepare_survival_hazard_interval",
    "prepare_survival_rmst_interval",
    "survival_curve",
    "survival_from_hazard",
    "survival_residuals",
]
