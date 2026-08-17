"""Public fit-core entry points for the mgcv-aligned GAM subsystem."""

from .fit import (
    FitCoreSolution,
    fit_model_core,
    solve_fit,
)

__all__ = [
    "fit_model_core",
    "solve_fit",
    "FitCoreSolution",
]
