"""Public entry points for the mgcv-aligned GAM subsystem."""

from .fit import (
    FitCoreSolution,
    fit_model_core,
    solve_fit,
)
from .model import GAM

__all__ = [
    "GAM",
    "fit_model_core",
    "solve_fit",
    "FitCoreSolution",
]
