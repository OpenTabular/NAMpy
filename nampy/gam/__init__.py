"""mgcv-aligned GAM subsystem for formula parsing, fitting, and prediction."""

from . import engine, families, parity

# Stable user-facing entry points only.  Internal fit-subsystem symbols are
# accessible via `nampy.gam.fit.*` and are not re-exported here.
from .api import GAM
from .engine import (
    FitCoreSolution,
    fit_model_core,
    solve_fit,
)

__all__ = [
    "engine",
    "families",
    "parity",
    "GAM",
    "fit_model_core",
    "solve_fit",
    "FitCoreSolution",
]
