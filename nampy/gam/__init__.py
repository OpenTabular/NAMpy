"""mgcv-aligned GAM subsystem for formula parsing, fitting, and prediction."""

from . import families, parity

# Stable user-facing entry points only.  Internal fit-subsystem symbols are
# accessible via `nampy.gam.fit.*` and are not re-exported here.
from .api import GAM
from .fit import (
    FitCoreSolution,
    fit_model_core,
    solve_fit,
)

__all__ = [
    "families",
    "parity",
    "GAM",
    "fit_model_core",
    "solve_fit",
    "FitCoreSolution",
]
