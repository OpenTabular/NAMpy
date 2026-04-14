"""
mgcv-aligned GAM subsystem: formulas, smooth construction, fitting, and parity.

Fit pipeline
------------
Stage 1  Formula / spec layer
         ``gam/formula/``, ``gam/specs/`` — parse formulas into predictor and term specs

Stage 2  Runtime terms
         ``gam/smooths/*`` — canonical basis semantics, penalties, and runtime transforms

Stage 3  Construction / compilation
         ``gam/compiler/construct.py`` — materialize runtime terms into ``ConstructedSmooth``
         ``gam/compiler/compile_*.py`` — assemble compiled predictors and model-wide design state

Stage 4  Side conditions
         ``gam/constraints/identifiability.py`` — centre terms, drop redundant columns,
         and update coefficient maps

Stage 5  Model fitting
         ``gam/fit/`` — Gaussian, PIRLS, and general-family solvers plus covariance assembly

Stage 6  Prediction / parity / diagnostics
         ``gam/predict/`` — lpmatrix and response/link/term prediction helpers
         ``gam/parity/`` — snapshot and trace comparisons against upstream ``mgcv``
         ``gam/diagnostics/`` — summaries, plots, and checks

--------------------------------------------------------------------
6.2  ``CompiledTerm.basis_transform`` is the only coefficient map used at prediction.
6.3  If a coefficient transform ``T`` is applied, penalties become ``T.T @ S @ T``.
6.6  Exempt terms (random effects, factor smooths) still span predictor space.
6.7  Zero-width terms are dropped from the final compiled design.
"""

from . import engine, families, parity, selection

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
    "selection",
    "GAM",
    "fit_model_core",
    "solve_fit",
    "FitCoreSolution",
]
