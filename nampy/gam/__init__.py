"""
mgcv-aligned GAM subsystem: families, fitting, smoothness, and parity checks.

Fit pipeline
------------
Stage 1  Formula / spec layer
         gam/formula/        — parse formulas, build LinearPredictorSpec / TermSpec

Stage 2  Runtime term materialization
         gam/runtime/        — instantiate TermSpecs into fitted runtime terms
         gam/smooths/*       — canonical per-family basis construction and penalties

Stage 3  Term construction wrapper
         gam/design/constructors.py — fit runtime terms, handle delegated by/constraints,
                                      wrap into ConstructedTerm

Stage 4  Predictor compilation
         gam/design/compiler.py    — assemble ConstructedTerms into CompiledPredictor,
                                     assign coef slices and smoothing-parameter ids

Stage 5  Predictor-wide side conditions
         gam/constraints/identifiability.py — centre terms, drop redundant columns,
                                              update basis_transform to canonical form

Stage 6  Model fitting
         gam/fit/            — PIRLS / Gaussian solvers, smoothness optimisation

Stage 7  Prediction / parity / diagnostics
         gam/predict/        — lpmatrix, response/link/term predictions
         gam/parity/         — snapshot build, load, and comparison against mgcv
         gam/diagnostics/    — summaries and plots

Architectural invariants (see gam/ARCHITECTURE.md for full details)
--------------------------------------------------------------------
6.2  CompiledTerm.basis_transform is the only coefficient map used at prediction.
6.3  If a coefficient transform T is applied, penalties become T.T @ S @ T.
6.6  Exempt terms (random effects, factor smooths) still span predictor space.
6.7  Zero-width terms are dropped from the final compiled design.
"""

from . import families, fit, parity, smoothing_selection

# Stable user-facing entry points only.  Internal fit-subsystem symbols are
# accessible via `nampy.gam.fit.*` and are not re-exported here.
from .fit import (
    FitCoreSolution,
    fit_model_core,
    solve_fit,
)

__all__ = [
    "families",
    "fit",
    "parity",
    "smoothing_selection",
    "fit_model_core",
    "solve_fit",
    "FitCoreSolution",
]
