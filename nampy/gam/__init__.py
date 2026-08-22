"""Public entry points for the mgcv-aligned GAM subsystem.

Layer map (data flows top to bottom)::

    formula/, specs/          parse formulas into TermSpec objects
    smooths/, splines/        runtime terms: fitted bases and penalties
                              (splines/basis mirrors mgcv C; smooths/ owns
                              all basis semantics)
    compiler/                 TermSpec -> CompiledTerm -> CompiledPredictor
                              -> CompiledModel
    constraints/              identifiability side conditions
    fit/                      solve coefficients at given smoothing params
      fit/solvers/            PIRLS / stacked-QR / general-family Newton
                              (mirrors mgcv gam.fit3/gam.fit5 - do not
                              restructure)
      fit/selection/          smoothing-parameter selection: criteria/
                              (REML/GCV values and derivatives, mirrors
                              mgcv gdi1/gdi2), optimize/ (outer Newton,
                              BFGS, EFS drivers), reparam (mirrors mgcv
                              gam.reparam/Sl.setup - do not restructure)
      fit/postprocess/        covariance / smoothness post-fit corrections
    results/                  FitResult (numeric record), GAMFitResult
                              (presentation view), GAMResult aggregate,
                              snapshots and optimizer traces
    predict/, inference/,     prediction, summary/anova/loglik algebra,
    diagnostics/, parity/     residual checks, mgcv comparison tools
    linalg/                   shared numerics (QR, eigen, rank, reindexing)
    model/                    the user-facing GAM facade
    model_state.py            canonical fitted-state accessors
    workspace.py              per-fit transient FitWorkspace (model._ws)

Fitted state lives on ``model.gam_result_`` (compiled model, core solution,
fit summary); solver scratch lives on ``model._ws`` and is never pickled.
Modules marked "mirrors mgcv" follow the vendored upstream R/C sources
line-by-line where practical - treat upstream as the specification before
restructuring anything inside them.
"""

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
