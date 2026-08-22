"""Canonical post-fit artifacts shared by prediction, diagnostics, and results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..compiler.structures import CompiledModel
from .fit_result import GAMFitResult

if TYPE_CHECKING:
    from ..fit.state import FitCoreSolution


@dataclass(frozen=True)
class GAMResult:
    """Canonical fitted-state aggregate.

    The aggregate is intentionally staged: compilation, solving, and summary
    construction happen at different points in the fit pipeline.  The outer
    object is replaced at each transition; the compiled terms themselves are
    internal mutable compilation state.
    """

    compiled_model: CompiledModel | None = None
    fit_core_solution: FitCoreSolution | None = None
    fit_summary: GAMFitResult | None = None

    def with_compiled_model(self, compiled_model: CompiledModel):
        return replace(self, compiled_model=compiled_model)

    def with_fit_solution(self, fit_core_solution: FitCoreSolution, **kwargs):
        # A summary is derived from one exact core solution.  Replacing the
        # solution must invalidate it; the orchestration/public API rebuilds
        # the summary after all post-processing is complete.
        kwargs.setdefault("fit_summary", None)
        return replace(self, fit_core_solution=fit_core_solution, **kwargs)

    def require_compiled_model(self) -> CompiledModel:
        if self.compiled_model is None:
            raise RuntimeError("Model has no compiled design; fit the model first.")
        return self.compiled_model

    def require_fit_core_solution(self) -> FitCoreSolution:
        if self.fit_core_solution is None:
            raise RuntimeError("Model has no fitted core solution.")
        return self.fit_core_solution

    def require_fit_summary(self) -> GAMFitResult:
        if self.fit_summary is None:
            raise RuntimeError("Model has no fitted result summary.")
        return self.fit_summary


__all__ = ["GAMResult"]
