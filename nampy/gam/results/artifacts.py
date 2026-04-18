"""Canonical post-fit artifacts shared by prediction, diagnostics, and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..compiler.structures import CompiledModel
from .fit_result import GAMFitResult

if TYPE_CHECKING:
    from ..fit.state import FitCoreSolution


@dataclass(frozen=True)
class GAMResult:
    compiled_model: CompiledModel
    fit_core_solution: FitCoreSolution
    fit_summary: GAMFitResult


__all__ = ["GAMResult"]
