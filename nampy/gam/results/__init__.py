"""Fit outputs and summary-facing result structures."""

from .artifacts import GAMResult
from .fit_result import GAMFitResult, TermFitResult
from .solution import FitResult

__all__ = ["FitResult", "GAMFitResult", "TermFitResult", "GAMResult"]
