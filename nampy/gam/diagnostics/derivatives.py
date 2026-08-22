"""Smooth derivative extraction following ``scam::derivative.scam``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model_state import (
    _coef_full,
    _require_fitted,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)


@dataclass(frozen=True)
class SmoothDerivativeResult:
    """Derivative values and Bayesian standard errors for one smooth term."""

    derivative: np.ndarray
    se: np.ndarray
    derivative_matrix: np.ndarray
    term_label: str
    order: int

    @property
    def d(self) -> np.ndarray:
        return self.derivative

    @property
    def se_d(self) -> np.ndarray:
        return self.se


def smooth_derivative(model, *, X=None, smooth_number: int = 1, deriv: int = 1):
    """Evaluate a term-owned univariate derivative and Bayesian uncertainty."""
    _require_fitted(model)
    deriv = int(deriv)
    if deriv not in {1, 2}:
        raise ValueError("deriv can be either 1 or 2")
    smooths = [
        term
        for term in _term_blocks_seq(model)
        if str(getattr(term, "term_type", "")) != "parametric"
    ]
    index = int(smooth_number) - 1
    if index < 0 or index >= len(smooths):
        raise IndexError(
            f"smooth_number must be between 1 and {len(smooths)}, got {smooth_number}."
        )
    term = smooths[index]
    if len(getattr(term.feature_info, "feature_indices", ())) != 1:
        raise NotImplementedError(
            "Smooth derivative extraction currently handles only 1D smooths."
        )
    Xd = np.asarray(term.derivative_matrix(X, order=deriv), dtype=np.float64)
    full_indices = _term_full_coefficient_indices(model, term)
    beta = np.asarray(_coef_full(model), dtype=np.float64)[full_indices]
    covariance = np.asarray(model._select_cov("bayes"), dtype=np.float64)
    Vp = covariance[np.ix_(full_indices, full_indices)]
    values = np.asarray(Xd @ beta, dtype=np.float64)
    variance = np.einsum("ij,jk,ik->i", Xd, Vp, Xd)
    return SmoothDerivativeResult(
        derivative=values,
        se=np.sqrt(np.maximum(variance, 0.0)),
        derivative_matrix=Xd,
        term_label=str(term.label),
        order=deriv,
    )


__all__ = ["SmoothDerivativeResult", "smooth_derivative"]
