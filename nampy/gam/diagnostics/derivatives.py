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
from ..term_labels import mgcv_term_display_label


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
    """Evaluate a univariate smooth derivative on its additive-predictor scale.

    Parameters
    ----------
    model : GAM
        Fitted statistical GAM.
    X : array-like or DataFrame, optional
        Evaluation rows; None uses the term's fitted evaluation data.
    smooth_number : int, default=1
        One-based smooth index, excluding parametric terms.
    deriv : {1, 2}, default=1
        Derivative order: slope or curvature.

    Returns
    -------
    SmoothDerivativeResult
        Derivative, Bayesian standard error, derivative design matrix, term
        label, and derivative order. The ``d`` and ``se_d`` properties are aliases.

    Notes
    -----
    A pointwise derivative band can help locate increasing or decreasing
    regions. It is conditional on the fitted basis, smoothing and covariance;
    it is not a simultaneous guarantee across an entire curve. Derivatives of
    a response mean under a nonlinear inverse link require the chain rule.
    Only univariate smooths and first/second derivatives are supported here;
    a joint interaction surface needs a separate partial-derivative analysis.
    """
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
        term_label=mgcv_term_display_label(term),
        order=deriv,
    )


def prepare_derivative_plot(model, *, X=None, smooth_number=1, deriv=1, level=0.95):
    """Prepare a sorted derivative curve with a pointwise Bayesian band.

    Uses observed rows by default, preserving the training-only SCAM
    derivative contract. The horizontal axis is the smooth's covariate,
    including any fitted formula transformation.
    """
    from scipy.stats import norm

    if not 0 < float(level) < 1:
        raise ValueError("level must be strictly between zero and one.")
    result = smooth_derivative(model, X=X, smooth_number=smooth_number, deriv=deriv)
    smooths = [t for t in _term_blocks_seq(model) if t.term_type != "parametric"]
    term = smooths[int(smooth_number) - 1]
    idx = int(term.feature_info.feature_indices[0])
    rows = np.asarray(model.X_ if X is None else X)
    x = np.asarray(rows[:, idx], dtype=float)
    order = np.argsort(x, kind="stable")
    fit, se = result.derivative[order], result.se[order]
    width = norm.ppf((1 + float(level)) / 2) * se
    names = getattr(model, "formula_feature_columns_", None)
    xlab = str(names[idx]) if names is not None else f"x{idx}"
    state = getattr(model, "formula_preprocess_state_", None) or {}
    for expression in state.get("formula_expression_columns", []):
        if expression["hidden_name"] == xlab:
            xlab = expression["expr"]
            break
    return {
        "pd": [
            {
                "kind": "1d",
                "x": x[order],
                "fit": fit,
                "se_fit": se,
                "ll": (fit - width)[:, None],
                "ul": (fit + width)[:, None],
                "raw": x,
                "plot_ci": True,
                "plot_me": True,
                "scheme": 1,
                "zero_line": True,
                "xlab": xlab,
                "ylab": f"Derivative (order {deriv})",
                "main": result.term_label,
                "level": float(level),
            }
        ],
        "trans": lambda values: values,
        "shift": 0.0,
        "ylim": None,
        "select": None,
        "partial_resids": False,
        "jit": False,
        "rug_default": False,
    }


__all__ = ["SmoothDerivativeResult", "smooth_derivative", "prepare_derivative_plot"]
