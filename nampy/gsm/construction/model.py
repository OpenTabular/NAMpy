from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ConstructedSmooth:
    """
    Framework smoothCon-style constructed smooth contract.

    This is the fit-time smooth object the design compiler should consume.
    """

    label: str
    runtime: object

    X: np.ndarray
    penalty_definitions: list = field(default_factory=list)

    basis_name: str = "unknown"
    term_type: str = "smooth"
    by_variable: str | None = None
    smoothing_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    fit_constraint: np.ndarray | None = None
    predict_constraint: np.ndarray | None = None
    constraints_absorbed: bool = True

    prediction_offset: np.ndarray | None = None
    X0: np.ndarray | None = None

    constructor_metadata: dict[str, Any] = field(default_factory=dict)

    _predict_fn: Callable | None = field(default=None, repr=False, compare=False)

    @property
    def n_coef(self) -> int:
        return int(self.X.shape[1])

    def predict_matrix(self, X_new):
        if self._predict_fn is None:
            M = np.asarray(self.runtime.transform_new(X_new), dtype=np.float64)
        else:
            M = np.asarray(self._predict_fn(X_new), dtype=np.float64)

        if M.ndim != 2:
            raise ValueError(
                f"Predict matrix for smooth {self.label!r} must be 2D, got {M.shape}."
            )
        if M.shape[1] != self.n_coef:
            raise ValueError(
                f"Predict matrix for smooth {self.label!r} has width {M.shape[1]}, "
                f"but fitted width is {self.n_coef}."
            )
        return M


def predict_mat(smooth: ConstructedSmooth, X_new, return_offset=False):
    """
    PredictMat analogue.

    Returns the matrix that maps the smooth's coefficients to the smooth values
    at X_new. If return_offset=True, also returns any smooth-specific offset.
    """
    Xp = smooth.predict_matrix(X_new)
    if return_offset:
        return Xp, smooth.prediction_offset
    return Xp