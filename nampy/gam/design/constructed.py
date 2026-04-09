from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ConstructedTerm:
    """
    Stage-3 representation of a fitted term, bridging runtime semantics into
    the stage-4 predictor compiler.

    A ``ConstructedTerm`` wraps one runtime term after its basis has been fitted
    and any delegated by-variable or constraint handling has been applied by the
    stage-3 wrapper.  The stage-4 compiler treats it as an opaque block: it
    reads ``train_design_matrix``, ``penalty_specs``, and calls
    ``predict_matrix(X_new)`` without knowing anything about the underlying
    basis family.

    The canonical rule for new-data prediction is::

        predict_matrix(X_new)

    which internally calls either the runtime's ``transform_new`` directly, or a
    ``_predict_fn`` closure that applies any wrapper-level non-coefficient-space
    logic (for example by-scaling) on top of it. Any linear map from the raw
    prediction basis into the fitted coefficient coordinates is carried
    explicitly by ``predict_coefficient_map``.

    After stage 5 (``apply_global_side_conditions``), the caller must apply
    ``CompiledTerm.basis_transform`` on top of the result of ``predict_matrix``.
    That composed application is implemented in ``CompiledPredictor.build_new_matrix``.
    """

    label: str
    term_id: str
    runtime: object
    train_design_matrix: np.ndarray
    penalty_specs: tuple = field(default_factory=tuple)
    basis_name: str = "unknown"
    term_type: str = "smooth"
    smoothing_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fit_constraint_operator: np.ndarray | None = None
    fit_coefficient_map: np.ndarray | None = None
    predict_coefficient_map: np.ndarray | None = None
    constraints_absorbed: bool = True
    prediction_offset: np.ndarray | None = None
    original_design_matrix: np.ndarray | None = None
    constructor_metadata: dict[str, Any] = field(default_factory=dict)
    _predict_fn: Callable | None = field(default=None, repr=False, compare=False)

    @property
    def n_coef(self) -> int:
        return int(self.train_design_matrix.shape[1])

    @property
    def original_n_coef(self) -> int | None:
        if self.original_design_matrix is None:
            return None
        return int(self.original_design_matrix.shape[1])

    @property
    def transformed_n_coef(self) -> int:
        return int(self.train_design_matrix.shape[1])

    def predict_matrix(self, X_new):
        if self._predict_fn is None:
            M = np.asarray(self.runtime.transform_new(X_new), dtype=np.float64)
        else:
            M = np.asarray(self._predict_fn(X_new), dtype=np.float64)

        C_pred = self.predict_coefficient_map
        if C_pred is not None:
            M = M @ np.asarray(C_pred, dtype=np.float64)

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


def build_term_matrix(term: ConstructedTerm, X_new, return_offset=False):
    Xp = term.predict_matrix(X_new)
    if return_offset:
        return Xp, term.prediction_offset
    return Xp


__all__ = ["ConstructedTerm", "build_term_matrix"]
