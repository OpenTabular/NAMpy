from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PenaltyDefinition:
    """
    Declarative penalty specification returned by a constructed smooth.
    """

    matrix: np.ndarray
    smoothing_id: str | None = None
    kind: str = "smooth"
    rank: int | None = None
    null_space_dim: int | None = None
    is_null_space_penalty: bool = False

    # term-level smoothing-parameter override
    # None       -> no explicit term-level override
    # "fixed"    -> fixed at sp_value
    # "estimate" -> free/estimated by outer optimizer
    sp_mode: str | None = None
    sp_value: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PenaltyBlock:
    """
    One quadratic penalty block acting on a coefficient slice.
    """

    label: str
    coef_slice: slice
    matrix: np.ndarray
    smoothing_index: int
    smoothing_id: str | None = None
    kind: str = "smooth"
    rank: int | None = None
    null_space_dim: int | None = None
    is_null_space_penalty: bool = False

    sp_mode: str | None = None
    sp_value: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TermBlock:
    """
    Compiled constructed smooth block inside one linear predictor.
    """

    label: str
    coef_slice: slice
    smooth: object
    basis_train: np.ndarray

    # prediction-side transform: new-data basis is
    # smooth.predict_matrix(X_new) @ basis_transform
    basis_transform: np.ndarray | None = None

    # bookkeeping for side-condition column deletions
    original_n_coef: int | None = None
    kept_columns: np.ndarray | None = None
    deleted_columns: np.ndarray | None = None

    smoothing_indices: list[int] = field(default_factory=list)
    smoothing_ids: list[str | None] = field(default_factory=list)
    n_penalties: int = 0
    term_type: str = "smooth"
    basis_name: str = "unknown"
    by_variable: str | None = None
    term_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictorDesign:
    """
    Compiled design for one linear predictor.
    """

    name: str
    term_blocks: list
    penalty_blocks: list
    matrix_train: np.ndarray
    n_coef: int
    n_smoothing_params: int
    smoothing_id_map: dict[str, int]

    # per-underlying-smoothing-parameter term-level override state
    smoothing_override_modes: list[str | None] = field(default_factory=list)
    smoothing_override_values: np.ndarray | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def build_new_matrix(self, X_new):
        if len(self.term_blocks) == 0:
            return np.empty((len(X_new), 0), dtype=np.float64)

        blocks = []
        for tb in self.term_blocks:
            B = np.asarray(tb.smooth.predict_matrix(X_new), dtype=np.float64)
            C = tb.basis_transform
            if C is not None:
                B = B @ C
            blocks.append(B)

        return (
            np.column_stack(blocks)
            if blocks
            else np.empty((len(X_new), 0), dtype=np.float64)
        )