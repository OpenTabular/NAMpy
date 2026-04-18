"""Stable compiled-model contracts bridging specs, engine, and prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .contracts import (
    ByVariableInfo,
    CoefficientMap,
    SideConditionPolicy,
    TermFeatureInfo,
)


@dataclass
class PenaltySpec:
    matrix: np.ndarray
    smoothing_id: str | None = None
    kind: str = "smooth"
    rank: int | None = None
    null_space_dim: int | None = None
    is_null_space_penalty: bool = False
    sp_mode: str | None = None
    sp_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledPenalty:
    label: str
    coef_slice: slice
    matrix: np.ndarray
    smoothing_index: int
    term_index: int = -1
    smoothing_id: str | None = None
    kind: str = "smooth"
    rank: int | None = None
    null_space_dim: int | None = None
    is_null_space_penalty: bool = False
    sp_mode: str | None = None
    sp_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledTerm:
    label: str
    coef_slice: slice
    basis_train: np.ndarray
    predict_fn: Callable[..., np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )
    predict_coefficient_map: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    basis_transform: np.ndarray | None = None
    coefficient_maps: tuple[CoefficientMap, ...] = field(default_factory=tuple)
    feature_info: TermFeatureInfo = field(default_factory=TermFeatureInfo)
    by_variable_info: ByVariableInfo = field(default_factory=ByVariableInfo)
    side_condition_policy: SideConditionPolicy = field(
        default_factory=SideConditionPolicy
    )
    kept_columns: np.ndarray | None = None
    deleted_columns: np.ndarray | None = None
    smoothing_indices: list[int] = field(default_factory=list)
    smoothing_ids: list[str | None] = field(default_factory=list)
    n_penalties: int = 0
    term_type: str = "smooth"
    basis_name: str = "unknown"
    term_id: str = ""
    smoothing_group_id: str | None = None
    penalty_specs: tuple = field(default_factory=tuple)
    constructor_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def smoothing_id(self):
        return self.smoothing_group_id

    def predict_matrix(self, X_new):
        if self.predict_fn is None:
            raise RuntimeError(
                f"Compiled term {self.label!r} has no prediction callback."
            )
        basis = np.asarray(self.predict_fn(X_new), dtype=np.float64)
        if self.predict_coefficient_map is not None:
            basis = basis @ np.asarray(self.predict_coefficient_map, dtype=np.float64)
        if self.basis_transform is not None:
            basis = basis @ np.asarray(self.basis_transform, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError(
                f"Predict matrix for compiled term {self.label!r} must be 2D, got {basis.shape}."
            )
        expected_width = int(self.basis_train.shape[1])
        if basis.shape[1] != expected_width:
            raise ValueError(
                f"Predict matrix for compiled term {self.label!r} has width {basis.shape[1]}, "
                f"expected {expected_width}."
            )
        return basis


@dataclass(frozen=True)
class CompiledPredictor:
    name: str
    design_matrix: np.ndarray
    compiled_terms: tuple
    compiled_penalties: tuple
    smoothing_parameter_map: dict[str, int]
    n_coef: int
    n_smoothing_params: int
    has_intercept: bool = False
    term_index_map: dict[str, int] = field(default_factory=dict)
    side_condition_Q: np.ndarray | None = None
    smoothing_override_modes: list[str | None] = field(default_factory=list)
    smoothing_override_values: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_new_matrix(self, X_new):
        if len(self.compiled_terms) == 0:
            return np.empty((len(X_new), 0), dtype=np.float64)

        blocks = []
        for term in self.compiled_terms:
            blocks.append(np.asarray(term.predict_matrix(X_new), dtype=np.float64))

        return np.column_stack(blocks)


@dataclass(frozen=True)
class CompiledModel:
    predictors: tuple[CompiledPredictor, ...]
    design_matrix: np.ndarray
    compiled_terms: tuple[CompiledTerm, ...]
    compiled_penalties: tuple[CompiledPenalty, ...]
    metadata: dict[str, Any]
    n_coef: int
    n_smoothing_params: int
    predictor_full_slices: tuple[slice, ...]
    coef_reduced_to_full_idx: np.ndarray
    smoothing_override_modes: list[str | None] = field(default_factory=list)
    smoothing_override_values: np.ndarray | None = None
    side_condition_reports: tuple[dict[str, Any], ...] | None = None

    def build_new_matrix(self, X_new):
        if len(self.predictors) == 0:
            return np.empty((len(X_new), 0), dtype=np.float64)

        blocks = [
            np.asarray(pred.build_new_matrix(X_new), dtype=np.float64)
            for pred in self.predictors
        ]
        return (
            np.column_stack(blocks)
            if blocks
            else np.empty((len(X_new), 0), dtype=np.float64)
        )


__all__ = [
    "PenaltySpec",
    "CompiledPenalty",
    "CompiledPredictor",
    "CompiledTerm",
    "CompiledModel",
]
