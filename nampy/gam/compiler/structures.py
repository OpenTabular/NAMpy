"""Internal mutable compilation state bridging specs, fitting, and prediction.

Compiled objects contain NumPy arrays, mutable metadata, callbacks, and terms
that are completed across compiler stages. They are not public immutable
results; fitted public artifacts live under :mod:`nampy.gam.results`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..coefficients import (
    CoefficientTransform,
    CoordinatewiseCoefficientTransform,
    IdentityCoefficientTransform,
)
from ..observations import IdentityObservationTransform, ObservationTransform
from .contracts import (
    ByVariableInfo,
    CoefficientMap,
    SideConditionPolicy,
    TermFeatureInfo,
)


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
    # Set by ``compile_model`` once predictor-local terms are assembled into
    # the model-wide coefficient layout.  Labels are intentionally not used as
    # identity: distributional models may repeat the same formula term in
    # several linear predictors.
    predictor_index: int = 0
    predictor_name: str = "predictor_0"
    predictor_indices: tuple[int, ...] = (0,)
    full_coef_indices: np.ndarray | None = None
    predict_fn: Callable[..., np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )
    derivative_fn: Callable[..., np.ndarray] | None = field(
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
    positive_coefficient_mask: np.ndarray | None = None
    coefficient_transform: CoefficientTransform | None = None

    def __post_init__(self) -> None:
        width = int(np.asarray(self.basis_train).shape[1])
        if self.full_coef_indices is not None:
            full_indices = np.asarray(self.full_coef_indices, dtype=int).reshape(-1)
            if full_indices.shape != (width,):
                raise ValueError(
                    f"Compiled term {self.label!r} full coefficient indices have "
                    f"shape {full_indices.shape}, expected {(width,)}."
                )
            self.full_coef_indices = full_indices.copy()
        mask = (
            np.zeros(width, dtype=bool)
            if self.positive_coefficient_mask is None
            else np.asarray(self.positive_coefficient_mask, dtype=bool).reshape(-1)
        )
        if mask.shape != (width,):
            raise ValueError(
                f"Compiled term {self.label!r} coefficient mask has shape "
                f"{mask.shape}, expected {(width,)}."
            )
        self.positive_coefficient_mask = mask.copy()
        if self.coefficient_transform is None:
            self.coefficient_transform = (
                IdentityCoefficientTransform(width)
                if not np.any(mask)
                else CoordinatewiseCoefficientTransform(mask, positive_map="exp")
            )
        elif int(self.coefficient_transform.size) != width:
            raise ValueError(
                f"Compiled term {self.label!r} coefficient transform has size "
                f"{self.coefficient_transform.size}, expected {width}."
            )

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

    def prediction_parameterization_matrix(self, X_new):
        if self.predict_fn is None:
            raise RuntimeError(
                f"Compiled term {self.label!r} has no prediction callback."
            )
        basis = np.asarray(self.predict_fn(X_new), dtype=np.float64)
        pred_basis_map = dict(self.metadata or {}).get("prediction_basis_map", None)
        if pred_basis_map is not None:
            basis = basis @ np.asarray(pred_basis_map, dtype=np.float64)
        if self.basis_transform is not None:
            basis = basis @ np.asarray(self.basis_transform, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError(
                f"Predict matrix for compiled term {self.label!r} must be 2D, got {basis.shape}."
            )
        return basis

    def derivative_matrix(self, X_new=None, *, order: int = 1):
        """Evaluate this term's derivative in its compiled coefficient space."""
        if self.derivative_fn is None:
            raise NotImplementedError(
                f"Compiled term {self.label!r} has no derivative provider."
            )
        basis = np.asarray(
            self.derivative_fn(X_new=X_new, order=order), dtype=np.float64
        )
        if self.predict_coefficient_map is not None:
            coefficient_map = np.asarray(
                self.predict_coefficient_map, dtype=np.float64
            )
            if basis.shape[1] == coefficient_map.shape[0]:
                basis = basis @ coefficient_map
        if self.basis_transform is not None:
            basis_transform = np.asarray(self.basis_transform, dtype=np.float64)
            if basis.shape[1] == basis_transform.shape[0]:
                basis = basis @ basis_transform
        expected_width = int(self.basis_train.shape[1])
        if basis.ndim != 2 or basis.shape[1] != expected_width:
            raise ValueError(
                f"Derivative matrix for {self.label!r} has shape {basis.shape}; "
                f"expected (_, {expected_width})."
            )
        return basis


@dataclass
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
    positive_coefficient_mask: np.ndarray | None = None
    coefficient_transform: CoefficientTransform | None = None

    def __post_init__(self) -> None:
        width = int(self.n_coef)
        mask = (
            np.zeros(width, dtype=bool)
            if self.positive_coefficient_mask is None
            else np.asarray(self.positive_coefficient_mask, dtype=bool).reshape(-1)
        )
        if mask.shape != (width,):
            raise ValueError(
                f"Compiled predictor {self.name!r} coefficient mask has shape "
                f"{mask.shape}, expected {(width,)}."
            )
        self.positive_coefficient_mask = mask.copy()
        if self.coefficient_transform is None:
            self.coefficient_transform = (
                IdentityCoefficientTransform(width)
                if not np.any(mask)
                else CoordinatewiseCoefficientTransform(mask, positive_map="exp")
            )
        elif int(self.coefficient_transform.size) != width:
            raise ValueError(
                f"Compiled predictor {self.name!r} coefficient transform has size "
                f"{self.coefficient_transform.size}, expected {width}."
            )

    @property
    def prediction_has_intercept(self) -> bool:
        if not bool(self.has_intercept):
            return False
        for term in self.compiled_terms:
            if bool(getattr(term, "metadata", {}).get("prediction_replaces_intercept")):
                return False
        return True

    def build_new_matrix(self, X_new, *, skip_term_ids=()):
        if len(self.compiled_terms) == 0:
            return np.empty((len(X_new), 0), dtype=np.float64)

        blocks = []
        skipped = {str(value) for value in skip_term_ids}
        for term in self.compiled_terms:
            if str(getattr(term, "term_id", "")) in skipped:
                blocks.append(
                    np.zeros(
                        (len(X_new), np.asarray(term.basis_train).shape[1]),
                        dtype=np.float64,
                    )
                )
                continue
            use_raw = bool(
                getattr(term, "metadata", {}).get("expose_raw_prediction_basis")
            )
            if use_raw:
                basis = np.asarray(
                    term.prediction_parameterization_matrix(X_new), dtype=np.float64
                )
            else:
                basis = np.asarray(term.predict_matrix(X_new), dtype=np.float64)
            blocks.append(basis)

        return np.column_stack(blocks)


@dataclass
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
    predictor_full_indices: tuple[np.ndarray, ...] = ()
    smoothing_override_modes: list[str | None] = field(default_factory=list)
    smoothing_override_values: np.ndarray | None = None
    side_condition_reports: tuple[dict[str, Any], ...] | None = None
    fit_to_prediction_parameterization_map: np.ndarray | None = None
    positive_coefficient_mask: np.ndarray | None = None
    coefficient_transform: CoefficientTransform | None = None
    observation_transform: ObservationTransform | None = None

    def __post_init__(self) -> None:
        width = int(
            sum(int(pred.n_coef) + int(bool(pred.has_intercept)) for pred in self.predictors)
        )
        if not self.predictors and self.positive_coefficient_mask is not None:
            width = int(np.asarray(self.positive_coefficient_mask).size)
        mask = (
            np.zeros(width, dtype=bool)
            if self.positive_coefficient_mask is None
            else np.asarray(self.positive_coefficient_mask, dtype=bool).reshape(-1)
        )
        if mask.shape != (width,):
            raise ValueError(
                "Compiled model coefficient mask has shape "
                f"{mask.shape}, expected {(width,)}."
            )
        self.positive_coefficient_mask = mask.copy()
        if not self.predictor_full_indices:
            self.predictor_full_indices = tuple(
                np.arange(int(sl.start), int(sl.stop), dtype=int)
                for sl in self.predictor_full_slices
            )
        else:
            self.predictor_full_indices = tuple(
                np.asarray(indices, dtype=int).reshape(-1).copy()
                for indices in self.predictor_full_indices
            )
        if self.coefficient_transform is None:
            self.coefficient_transform = (
                IdentityCoefficientTransform(width)
                if not np.any(mask)
                else CoordinatewiseCoefficientTransform(mask, positive_map="exp")
            )
        elif int(self.coefficient_transform.size) != width:
            raise ValueError(
                "Compiled model coefficient transform has size "
                f"{self.coefficient_transform.size}, expected {width}."
            )
        n_obs = int(np.asarray(self.design_matrix).shape[0])
        if self.observation_transform is None:
            self.observation_transform = IdentityObservationTransform(n_obs)
        elif int(self.observation_transform.size) != n_obs:
            raise ValueError(
                "Compiled model observation transform has size "
                f"{self.observation_transform.size}, expected {n_obs}."
            )

    def build_new_matrix(self, X_new, *, skip_term_ids=()):
        if len(self.predictors) == 0:
            return np.empty((len(X_new), 0), dtype=np.float64)

        blocks = [
            np.asarray(
                pred.build_new_matrix(X_new, skip_term_ids=skip_term_ids),
                dtype=np.float64,
            )
            for pred in self.predictors
        ]
        return (
            np.column_stack(blocks)
            if blocks
            else np.empty((len(X_new), 0), dtype=np.float64)
        )


__all__ = [
    "CompiledPenalty",
    "CompiledPredictor",
    "CompiledTerm",
    "CompiledModel",
]
