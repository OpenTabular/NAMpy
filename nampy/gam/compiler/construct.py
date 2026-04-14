"""Compiler-owned smoothCon-like construction."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..constraints.absorption import absorb_explicit_constraints
from ..penalties import normalize_penalty_spec
from ..specs import TermSpec
from .factory import instantiate_term


@dataclass(frozen=True)
class ConstructedSmooth:
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
    transform_applied: bool = False
    skip_centering: bool = False
    prediction_offset: np.ndarray | None = None
    original_design_matrix: np.ndarray | None = None
    constructor_metadata: dict[str, Any] = field(default_factory=dict)
    _predict_fn: Callable | None = field(default=None, repr=False, compare=False)

    @property
    def n_coef(self) -> int:
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


def build_term_matrix(term: ConstructedSmooth, X_new, return_offset=False):
    Xp = term.predict_matrix(X_new)
    if return_offset:
        return Xp, term.prediction_offset
    return Xp


def _copy_penalty_defs(penalty_defs):
    return [copy.copy(p) for p in penalty_defs]


def _set_penalty_matrix_and_meta(pdef, P):
    pdef.matrix = np.asarray(P, dtype=np.float64)
    return normalize_penalty_spec(pdef)


def _extract_runtime_state(runtime):
    B = np.asarray(runtime.basis_train, dtype=np.float64)
    if hasattr(runtime, "get_penalty_definitions"):
        penalty_defs = list(runtime.get_penalty_definitions())
    else:
        penalty_defs = []
    penalty_defs = _copy_penalty_defs(penalty_defs)
    for pdef in penalty_defs:
        _set_penalty_matrix_and_meta(pdef, np.asarray(pdef.matrix, dtype=np.float64))
    return B, penalty_defs


def construct_smooth(
    term_like: TermSpec | Any,
    X: np.ndarray,
    feature_names: list[str],
    *,
    absorb_cons=True,
    apply_by=True,
    null_space_penalty=False,
):
    runtime = instantiate_term(term_like)
    runtime.fit(X, feature_names)
    B, penalty_defs = _extract_runtime_state(runtime)
    runtime_transform_applied = bool(
        getattr(
            runtime,
            "transform_applied",
            getattr(runtime, "constraint_transform", None) is not None,
        )
    )
    runtime_skip_centering = bool(
        getattr(
            runtime,
            "skip_centering",
            getattr(runtime, "constraints_absorbed", False),
        )
    )
    fit_constraint = getattr(runtime, "fit_constraint_matrix", None)
    predict_coefficient_map = getattr(runtime, "predict_coefficient_map", None)
    prediction_offset = getattr(runtime, "prediction_offset", None)
    _by_state = getattr(runtime, "_by_state", None)

    constructor_metadata = {
        "runtime_transform_applied": runtime_transform_applied,
        "runtime_skip_centering": runtime_skip_centering,
        "runtime_constraint_kind": getattr(runtime, "constraint_kind", None),
        "runtime_by_name": _by_state.feature_name if _by_state is not None else None,
        "runtime_by_is_constant": (
            _by_state.is_constant if _by_state is not None else None
        ),
    }

    raw_predict_n_coef = int(B.shape[1])
    constructor_metadata["by_handling"] = (
        "runtime" if getattr(runtime, "by", None) is not None else "none"
    )

    fit_coefficient_map = None
    transform_applied = runtime_transform_applied
    skip_centering = runtime_skip_centering
    predict_coefficient_map_arr = (
        None
        if predict_coefficient_map is None
        else np.asarray(predict_coefficient_map, dtype=np.float64)
    )

    if absorb_cons and (not runtime_transform_applied) and fit_constraint is not None:
        B, penalty_defs, T_fit, n_cons = absorb_explicit_constraints(
            B,
            _copy_penalty_defs(penalty_defs),
            fit_constraint,
        )
        fit_coefficient_map = np.asarray(T_fit, dtype=np.float64)
        if predict_coefficient_map_arr is None:
            predict_coefficient_map_arr = fit_coefficient_map
        expected_shape = (int(raw_predict_n_coef), int(B.shape[1]))
        if predict_coefficient_map_arr.shape != expected_shape:
            raise ValueError(
                f"Predict coefficient map for term {getattr(runtime, 'label', str(runtime))!r} "
                f"has shape {predict_coefficient_map_arr.shape}, expected {expected_shape}."
            )
        transform_applied = True
        skip_centering = True
        constructor_metadata["constraint_absorption"] = "wrapper"
        constructor_metadata["n_constraints_absorbed"] = int(n_cons)
        constructor_metadata["predict_map_source"] = (
            "runtime" if predict_coefficient_map is not None else "fit_coefficient_map"
        )
    else:
        if predict_coefficient_map_arr is not None:
            expected_shape = (int(raw_predict_n_coef), int(B.shape[1]))
            if predict_coefficient_map_arr.shape != expected_shape:
                raise ValueError(
                    f"Predict coefficient map for term {getattr(runtime, 'label', str(runtime))!r} "
                    f"has shape {predict_coefficient_map_arr.shape}, expected {expected_shape}."
                )
        constructor_metadata["constraint_absorption"] = (
            "runtime" if transform_applied else "none"
        )
        constructor_metadata["n_constraints_absorbed"] = None
        constructor_metadata["predict_map_source"] = (
            "runtime" if predict_coefficient_map_arr is not None else "none"
        )

    if null_space_penalty:
        raise NotImplementedError(
            "Generic smoothCon-level null-space penalty insertion is not enabled yet in this wrapper."
        )

    return ConstructedSmooth(
        label=str(getattr(runtime, "label", str(runtime))),
        term_id=str(getattr(runtime, "term_id", "")),
        runtime=runtime,
        train_design_matrix=np.asarray(B, dtype=np.float64),
        penalty_specs=tuple(penalty_defs),
        basis_name=str(getattr(runtime, "basis_name", "unknown")),
        term_type=str(getattr(runtime, "term_type", "smooth")),
        smoothing_id=(
            None
            if getattr(runtime, "smoothing_id", None) is None
            else str(runtime.smoothing_id)
        ),
        metadata=dict(getattr(runtime, "metadata", {}) or {}),
        fit_constraint_operator=(
            None
            if fit_constraint is None
            else np.asarray(fit_constraint, dtype=np.float64)
        ),
        fit_coefficient_map=(
            None
            if fit_coefficient_map is None
            else np.asarray(fit_coefficient_map, dtype=np.float64)
        ),
        predict_coefficient_map=(
            None
            if predict_coefficient_map_arr is None
            else np.asarray(predict_coefficient_map_arr, dtype=np.float64)
        ),
        transform_applied=bool(transform_applied),
        skip_centering=bool(skip_centering),
        prediction_offset=(
            None
            if prediction_offset is None
            else np.asarray(prediction_offset, dtype=np.float64)
        ),
        original_design_matrix=None,
        constructor_metadata=constructor_metadata,
        _predict_fn=None,
    )


__all__ = ["ConstructedSmooth", "build_term_matrix", "construct_smooth"]
