"""Compiler-owned smoothCon-like construction."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..constraints.absorption import absorb_explicit_constraints
from ..penalties import normalize_penalty_spec
from ..specs import TermSpec
from .contracts import (
    ByVariableInfo,
    CoefficientMap,
    TermFeatureInfo,
    compatibility_constructor_metadata,
    compose_coefficient_maps,
    default_side_condition_policy,
)
from .factory import instantiate_term
from .structures import CompiledTerm


@dataclass(frozen=True)
class ConstructedSmooth:
    compiled_term: CompiledTerm
    penalty_specs: tuple = field(default_factory=tuple)
    fit_constraint_operator: np.ndarray | None = None
    prediction_offset: np.ndarray | None = None
    original_design_matrix: np.ndarray | None = None

    def __getattr__(self, name: str):
        return getattr(self.compiled_term, name)

    @property
    def n_coef(self) -> int:
        return int(self.compiled_term.basis_train.shape[1])

    @property
    def basis_train(self) -> np.ndarray:
        return np.asarray(self.compiled_term.basis_train, dtype=np.float64)

    @property
    def fit_coefficient_map(self) -> np.ndarray | None:
        for cmap in self.compiled_term.coefficient_maps:
            if cmap.reason == "local_constraint_absorption":
                return np.asarray(cmap.matrix, dtype=np.float64)
        return None

    @property
    def predict_coefficient_map(self) -> np.ndarray | None:
        return compose_coefficient_maps(self.compiled_term.coefficient_maps)

    @property
    def transform_applied(self) -> bool:
        return len(self.compiled_term.coefficient_maps) > 0

    @property
    def skip_centering(self) -> bool:
        policy = self.compiled_term.side_condition_policy
        return bool(policy is not None and policy.skip_centering)

    def predict_matrix(self, X_new):
        return self.compiled_term.predict_matrix(X_new)


def build_term_matrix(term: ConstructedSmooth, X_new, return_offset=False):
    if bool(
        getattr(term.compiled_term, "metadata", {}).get("expose_raw_prediction_basis")
    ):
        Xp = term.compiled_term.prediction_parameterization_matrix(X_new)
    else:
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


def _feature_info_from_runtime(runtime) -> TermFeatureInfo:
    feature_names = tuple(str(v) for v in getattr(runtime, "_feature_names", ()) or ())
    if len(feature_names) == 0 and getattr(runtime, "_feature_name", None) is not None:
        feature_names = (str(runtime._feature_name),)

    feature_indices: tuple[int, ...] = ()
    if getattr(runtime, "_feature_indices", None) is not None:
        feature_indices = tuple(int(v) for v in runtime._feature_indices)
    elif getattr(runtime, "_feature_index", None) is not None:
        feature_indices = (int(runtime._feature_index),)

    metric_feature_indices = tuple(
        int(v) for v in (getattr(runtime, "_metric_feature_indices", None) or ())
    )
    factor_feature_indices = tuple(
        int(v) for v in (getattr(runtime, "_factor_feature_indices", None) or ())
    )
    return TermFeatureInfo(
        feature_names=feature_names,
        feature_indices=feature_indices,
        metric_feature_indices=metric_feature_indices,
        factor_feature_indices=factor_feature_indices,
    )


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
    by_variable_info = ByVariableInfo(
        name=None if _by_state is None else _by_state.feature_name,
        is_constant=None if _by_state is None else _by_state.is_constant,
        handling="runtime" if getattr(runtime, "by", None) is not None else "none",
    )
    side_condition_policy = default_side_condition_policy(
        term_type=str(getattr(runtime, "term_type", "smooth")),
        runtime_skip_centering=runtime_skip_centering,
        by_variable_info=by_variable_info,
    )
    runtime_metadata = getattr(runtime, "metadata", None)
    if isinstance(runtime_metadata, dict) and runtime_metadata.get("factor_by", None):
        side_condition_policy = replace(
            side_condition_policy,
            exempt_from_dependency_pruning=True,
            allow_first_numeric_by_unpruned=False,
        )

    predict_coefficient_map_arr = (
        None
        if predict_coefficient_map is None
        else np.asarray(predict_coefficient_map, dtype=np.float64)
    )
    raw_predict_n_coef = (
        int(predict_coefficient_map_arr.shape[0])
        if predict_coefficient_map_arr is not None
        else int(B.shape[1])
    )
    coefficient_maps: list[CoefficientMap] = []

    if absorb_cons and (not runtime_transform_applied) and fit_constraint is not None:
        B, penalty_defs, T_fit, n_cons = absorb_explicit_constraints(
            B,
            _copy_penalty_defs(penalty_defs),
            fit_constraint,
        )
        fit_coefficient_map = np.asarray(T_fit, dtype=np.float64)
        coefficient_maps.append(
            CoefficientMap(
                source_space="raw_term_space",
                target_space="local_fit_space",
                matrix=fit_coefficient_map,
                reason="local_constraint_absorption",
            )
        )
        if predict_coefficient_map_arr is None:
            predict_coefficient_map_arr = fit_coefficient_map.copy()
        expected_shape = (int(raw_predict_n_coef), int(B.shape[1]))
        if predict_coefficient_map_arr.shape != expected_shape:
            raise ValueError(
                f"Predict coefficient map for term {getattr(runtime, 'label', str(runtime))!r} "
                f"has shape {predict_coefficient_map_arr.shape}, expected {expected_shape}."
            )
        constraint_absorption = "wrapper"
        n_constraints_absorbed = int(n_cons)
        predict_map_source = (
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
        constraint_absorption = "runtime" if runtime_transform_applied else "none"
        n_constraints_absorbed = None
        predict_map_source = (
            "runtime" if predict_coefficient_map_arr is not None else "none"
        )

    if predict_coefficient_map_arr is not None and not any(
        cmap.reason == "local_constraint_absorption" for cmap in coefficient_maps
    ):
        coefficient_maps.append(
            CoefficientMap(
                source_space="raw_term_space",
                target_space="local_fit_space",
                matrix=np.asarray(predict_coefficient_map_arr, dtype=np.float64),
                reason="runtime_constraint_transform",
            )
        )

    constructor_metadata = compatibility_constructor_metadata(
        runtime_transform_applied=runtime_transform_applied,
        side_condition_policy=side_condition_policy,
        by_variable_info=by_variable_info,
        constraint_kind=getattr(runtime, "constraint_kind", None),
        constraint_absorption=constraint_absorption,
        n_constraints_absorbed=n_constraints_absorbed,
        predict_map_source=predict_map_source,
    )

    if null_space_penalty:
        raise NotImplementedError(
            "Generic smoothCon-level null-space penalty insertion is not enabled yet in this wrapper."
        )

    original_design_matrix = None
    if not bool(apply_by):
        original_design_matrix = np.asarray(B, dtype=np.float64)
        if by_variable_info.is_active:
            X_no_by = np.asarray(X, dtype=object).copy()
            by_name = str(by_variable_info.name)
            if by_name not in feature_names:
                raise KeyError(f"by variable {by_name!r} not found in feature_names.")
            X_no_by[:, feature_names.index(by_name)] = 1.0
            original_design_matrix = np.asarray(
                runtime.transform_new(X_no_by), dtype=np.float64
            )

    compiled_term = CompiledTerm(
        label=str(getattr(runtime, "label", str(runtime))),
        coef_slice=slice(0, int(B.shape[1])),
        basis_train=np.asarray(B, dtype=np.float64),
        predict_fn=getattr(runtime, "transform_new", None),
        predict_coefficient_map=(
            compose_coefficient_maps(tuple(coefficient_maps))
            if predict_coefficient_map_arr is None
            else np.asarray(predict_coefficient_map_arr, dtype=np.float64)
        ),
        basis_transform=np.eye(int(B.shape[1]), dtype=np.float64),
        coefficient_maps=tuple(coefficient_maps),
        feature_info=_feature_info_from_runtime(runtime),
        by_variable_info=by_variable_info,
        side_condition_policy=side_condition_policy,
        kept_columns=np.arange(int(B.shape[1]), dtype=int),
        deleted_columns=np.array([], dtype=int),
        smoothing_indices=[],
        smoothing_ids=[],
        n_penalties=0,
        term_type=str(getattr(runtime, "term_type", "smooth")),
        basis_name=str(getattr(runtime, "basis_name", "unknown")),
        term_id=str(getattr(runtime, "term_id", "")),
        smoothing_group_id=(
            None
            if getattr(runtime, "smoothing_id", None) is None
            else str(runtime.smoothing_id)
        ),
        penalty_specs=tuple(penalty_defs),
        constructor_metadata=constructor_metadata,
        metadata=dict(getattr(runtime, "metadata", {}) or {}),
    )

    return ConstructedSmooth(
        compiled_term=compiled_term,
        penalty_specs=tuple(penalty_defs),
        fit_constraint_operator=(
            None
            if fit_constraint is None
            else np.asarray(fit_constraint, dtype=np.float64)
        ),
        prediction_offset=(
            None
            if prediction_offset is None
            else np.asarray(prediction_offset, dtype=np.float64)
        ),
        original_design_matrix=original_design_matrix,
    )


__all__ = ["ConstructedSmooth", "build_term_matrix", "construct_smooth"]
