"""Typed architecture contracts for GAM construction and compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoefficientMap:
    source_space: str
    target_space: str
    matrix: np.ndarray
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ByVariableInfo:
    name: str | None = None
    is_constant: bool | None = None
    handling: str = "none"

    @property
    def is_active(self) -> bool:
        return self.name is not None


@dataclass(frozen=True)
class TermFeatureInfo:
    feature_names: tuple[str, ...] = ()
    feature_indices: tuple[int, ...] = ()
    metric_feature_indices: tuple[int, ...] = ()
    factor_feature_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class SideConditionPolicy:
    skip_centering: bool = False
    exempt_from_predictor_side_conditions: bool = False
    exempt_from_predictor_centering: bool = False
    exempt_from_dependency_pruning: bool = False
    requires_penalty_aware_pruning: bool = True
    allow_first_numeric_by_unpruned: bool = False


def compose_coefficient_maps(
    maps: tuple[CoefficientMap, ...] | list[CoefficientMap],
) -> np.ndarray | None:
    ordered = list(maps)
    if len(ordered) == 0:
        return None
    result = np.asarray(ordered[0].matrix, dtype=np.float64)
    for cmap in ordered[1:]:
        result = result @ np.asarray(cmap.matrix, dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def compatibility_constructor_metadata(
    *,
    runtime_transform_applied: bool,
    side_condition_policy: SideConditionPolicy,
    by_variable_info: ByVariableInfo,
    constraint_kind: str | None,
    constraint_absorption: str,
    n_constraints_absorbed: int | None,
    predict_map_source: str,
) -> dict[str, Any]:
    return {
        "runtime_transform_applied": bool(runtime_transform_applied),
        "runtime_skip_centering": bool(side_condition_policy.skip_centering),
        "runtime_constraint_kind": constraint_kind,
        "runtime_by_name": by_variable_info.name,
        "runtime_by_is_constant": by_variable_info.is_constant,
        "by_handling": by_variable_info.handling,
        "constraint_absorption": constraint_absorption,
        "n_constraints_absorbed": n_constraints_absorbed,
        "predict_map_source": predict_map_source,
    }


def default_side_condition_policy(
    *,
    term_type: str,
    runtime_skip_centering: bool,
    by_variable_info: ByVariableInfo,
) -> SideConditionPolicy:
    exempt = term_type in {"random_effect", "factor_smooth_fs", "factor_smooth_sz"}
    return SideConditionPolicy(
        skip_centering=bool(runtime_skip_centering),
        exempt_from_predictor_side_conditions=exempt,
        exempt_from_predictor_centering=exempt,
        exempt_from_dependency_pruning=exempt,
        requires_penalty_aware_pruning=True,
        allow_first_numeric_by_unpruned=(
            by_variable_info.is_active and not bool(by_variable_info.is_constant)
        ),
    )


__all__ = [
    "ByVariableInfo",
    "CoefficientMap",
    "SideConditionPolicy",
    "TermFeatureInfo",
    "compatibility_constructor_metadata",
    "compose_coefficient_maps",
    "default_side_condition_policy",
]
