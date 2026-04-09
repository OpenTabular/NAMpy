"""
Stage 3 of the GAM fit pipeline: term construction wrapper.

Bridges a fitted runtime term (stage 2 output) into a ConstructedTerm that the
stage-4 compiler can assemble without knowing anything about basis-specific
mathematics.

Responsibilities of this layer
-------------------------------
- Extract the fitted basis and penalty definitions from the runtime term.
- If the runtime delegated explicit constraint absorption: call
  ``absorb_explicit_constraints`` for the fit basis and store the resulting
  fit/predict coefficient maps explicitly on the constructed term.
- Record in ``constructor_metadata`` which layer handled each concern (runtime
  vs. wrapper) so that stage 5 can detect what was already done.

What this layer must NOT do
----------------------------
- Implement basis-specific mathematics.
- Apply predictor-wide side conditions (that is stage 5).
- Duplicate transforms that the runtime already applied.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..constraints.absorption import absorb_explicit_constraints
from ..penalties import normalize_penalty_spec
from ..runtime.factory import instantiate_term
from ..specs import TermSpec
from .constructed import ConstructedTerm


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


def construct_terms(
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
    constraints_absorbed = bool(getattr(runtime, "constraints_absorbed", True))
    fit_constraint = getattr(runtime, "fit_constraint_matrix", None)
    predict_coefficient_map = getattr(runtime, "predict_coefficient_map", None)
    prediction_offset = getattr(runtime, "prediction_offset", None)
    _by_state = getattr(runtime, "_by_state", None)

    constructor_metadata = {
        "constraints_absorbed_by_runtime": constraints_absorbed,
        "runtime_constraint_kind": getattr(runtime, "constraint_kind", None),
        "runtime_by_name": _by_state.feature_name if _by_state is not None else None,
        "runtime_by_is_constant": (
            _by_state.is_constant if _by_state is not None else None
        ),
    }

    # Prediction basis already lives in runtime-owned by/constraint semantics.
    # Any remaining linear map into fitted coefficient coordinates is stored
    # explicitly on ConstructedTerm.
    raw_predict_n_coef = int(B.shape[1])
    constructor_metadata["by_handling"] = (
        "runtime" if getattr(runtime, "by", None) is not None else "none"
    )

    fit_coefficient_map = None
    predict_coefficient_map_arr = (
        None
        if predict_coefficient_map is None
        else np.asarray(predict_coefficient_map, dtype=np.float64)
    )

    if absorb_cons and (not constraints_absorbed) and fit_constraint is not None:
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
        constraints_absorbed = True
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
            "runtime" if constraints_absorbed else "none"
        )
        constructor_metadata["n_constraints_absorbed"] = None
        constructor_metadata["predict_map_source"] = (
            "runtime" if predict_coefficient_map_arr is not None else "none"
        )

    predict_fn = None

    if null_space_penalty:
        raise NotImplementedError(
            "Generic smoothCon-level null-space penalty insertion is not enabled yet in this wrapper."
        )

    smooth = ConstructedTerm(
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
        constraints_absorbed=bool(constraints_absorbed),
        prediction_offset=(
            None
            if prediction_offset is None
            else np.asarray(prediction_offset, dtype=np.float64)
        ),
        original_design_matrix=None,
        constructor_metadata=constructor_metadata,
        _predict_fn=predict_fn,
    )
    return [smooth]


__all__ = ["construct_terms"]
