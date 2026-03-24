"""
Stage 3 of the GAM fit pipeline: term construction wrapper.

Bridges a fitted runtime term (stage 2 output) into a ConstructedTerm that the
stage-4 compiler can assemble without knowing anything about basis-specific
mathematics.

Responsibilities of this layer
-------------------------------
- Extract the fitted basis and penalty definitions from the runtime term.
- If the runtime delegated by-variable handling: apply the numeric by-vector
  and build a by-aware predict function.
- If the runtime delegated explicit constraint absorption: call
  ``absorb_explicit_constraints`` and wrap the predict function to apply the
  same transform at prediction time.
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
from ..specs import TermSpec
from ..runtime.factory import instantiate_term
from ..smooths.base import _resolve_feature
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
    if isinstance(term_like, TermSpec) and hasattr(term_like, "term_id"):
        runtime.term_id = str(term_like.term_id)
    runtime.fit(X, feature_names)
    B, penalty_defs = _extract_runtime_state(runtime)
    by_done = bool(getattr(runtime, "by_done", True))
    constraints_absorbed = bool(getattr(runtime, "constraints_absorbed", True))
    fit_constraint = getattr(runtime, "fit_constraint_matrix", None)
    predict_constraint = getattr(runtime, "predict_constraint_matrix", None)
    prediction_offset = getattr(runtime, "prediction_offset", None)

    constructor_metadata = {
        "by_done": by_done,
        "constraints_absorbed_by_runtime": constraints_absorbed,
        "runtime_constraint_kind": getattr(runtime, "_constraint_kind", None),
        "runtime_by_name": getattr(runtime, "_by_name", None),
        "runtime_by_is_constant": getattr(runtime, "_by_is_constant", None),
    }

    X0 = None
    predict_fn = None
    by_variable = getattr(runtime, "by", None)
    if by_variable is not None and apply_by and not by_done:
        idx, _by_name = _resolve_feature(by_variable, feature_names)
        z_train = np.asarray(X[:, idx], dtype=np.float64).ravel()
        if hasattr(runtime, "basis_train_base") and runtime.basis_train_base is not None:
            X0 = np.asarray(runtime.basis_train_base, dtype=np.float64)
        else:
            raise NotImplementedError("Runtime term requested wrapper-level by handling without basis_train_base.")
        B = X0 * z_train[:, None]
        if hasattr(runtime, "transform_new_base") and callable(runtime.transform_new_base):
            base_predict_fn = runtime.transform_new_base
        else:
            raise NotImplementedError("Runtime term requested wrapper-level by handling without transform_new_base().")

        def predict_fn(X_new, *, _base=base_predict_fn, _idx=idx):
            X_new = np.asarray(X_new, dtype=np.float64)
            z_new = np.asarray(X_new[:, _idx], dtype=np.float64).ravel()
            B0_new = np.asarray(_base(X_new), dtype=np.float64)
            return B0_new * z_new[:, None]

        constructor_metadata["by_handling"] = "wrapper"
    else:
        constructor_metadata["by_handling"] = "runtime" if by_variable is not None else "none"

    if absorb_cons and (not constraints_absorbed) and fit_constraint is not None:
        B, penalty_defs, T_fit, n_cons = absorb_explicit_constraints(
            B,
            _copy_penalty_defs(penalty_defs),
            fit_constraint,
        )
        base_predict_fn = runtime.transform_new if predict_fn is None else predict_fn

        def predict_fn(X_new, *, _base=base_predict_fn, _T=T_fit):
            return np.asarray(_base(X_new), dtype=np.float64) @ _T

        constraints_absorbed = True
        constructor_metadata["constraint_absorption"] = "wrapper"
        constructor_metadata["n_constraints_absorbed"] = int(n_cons)
    else:
        constructor_metadata["constraint_absorption"] = "runtime" if constraints_absorbed else "none"
        constructor_metadata["n_constraints_absorbed"] = None

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
        by_variable=by_variable,
        smoothing_id=(None if getattr(runtime, "smoothing_id", None) is None else str(getattr(runtime, "smoothing_id"))),
        metadata=dict(getattr(runtime, "metadata", {}) or {}),
        fit_constraint=(None if fit_constraint is None else np.asarray(fit_constraint, dtype=np.float64)),
        predict_constraint=(None if predict_constraint is None else np.asarray(predict_constraint, dtype=np.float64)),
        constraints_absorbed=bool(constraints_absorbed),
        prediction_offset=(None if prediction_offset is None else np.asarray(prediction_offset, dtype=np.float64)),
        original_design_matrix=None if X0 is None else np.asarray(X0, dtype=np.float64),
        constructor_metadata=constructor_metadata,
        _predict_fn=predict_fn,
    )
    return [smooth]


__all__ = ["construct_terms"]
