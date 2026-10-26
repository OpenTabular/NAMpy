"""Explicit homogeneous smooth constraints at the specification boundary."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ..term_labels import normalize_mgcv_term_label


def with_smooth_constraints(predictors, constraints):
    """Attach constraints selected by unique label or one-based smooth index."""
    if constraints is None:
        return predictors
    if not isinstance(constraints, Mapping):
        raise TypeError(
            "constraints must map smooth labels or one-based indices to matrices."
        )
    smooths = [t for p in predictors for t in p.terms if t.kind == "smooth"]
    selected = {}
    for key, value in constraints.items():
        if isinstance(key, int) and not isinstance(key, bool):
            matches = [smooths[key - 1]] if 1 <= key <= len(smooths) else []
        else:
            matches = [
                t
                for t in smooths
                if t.label == key or normalize_mgcv_term_label(t.label) == key
            ]
        if len(matches) != 1:
            raise ValueError(
                f"Constraint selector {key!r} must identify exactly one smooth."
            )
        term = matches[0]
        if term.term_id in selected:
            raise ValueError(
                f"Multiple constraints supplied for smooth {term.label!r}."
            )
        selected[term.term_id] = np.array(value, dtype=float, copy=True)
    return [
        replace(
            p,
            terms=[
                replace(
                    t, metadata={**t.metadata, "user_constraint": selected[t.term_id]}
                )
                if t.term_id in selected
                else t
                for t in p.terms
            ],
        )
        for p in predictors
    ]


def validate_constraint_matrix(value, width):
    C = np.asarray(value, dtype=float)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2 or C.shape[1] != width or not np.all(np.isfinite(C)):
        raise ValueError(f"Constraint matrix must be finite with {width} columns.")
    if C.shape[0] >= width:
        raise ValueError("Constraints must leave at least one smooth coefficient.")
    if C.shape[0] and np.linalg.matrix_rank(C) != C.shape[0]:
        raise ValueError(
            "Constraint rows must be linearly independent; use zero rows for no constraints."
        )
    if np.any((np.sum(C, axis=0) == 0) & np.any(C != 0, axis=0)):
        raise ValueError(
            "Constraint columns with cancelling row sums are unsupported by upstream localized absorption."
        )
    return C.copy()
