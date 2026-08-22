"""Predictor-aware term grouping and contribution primitives.

This module is deliberately independent of the prediction entry points so
ordinary and general-family prediction can share term semantics without
importing each other.
"""

from __future__ import annotations

import numpy as np

from ..model_state import (
    _coef,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from ..term_labels import normalize_mgcv_term_label


def _parametric_formula_term(term) -> str | None:
    metadata = dict(getattr(term, "metadata", {}) or {})
    formula_term = metadata.get("formula_term", None)
    return None if formula_term is None else str(formula_term)


def _prediction_term_groups(model):
    """Return ordered mgcv term groups without treating labels as identity."""
    groups = []
    for term in _term_blocks_seq(model):
        term_type = str(getattr(term, "term_type", ""))
        predictor_index = int(getattr(term, "predictor_index", 0))
        predictor_name = str(getattr(term, "predictor_name", "predictor_0"))
        if term_type == "parametric":
            formula_term = _parametric_formula_term(term)
            group_key = (
                "parametric",
                predictor_index,
                formula_term or str(getattr(term, "label", "")),
            )
            if groups and groups[-1]["key"] == group_key:
                groups[-1]["blocks"].append(term)
                continue
            groups.append(
                {
                    "key": group_key,
                    "label": formula_term or str(getattr(term, "label", "")),
                    "blocks": [term],
                    "term_type": term_type,
                    "predictor_index": predictor_index,
                    "predictor_name": predictor_name,
                }
            )
            continue

        groups.append(
            {
                "key": (
                    "term",
                    predictor_index,
                    str(normalize_mgcv_term_label(getattr(term, "label", ""))),
                ),
                "label": str(
                    normalize_mgcv_term_label(getattr(term, "label", ""))
                ),
                "blocks": [term],
                "term_type": term_type,
                "predictor_index": predictor_index,
                "predictor_name": predictor_name,
            }
        )
    return groups


def _term_contribution(model, design_matrix: np.ndarray, term):
    beta = np.asarray(_coef(model), dtype=np.float64)
    # mgcv/R/mgcv.r::predict.gam forms every smooth contribution directly
    # from its PredictMat block and the corresponding coefficient block.
    return design_matrix[:, term.coef_slice] @ beta[term.coef_slice]


def _group_term_contribution(model, design_matrix: np.ndarray, group):
    contribution = np.zeros(design_matrix.shape[0], dtype=np.float64)
    for term in group["blocks"]:
        contribution += np.asarray(
            _term_contribution(model, design_matrix, term), dtype=np.float64
        )
    return np.asarray(contribution, dtype=np.float64)


def _term_standard_error_rows(model, lpmatrix: np.ndarray, term):
    full_indices = _term_full_coefficient_indices(model, term)
    return np.asarray(lpmatrix[:, full_indices], dtype=np.float64), full_indices


def _group_standard_error_rows(model, lpmatrix: np.ndarray, group, *, type="terms"):
    del type
    if len(group["blocks"]) == 1:
        return _term_standard_error_rows(model, lpmatrix, group["blocks"][0])

    if group["term_type"] != "parametric":
        raise NotImplementedError("Only parametric prediction groups may span blocks.")

    columns = []
    for term in group["blocks"]:
        columns.extend(_term_full_coefficient_indices(model, term).tolist())
    if not columns:
        return np.empty((lpmatrix.shape[0], 0), dtype=np.float64), slice(0, 0)
    indices = np.asarray(columns, dtype=int)
    return np.asarray(lpmatrix[:, indices], dtype=np.float64), indices


__all__ = [
    "_group_standard_error_rows",
    "_group_term_contribution",
    "_prediction_term_groups",
]
