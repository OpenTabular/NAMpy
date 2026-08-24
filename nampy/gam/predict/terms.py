"""Predictor-aware term grouping and contribution primitives.

This module is deliberately independent of the prediction entry points so
ordinary and general-family prediction can share term semantics without
importing each other.
"""

from __future__ import annotations

import re
import warnings

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


def _multi_predictor_term_label(label: str, *, predictor_index: int, term_type: str):
    """Apply mgcv's formula-list suffix to later-predictor term labels."""
    if predictor_index <= 0:
        return label
    if term_type == "parametric":
        return f"{label}.{predictor_index}"
    open_index = label.find("(")
    if open_index < 0:
        return f"{label}.{predictor_index}"
    return f"{label[:open_index]}.{predictor_index}{label[open_index:]}"


def _prediction_term_groups(model):
    """Return ordered mgcv term groups without treating labels as identity."""
    groups = []
    for term in _term_blocks_seq(model):
        term_type = str(getattr(term, "term_type", ""))
        predictor_index = int(getattr(term, "predictor_index", 0))
        predictor_name = str(getattr(term, "predictor_name", "predictor_0"))
        if term_type == "parametric":
            formula_term = _parametric_formula_term(term)
            group_label = _multi_predictor_term_label(
                formula_term or str(getattr(term, "label", "")),
                predictor_index=predictor_index,
                term_type=term_type,
            )
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
                    "label": group_label,
                    # predict.gam zeros each formula's parametric model matrix
                    # against that formula's unsuffixed term.labels, even
                    # though the returned multi-predictor columns are suffixed.
                    "filter_label": formula_term or str(getattr(term, "label", "")),
                    "blocks": [term],
                    "term_type": term_type,
                    "predictor_index": predictor_index,
                    "predictor_name": predictor_name,
                }
            )
            continue

        group_label = _multi_predictor_term_label(
            str(normalize_mgcv_term_label(getattr(term, "label", ""))),
            predictor_index=predictor_index,
            term_type=term_type,
        )
        groups.append(
            {
                "key": (
                    "term",
                    predictor_index,
                    str(normalize_mgcv_term_label(getattr(term, "label", ""))),
                ),
                "label": group_label,
                "filter_label": group_label,
                "blocks": [term],
                "term_type": term_type,
                "predictor_index": predictor_index,
                "predictor_name": predictor_name,
            }
        )
    # predict.gam allocates all parametric term columns before the smooth
    # columns, including when object$pterms is a list for multiple linear
    # predictors (mgcv/R/mgcv.r::predict.gam, lines 2909-2918 and 3041-3094).
    # Keep the grouping owner in that same output order so every consumer maps
    # the returned term matrix consistently.
    return [group for group in groups if group["term_type"] == "parametric"] + [
        group for group in groups if group["term_type"] != "parametric"
    ]


def _term_filter_key(value):
    """Canonicalize inconsequential deparse spacing in mgcv term filters."""
    normalized = str(normalize_mgcv_term_label(value))
    return re.sub(r",\s*", ",", normalized)


def _prediction_group_selection(groups, *, terms, exclude):
    labels = tuple(str(group["label"]) for group in groups)
    filter_keys = tuple(
        _term_filter_key(group.get("filter_label", group["label"])) for group in groups
    )
    term_keys = None if terms is None else {_term_filter_key(term) for term in terms}
    exclude_keys = (
        None if exclude is None else {_term_filter_key(term) for term in exclude}
    )
    selected = np.ones(len(groups), dtype=bool)
    if term_keys is not None:
        selected &= np.asarray([key in term_keys for key in filter_keys], dtype=bool)
    if exclude_keys is not None:
        selected &= np.asarray(
            [key not in exclude_keys for key in filter_keys], dtype=bool
        )
    return labels, selected


def _filtered_term_output_indices(labels, *, terms, exclude):
    """Mirror predict.gam's post-computation terms/exclude column filtering."""
    indices = list(range(len(labels)))
    label_keys = [_term_filter_key(label) for label in labels]
    if terms is not None:
        term_keys = [_term_filter_key(label) for label in terms]
        missing = [
            label
            for label, key in zip(terms, term_keys, strict=True)
            if key not in label_keys
        ]
        if missing:
            warnings.warn(
                "non-existent terms requested - ignoring",
                stacklevel=3,
            )
        else:
            # Character indexing in R selects the first column when output
            # names are duplicated across linear predictors.
            indices = [label_keys.index(key) for key in term_keys]
    if exclude is not None:
        exclude_keys = [_term_filter_key(label) for label in exclude]
        missing = [
            label
            for label, key in zip(exclude, exclude_keys, strict=True)
            if key not in label_keys
        ]
        if missing:
            warnings.warn(
                "non-existent exclude terms requested - ignoring",
                stacklevel=3,
            )
        else:
            indices = [
                index for index in indices if label_keys[index] not in exclude_keys
            ]
    return np.asarray(indices, dtype=int)


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
    "_filtered_term_output_indices",
    "_group_standard_error_rows",
    "_group_term_contribution",
    "_prediction_group_selection",
    "_prediction_term_groups",
]
