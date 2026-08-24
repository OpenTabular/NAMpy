"""Prediction-time reconstruction of formula preprocessing state."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .build import _evaluate_formula_numeric_expression, numeric_1d_values


def apply_formula_preprocess_to_new_data(
    data, preprocess_state, *, skip_output_columns=()
):
    if preprocess_state is None:
        return data

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Formula preprocessing for prediction requires a pandas DataFrame."
        )

    out = data.copy()
    skipped_outputs = {str(value) for value in skip_output_columns}

    def validate_factor_levels(source, allowed_levels):
        values = out[source]
        if values.isna().any():
            raise ValueError(
                f"Prediction factor column {source!r} contains missing values."
            )
        observed = list(pd.unique(values))
        unseen = [
            value
            for value in observed
            if not any(value == level for level in allowed_levels)
        ]
        if unseen:
            raise ValueError(
                f"Prediction factor column {source!r} contains unseen levels: "
                f"{unseen}. Training levels are {list(allowed_levels)}."
            )

    for item in preprocess_state.get("formula_expression_columns", []):
        if str(item["hidden_name"]) in skipped_outputs:
            continue
        values, _src_vars = _evaluate_formula_numeric_expression(item["expr"], out)
        out[item["hidden_name"]] = values

    for item in preprocess_state.get("parametric_expansions", []):
        if str(item["hidden_name"]) in skipped_outputs:
            continue
        vals = np.ones(len(out), dtype=np.float64)
        for comp in item["recipe"]:
            src = comp["var"]
            if src not in out.columns:
                raise KeyError(
                    f"Prediction data is missing parametric source column {src!r} "
                    f"needed to rebuild {item['hidden_name']!r}."
                )

            if comp["type"] == "numeric":
                vals = vals * numeric_1d_values(out[src], name=src)
            elif comp["type"] == "factor":
                validate_factor_levels(src, comp["levels"])
                vals = vals * np.asarray(
                    (out[src] == comp["level"]).astype(float), dtype=np.float64
                )
            else:
                raise ValueError(f"Unknown parametric recipe type {comp['type']!r}.")

        out[item["hidden_name"]] = vals

    for item in preprocess_state.get("factor_by_expansions", []):
        src = item["source_by"]
        if str(item["hidden_by"]) in skipped_outputs:
            if src in out.columns:
                validate_factor_levels(src, item["all_levels"])
            continue
        if src not in out.columns:
            raise KeyError(
                f"Prediction data is missing factor by-variable {src!r} "
                f"needed to rebuild formula columns."
            )

        validate_factor_levels(src, item["all_levels"])

        out[item["hidden_by"]] = np.asarray(
            (out[src] == item["level"]).astype(float), dtype=np.float64
        )

    return out


__all__ = ["apply_formula_preprocess_to_new_data"]
