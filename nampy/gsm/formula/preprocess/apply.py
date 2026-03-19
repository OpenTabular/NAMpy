import numpy as np
import pandas as pd

from ...specs import LinearPredictorSpec, ParametricTermSpec, SmoothTermSpec
from .by_factor import expand_factor_by_term
from .parametric import expand_parametric_term, numeric_1d_values


def preprocess_formula_predictor_specs(parsed, predictor_specs, data):
    """
    Expand formula preprocessing needs for formula mode.

    Current scope
    -------------
    - factor / ordered-factor `by` variables for s(...)
    - one smooth replica per active factor level
    - ordered-factor smooths omit the first level
    - parametric factor terms and parametric interactions expanded into hidden numeric columns

    Important limitation
    --------------------
    Parametric ordered factors are currently treatment-coded here, not given
    full R-style ordered contrasts. That is a later parity step.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Formula preprocessing requires `data` to be a pandas DataFrame.")

    if len(parsed.predictors) != len(predictor_specs):
        raise ValueError(
            f"Parsed predictor count ({len(parsed.predictors)}) does not match "
            f"predictor spec count ({len(predictor_specs)})."
        )

    data_work = data.copy()
    state = {
        "factor_by_expansions": [],
        "parametric_expansions": [],
    }

    out_specs = []
    hidden_counter = 0

    for parsed_pred, pred in zip(parsed.predictors, predictor_specs):
        out_terms = []
        include_intercept = bool(parsed_pred.intercept)

        for term in pred.terms:
            if isinstance(term, ParametricTermSpec):
                expanded, hidden_counter = expand_parametric_term(
                    term,
                    data_work,
                    pred_name=pred.name,
                    include_intercept=include_intercept,
                    hidden_counter=hidden_counter,
                    state=state,
                )
                out_terms.extend(expanded)
                continue

            if not isinstance(term, SmoothTermSpec):
                out_terms.append(term)
                continue

            expanded, hidden_counter = expand_factor_by_term(
                term,
                data_work,
                pred_name=pred.name,
                hidden_counter=hidden_counter,
                state=state,
            )
            out_terms.extend(expanded)

        out_specs.append(
            LinearPredictorSpec(
                name=pred.name,
                terms=out_terms,
                parameter_name=pred.parameter_name,
                offset_name=pred.offset_name,
                metadata=dict(pred.metadata),
            )
        )

    return out_specs, data_work, state


def apply_formula_preprocess_to_new_data(data, preprocess_state):
    if preprocess_state is None:
        return data

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Formula preprocessing for prediction requires a pandas DataFrame."
        )

    out = data.copy()

    for item in preprocess_state.get("parametric_expansions", []):
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
                vals = vals * np.asarray((out[src] == comp["level"]).astype(float), dtype=np.float64)
            else:
                raise ValueError(f"Unknown parametric recipe type {comp['type']!r}.")

        out[item["hidden_name"]] = vals

    for item in preprocess_state.get("factor_by_expansions", []):
        src = item["source_by"]
        if src not in out.columns:
            raise KeyError(
                f"Prediction data is missing factor by-variable {src!r} "
                f"needed to rebuild formula columns."
            )

        out[item["hidden_by"]] = np.asarray((out[src] == item["level"]).astype(float), dtype=np.float64)

    return out
