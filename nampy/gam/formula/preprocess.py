"""Formula preprocessing (factor-by and parametric expansion)."""

import itertools
import re
from dataclasses import replace

import numpy as np
import pandas as pd

from ..specs import (
    build_smooth_spec,
    LinearPredictorSpec,
    TermSpec,
    replace_smooth_spec,
)
from ..specs.smooth import FactorSmoothInteractionSpec


def is_factor_like_series(s: pd.Series) -> bool:
    dtype = s.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


def factor_info(s: pd.Series):
    cat = pd.Categorical(s)
    levels = list(cat.categories)
    ordered = bool(getattr(cat.dtype, "ordered", False))
    return cat, levels, ordered


def safe_token(x) -> str:
    txt = str(x)
    txt = re.sub(r"\s+", "_", txt)
    txt = re.sub(r"[^0-9A-Za-z_]+", "", txt)
    return txt or "level"


def numeric_1d_values(s: pd.Series, *, name: str):
    vals = np.asarray(s, dtype=np.float64).ravel()
    if vals.ndim != 1:
        raise ValueError(f"Column {name!r} did not coerce to a 1D numeric array.")
    if not np.isfinite(vals).all():
        raise ValueError(f"Column {name!r} contains NaN or Inf.")
    return vals


def _fs_by_without_factor_feature_base_spec(smooth_spec: FactorSmoothInteractionSpec):
    """Mirror mgcv::smooth.construct.fs.smooth.spec when no factor term is present."""
    xt = smooth_spec.xt
    if xt is None:
        base_bs = "tp"
        xt_rest = None
    elif isinstance(xt, str):
        base_bs = str(xt).lower()
        xt_rest = None
    elif isinstance(xt, dict):
        base_bs = str(xt.get("bs", "tp")).lower()
        xt_rest = {k: v for k, v in xt.items() if k != "bs"} or None
    else:
        raise NotImplementedError(
            "For `bs=\"fs\"`, xt must be None, a basis string, or a dict "
            "containing optional key `bs`."
        )

    kwargs = {
        "special": str(smooth_spec.special),
        "bs": base_bs,
        "k": smooth_spec.k,
        "fx": smooth_spec.fx,
        "select": smooth_spec.select,
        "sp": smooth_spec.sp,
        "knots": smooth_spec.knots,
        "constraint_mode": str(getattr(smooth_spec, "constraint_mode", "auto")),
    }

    if base_bs == "ps":
        kwargs["m"] = None if xt_rest is None else xt_rest.get("m", None)
    elif base_bs in {"tp", "ts", "gp"}:
        kwargs["xt"] = xt_rest
    elif xt_rest:
        raise NotImplementedError(
            f"`bs=\"fs\"` factor-by fallback does not support extra xt options for "
            f"base bs={base_bs!r}."
        )

    return build_smooth_spec(**kwargs)


def expand_parametric_term(
    term,
    data_work,
    *,
    pred_name,
    include_intercept,
    hidden_counter,
    state,
):
    src_vars = list(
        term.metadata.get(
            "source_variables",
            list(term.features) if getattr(term, "features", None) else [term.label],
        )
    )
    if len(src_vars) == 0:
        raise ValueError(
            f"Parametric term {term.label!r} does not define any source variables."
        )

    n = len(data_work)
    comp_lists = []
    needs_expansion = len(src_vars) > 1

    for var in src_vars:
        if var not in data_work.columns:
            raise KeyError(
                f"Formula references parametric variable {var!r}, but it is not in `data`."
            )

        s = data_work[var]

        if is_factor_like_series(s):
            needs_expansion = True
            cat, levels, ordered = factor_info(s)
            active_levels = levels if not include_intercept else levels[1:]

            comps = []
            for lev in active_levels:
                comps.append(
                    {
                        "label": f"{var}[{lev}]",
                        "values": np.asarray(
                            (cat == lev).astype(float), dtype=np.float64
                        ),
                        "recipe": {
                            "var": var,
                            "type": "factor",
                            "level": lev,
                            "levels": list(levels),
                            "ordered": ordered,
                            "include_intercept": bool(include_intercept),
                        },
                    }
                )
            comp_lists.append(comps)
        else:
            vals = numeric_1d_values(s, name=var)
            comp_lists.append(
                [
                    {
                        "label": var,
                        "values": vals,
                        "recipe": {
                            "var": var,
                            "type": "numeric",
                        },
                    }
                ]
            )

    if not needs_expansion:
        return [term], hidden_counter

    if any(len(comps) == 0 for comps in comp_lists):
        return [], hidden_counter

    out_terms = []
    for combo in itertools.product(*comp_lists):
        values = np.ones(n, dtype=np.float64)
        labels = []
        recipe = []

        for c in combo:
            values = values * np.asarray(c["values"], dtype=np.float64)
            labels.append(c["label"])
            recipe.append(dict(c["recipe"]))

        hidden_col = f"__gam_param__{pred_name}__{hidden_counter}"
        hidden_counter += 1
        data_work[hidden_col] = values

        label = ":".join(labels)
        meta = dict(term.metadata)
        meta["parametric_expansion"] = {
            "source_variables": list(src_vars),
            "label": label,
            "hidden_name": hidden_col,
            "recipe": recipe,
        }

        out_terms.append(
            TermSpec(
                kind="parametric",
                features=(hidden_col,),
                by_variable=None,
                basis_options={},
                smoothing_id=None,
                label=label,
                metadata=meta,
            )
        )

        state["parametric_expansions"].append(
            {
                "hidden_name": hidden_col,
                "label": label,
                "source_variables": list(src_vars),
                "recipe": recipe,
            }
        )

    return out_terms, hidden_counter


def expand_factor_by_term(
    term,
    data_work,
    *,
    pred_name,
    hidden_counter,
    state,
):
    if not isinstance(term, TermSpec) or term.kind != "smooth":
        return [term], hidden_counter

    by_name = term.by_variable
    if by_name is None:
        return [term], hidden_counter

    if by_name not in data_work.columns:
        raise KeyError(
            f"Formula references by-variable {by_name!r}, but it is not in `data`."
        )

    by_series = data_work[by_name]

    if not is_factor_like_series(by_series):
        return [term], hidden_counter

    smooth_spec = term.smooth_spec
    if smooth_spec is None:
        raise ValueError(f"Smooth term {term.label!r} is missing smooth_spec.")

    if str(smooth_spec.special) != "s":
        raise NotImplementedError(
            f"Factor `by` expansion is implemented for s(...) only in this step, "
            f"not for {smooth_spec.special}(...)."
        )

    if isinstance(smooth_spec, FactorSmoothInteractionSpec):
        has_factor_feature = any(
            feature in data_work.columns and is_factor_like_series(data_work[feature])
            for feature in term.features
        )
        if not has_factor_feature:
            # Upstream mgcv falls back to the underlying base smoother here:
            # smooth.construct.fs.smooth.spec() returns smooth.construct(base.bs)
            # when the fs term itself has no factor argument.
            smooth_spec = _fs_by_without_factor_feature_base_spec(smooth_spec)

    cat, levels, ordered = factor_info(by_series)
    if len(levels) == 0:
        return [], hidden_counter

    active_levels = levels[1:] if ordered else levels
    original_label = term.label
    out_terms = []

    for lev in active_levels:
        token = safe_token(lev)
        hidden_col = f"__gam_by__{pred_name}__{by_name}__{token}__{hidden_counter}"
        hidden_counter += 1

        data_work[hidden_col] = np.asarray((cat == lev).astype(float), dtype=np.float64)

        new_meta = dict(term.metadata)
        new_meta["factor_by"] = {
            "source_by": by_name,
            "level": lev,
            "ordered": ordered,
            "hidden_by": hidden_col,
            "all_levels": list(levels),
        }

        new_label = f"{original_label}:{by_name}={lev}"
        out_terms.append(
            replace(
                term,
                by_variable=hidden_col,
                smooth_spec=replace_smooth_spec(
                    smooth_spec, constraint_mode="factor_by"
                ),
                label=new_label,
                metadata=new_meta,
            )
        )

        state["factor_by_expansions"].append(
            {
                "predictor_name": pred_name,
                "source_by": by_name,
                "level": lev,
                "ordered": ordered,
                "all_levels": list(levels),
                "hidden_by": hidden_col,
                "original_label": original_label,
                "expanded_label": new_label,
            }
        )

    return out_terms, hidden_counter


def preprocess_formula_predictor_specs(parsed, predictor_specs, data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Formula preprocessing requires `data` to be a pandas DataFrame."
        )

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
            if isinstance(term, TermSpec) and term.kind == "parametric":
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

            if not (isinstance(term, TermSpec) and term.kind == "smooth"):
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
                has_intercept=bool(getattr(pred, "has_intercept", False)),
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
                vals = vals * np.asarray(
                    (out[src] == comp["level"]).astype(float), dtype=np.float64
                )
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

        out[item["hidden_by"]] = np.asarray(
            (out[src] == item["level"]).astype(float), dtype=np.float64
        )

    return out


__all__ = [
    "preprocess_formula_predictor_specs",
    "apply_formula_preprocess_to_new_data",
    "expand_factor_by_term",
    "expand_parametric_term",
    "is_factor_like_series",
    "factor_info",
    "safe_token",
    "numeric_1d_values",
]
