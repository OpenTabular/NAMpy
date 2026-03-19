from dataclasses import replace

import numpy as np

from ...specs import SmoothTermSpec
from .parametric import factor_info, is_factor_like_series, safe_token


def expand_factor_by_term(
    term,
    data_work,
    *,
    pred_name,
    hidden_counter,
    state,
):
    if not isinstance(term, SmoothTermSpec):
        return [term], hidden_counter

    by_name = term.by
    if by_name is None:
        return [term], hidden_counter

    if by_name not in data_work.columns:
        raise KeyError(
            f"Formula references by-variable {by_name!r}, but it is not in `data`."
        )

    by_series = data_work[by_name]

    if not is_factor_like_series(by_series):
        return [term], hidden_counter

    if term.special != "s":
        raise NotImplementedError(
            f"Factor `by` expansion is implemented for s(...) only in this step, "
            f"not for {term.special}(...)."
        )

    cat, levels, ordered = factor_info(by_series)
    if len(levels) == 0:
        return [], hidden_counter

    active_levels = levels[1:] if ordered else levels
    original_label = term.label
    out_terms = []

    for lev in active_levels:
        token = safe_token(lev)
        hidden_col = (
            f"__gam_by__{pred_name}__{by_name}__{token}__{hidden_counter}"
        )
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
                by=hidden_col,
                label=new_label,
                constraint_mode="factor_by",
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
