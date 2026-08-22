"""Versioned pickle-state handling for fitted GAM models."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..families import clone_gam_family

GAM_STATE_SCHEMA_VERSION = 1
_SCHEMA_KEY = "_gam_state_schema_version"


def gam_pickle_state(model: Any) -> dict[str, Any]:
    state = dict(model.__dict__)
    # Warm starts and evaluation caches are private to one fit invocation.
    state.pop("_ws", None)
    state[_SCHEMA_KEY] = GAM_STATE_SCHEMA_VERSION
    return state


def _restore_compiled_term_coordinates(model: Any) -> None:
    result = getattr(model, "gam_result_", None)
    compiled = None if result is None else getattr(result, "compiled_model", None)
    if compiled is None:
        return
    predictors = tuple(getattr(compiled, "predictors", ()) or ())
    reduced_to_full = np.asarray(
        getattr(compiled, "coef_reduced_to_full_idx", ()), dtype=int
    ).reshape(-1)
    predictor_ranges = []
    start = 0
    for predictor in predictors:
        stop = start + int(predictor.n_coef)
        predictor_ranges.append((start, stop, predictor))
        start = stop

    for term in tuple(getattr(compiled, "compiled_terms", ()) or ()):
        term_start = int(term.coef_slice.start)
        term_stop = int(term.coef_slice.stop)
        predictor_index = next(
            (
                index
                for index, (start, stop, _predictor) in enumerate(predictor_ranges)
                if start <= term_start and term_stop <= stop
            ),
            0,
        )
        predictor = predictors[predictor_index] if predictors else None
        term.predictor_index = int(predictor_index)
        term.predictor_name = (
            "predictor_0" if predictor is None else str(predictor.name)
        )
        if reduced_to_full.size >= term_stop:
            term.full_coef_indices = reduced_to_full[term_start:term_stop].copy()


def restore_gam_pickle_state(model: Any, state: dict[str, Any]) -> None:
    state = dict(state)
    version = int(state.pop(_SCHEMA_KEY, 0))
    if version > GAM_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Cannot load GAM pickle schema "
            f"{version}; this NAMpy version supports up to "
            f"{GAM_STATE_SCHEMA_VERSION}."
        )
    model.__dict__.update(state)
    if "_family_template" not in model.__dict__:
        model._family_template = clone_gam_family(model.family)
    _restore_compiled_term_coordinates(model)


__all__ = [
    "GAM_STATE_SCHEMA_VERSION",
    "gam_pickle_state",
    "restore_gam_pickle_state",
]
