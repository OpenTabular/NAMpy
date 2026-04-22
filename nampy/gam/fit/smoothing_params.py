"""Smoothing-parameter coercion and optimization wrappers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .._model_state import (
    _compiled_metadata,
    _compiled_model,
    _n_smoothing_params,
    _penalty_blocks_seq,
)


def resolve_min_sp(model, min_sp):
    n_smoothing_params = _n_smoothing_params(model)
    penalty_blocks = tuple(_penalty_blocks_seq(model))
    if n_smoothing_params == 0 and _compiled_model(model) is None:
        raise RuntimeError("Design has not been compiled yet.")

    if min_sp is None:
        return np.zeros(n_smoothing_params, dtype=np.float64)

    arr = np.asarray(min_sp, dtype=np.float64).ravel()
    if arr.size == 0 and n_smoothing_params == 0:
        return np.empty((0,), dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("min_sp values must be finite and >= 0.")

    if arr.shape == (n_smoothing_params,):
        return arr.copy()
    if arr.shape == (len(penalty_blocks),):
        out = np.zeros(n_smoothing_params, dtype=np.float64)
        for val, pb in zip(arr, penalty_blocks):
            out[pb.smoothing_index] = max(out[pb.smoothing_index], float(val))
        return out
    raise ValueError(
        f"min_sp must have shape ({n_smoothing_params},) for underlying smoothing "
        f"parameters or ({len(penalty_blocks)},) for total penalties, got {arr.shape}."
    )


def resolve_smoothing_params(model, n_smoothing_params):
    sp = model.smoothing_params
    if sp is None:
        sp = np.ones(n_smoothing_params, dtype=np.float64)
    elif isinstance(sp, Mapping):
        out = np.ones(n_smoothing_params, dtype=np.float64)
        group_map = dict(_compiled_metadata(model).get("s_id_to_sp_indices", {}) or {})
        unknown = sorted(str(key) for key in sp.keys() if str(key) not in group_map)
        if unknown:
            raise ValueError(f"Unknown smoothing id(s) in smoothing_params: {unknown}.")
        for key, value in sp.items():
            indices = list(group_map[str(key)])
            vals = np.asarray(value, dtype=np.float64).ravel()
            if vals.ndim == 0 or vals.size == 1:
                if len(indices) != 1:
                    raise ValueError(
                        f"smoothing_params[{key!r}] must provide {len(indices)} "
                        "values for this multi-penalty smoothing id."
                    )
                out[indices[0]] = float(vals.reshape(-1)[0])
                continue
            if vals.shape != (len(indices),):
                raise ValueError(
                    f"smoothing_params[{key!r}] must have shape ({len(indices)},), "
                    f"got {vals.shape}."
                )
            out[np.asarray(indices, dtype=int)] = vals
        sp = out
    else:
        sp = np.asarray(sp, dtype=np.float64)
        if sp.ndim == 0:
            sp = np.full(n_smoothing_params, float(sp), dtype=np.float64)
        if sp.shape != (n_smoothing_params,):
            raise ValueError(
                f"smoothing_params must have shape ({n_smoothing_params},), got {sp.shape}"
            )
        sp = sp.copy()

    fixed_mask = np.zeros(n_smoothing_params, dtype=bool)
    compiled_model = _compiled_model(model)
    override_modes = (
        None
        if compiled_model is None
        else getattr(compiled_model, "smoothing_override_modes", None)
    )
    override_values = (
        None
        if compiled_model is None
        else getattr(compiled_model, "smoothing_override_values", None)
    )
    if override_modes is not None:
        if len(override_modes) != n_smoothing_params:
            raise ValueError(
                "CompiledPredictor smoothing_override_modes has incompatible length."
            )
        if override_values is None:
            override_values = np.full(n_smoothing_params, np.nan, dtype=np.float64)
        override_values = np.asarray(override_values, dtype=np.float64)
        if override_values.shape != (n_smoothing_params,):
            raise ValueError(
                "CompiledPredictor smoothing_override_values has incompatible shape."
            )
        for i, mode in enumerate(override_modes):
            if mode is None:
                continue
            if mode == "fixed":
                val = float(override_values[i])
                if not np.isfinite(val) or val < 0:
                    raise ValueError(
                        f"Fixed smoothing parameter override at index {i} "
                        f"must be finite and >= 0, got {val}."
                    )
                sp[i] = val
                fixed_mask[i] = True
            elif mode == "estimate":
                fixed_mask[i] = False
                if (not np.isfinite(sp[i])) or (sp[i] <= 0):
                    sp[i] = 1.0
            else:
                raise ValueError(
                    f"Unknown smoothing override mode {mode!r} at index {i}."
                )

    min_sp = (
        np.zeros(n_smoothing_params, dtype=np.float64)
        if model.min_sp_ is None
        else np.asarray(model.min_sp_, dtype=np.float64)
    )
    if min_sp.shape != (n_smoothing_params,):
        raise ValueError(
            f"min_sp must have shape ({n_smoothing_params},), got {min_sp.shape}"
        )
    if np.any(fixed_mask & (sp < min_sp)):
        raise ValueError(
            "Fixed smoothing parameters must satisfy the configured min_sp lower bounds."
        )

    sp = np.maximum(sp, min_sp)
    free_mask = ~fixed_mask
    if np.any(~np.isfinite(sp[free_mask])) or np.any(sp[free_mask] <= 0):
        raise ValueError("All free smoothing parameters must be finite and > 0.")
    if np.any(~np.isfinite(sp[fixed_mask])) or np.any(sp[fixed_mask] < 0):
        raise ValueError("All fixed smoothing parameters must be finite and >= 0.")

    model.smoothing_fixed_mask_ = fixed_mask
    model.smoothing_override_modes_ = (
        None if override_modes is None else list(override_modes)
    )
    return sp


def expand_smoothing_params_from_log(model, log_free_sp):
    from ..smoothing_selection.optimize import expand_smoothing_params_from_log

    return expand_smoothing_params_from_log(model, log_free_sp)


def optimize_smoothing_params(
    model, y, initial_smoothing_params=None, method="gcv", optimizer="lbfgsb"
):
    from ..smoothing_selection import optimize_smoothing_params as _optimize

    return _optimize(
        model,
        y=y,
        initial_smoothing_params=initial_smoothing_params,
        method=method,
        optimizer=optimizer,
    )
