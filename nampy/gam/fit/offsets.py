"""
Offset utilities for the fitting pipeline.

Offsets are additive constants on the link scale included in the linear predictor:
``eta = offset + X beta``.  This module provides:

- :func:`coerce_offset_array`: validate and coerce an offset to a 1-D float64 array.
- :func:`resolve_prediction_offset`: determine which offset to use at prediction time
  (explicit argument > model default > none).
"""

import numpy as np

from ..data import coerce_optional_offset, combine_offsets, copy_offset
from ..model_state import _predictor_designs, _term_blocks_seq


def smooth_offsets(model, X=None, *, skip_term_ids=()):
    """Sum smooth-owned offsets, as gam.setup and predict.gam do in mgcv.

    Known soap boundary values are part of the smooth's predictor even when
    the caller supplies another offset. They are not estimated coefficients.
    """
    n_predictors = len(_predictor_designs(model))
    offsets = [None] * max(1, n_predictors)
    for term in _term_blocks_seq(model):
        if term.term_id in skip_term_ids:
            continue
        if X is None:
            value = term.offset_train
        else:
            value = None if term.offset_fn is None else term.offset_fn(X)
        if value is None:
            continue
        for index in term.predictor_indices:
            offsets[index] = combine_offsets(offsets[index], value)
    if all(value is None for value in offsets):
        return None
    return offsets if n_predictors > 1 else offsets[0]


def coerce_offset_array(offset, n_rows, *, name="offset"):
    if offset is None:
        return None
    if isinstance(offset, (list, tuple)):
        return [
            (
                None
                if off is None
                else coerce_optional_offset(off, n_rows, name=f"{name}[{i}]")
            )
            for i, off in enumerate(offset)
        ]
    out = np.asarray(offset, dtype=np.float64).ravel()
    if out.shape != (int(n_rows),):
        raise ValueError(f"{name} must have shape ({int(n_rows)},), got {out.shape}.")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return out


def resolve_prediction_offset(model, X, offset, *, skip_term_ids=()):
    """
    Offset semantics: additive on the linear predictor scale
    --------------------------
    - Offsets used only via a separate fit-time `offset=` argument are not
      automatically reused for prediction.
    - Formula offsets remembered by the wrapper may be reused for in-sample
      prediction via model.offset_predict_default_.
    - Explicit predict-time offset always wins.
    """
    if X is None:
        if offset is None:
            default_offset = getattr(model, "offset_predict_default_", None)
            return combine_offsets(
                copy_offset(default_offset),
                smooth_offsets(model, skip_term_ids=skip_term_ids),
            )
        return combine_offsets(
            coerce_offset_array(offset, model.n_samples_),
            smooth_offsets(model, skip_term_ids=skip_term_ids),
        )

    return combine_offsets(
        coerce_offset_array(offset, len(X)) if offset is not None else None,
        smooth_offsets(model, X, skip_term_ids=skip_term_ids),
    )
