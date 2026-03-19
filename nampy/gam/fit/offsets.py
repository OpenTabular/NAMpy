import numpy as np


def coerce_offset_array(offset, n_rows, *, name="offset"):
    if offset is None:
        return None
    out = np.asarray(offset, dtype=np.float64).ravel()
    if out.shape != (int(n_rows),):
        raise ValueError(f"{name} must have shape ({int(n_rows)},), got {out.shape}.")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return out


def resolve_prediction_offset(model, X, offset):
    """
    mgcv-like offset semantics
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
            return None if default_offset is None else np.asarray(default_offset, dtype=np.float64)
        return coerce_offset_array(offset, model.n_samples_)

    return coerce_offset_array(offset, len(X)) if offset is not None else None