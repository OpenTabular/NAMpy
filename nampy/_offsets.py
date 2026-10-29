"""Known offsets in row-aligned additive predictor coordinates."""

import numpy as np


def predictor_offset(offset, n_rows, n_predictors, *, allow_missing=False):
    """Validate exact axes; a vector is accepted only for one predictor."""
    if offset is None:
        return None
    value = np.asarray(offset, dtype=float)
    if value.ndim == 1 and n_predictors == 1:
        value = value[:, None]
    valid = ~np.isinf(value) if allow_missing else np.isfinite(value)
    if value.shape != (n_rows, n_predictors) or not valid.all():
        raise ValueError(
            f"offset must be finite with shape ({n_rows}, {n_predictors}); "
            "no predictor broadcasting is supported."
        )
    return value
