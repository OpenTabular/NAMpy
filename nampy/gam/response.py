"""Explicit observation metadata for censored extended-family responses."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class CensoredResponse:
    """Primary observations and paired source censor bounds.

    Bounds retain their original ordering and infinities. Their interpretation
    belongs to the family (in particular, bcg differs from cnorm/clog/cpois).
    There is deliberately no implicit NumPy conversion that could lose bounds.
    """

    values: np.ndarray
    bounds: np.ndarray

    def __post_init__(self):
        values = np.array(self.values, dtype=np.float64, copy=True)
        bounds = np.array(self.bounds, dtype=np.float64, copy=True)
        if values.ndim != 1 or bounds.shape != values.shape:
            raise ValueError("Censored values and bounds must be aligned vectors.")
        if not np.isfinite(values).all() or np.isnan(bounds).any():
            raise ValueError("Censored values must be finite and bounds cannot be NaN.")
        values.setflags(write=False)
        bounds.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bounds", bounds)

    def __len__(self):
        return len(self.values)

    def subset(self, index):
        """Preserve pairing through masks, permutations and repeated indices."""
        return CensoredResponse(
            np.atleast_1d(self.values[index]), np.atleast_1d(self.bounds[index])
        )

    def __reduce__(self):
        # Reapply validation and read-only flags when restoring fitted state.
        return type(self), (self.values, self.bounds)


def normalize_censored_response(y):
    """Separate vector/exact and two-column upstream response encodings."""
    if isinstance(y, CensoredResponse):
        return y
    values = np.asarray(y, dtype=np.float64)
    if values.ndim == 1:
        return CensoredResponse(values, values)
    if values.ndim == 2 and values.shape[1] == 2:
        return CensoredResponse(values[:, 0], values[:, 1])
    raise ValueError("Censored y must be a vector or an (n, 2) response matrix.")


def response_values(y):
    """Numeric response for solver arithmetic, without changing metadata."""
    return y.values if isinstance(y, CensoredResponse) else np.asarray(y, dtype=float)


def family_response(y):
    """Coerce ordinary responses while retaining explicit censor metadata."""
    return y if isinstance(y, CensoredResponse) else np.asarray(y, dtype=np.float64)
