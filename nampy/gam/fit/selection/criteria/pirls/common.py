"""Small helpers shared by PIRLS criterion implementations."""

import numpy as np

from .....response import response_values


def _prior_weights(model, y):
    """Return model prior weights, defaulting to unit weights."""
    weights = getattr(model, "prior_weights_", None)
    if weights is None:
        return np.ones_like(response_values(y), dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)
