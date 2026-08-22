"""Small helpers shared by PIRLS criterion implementations."""

import numpy as np


def _prior_weights(model, y):
    """Return model prior weights, defaulting to unit weights."""
    weights = getattr(model, "prior_weights_", None)
    if weights is None:
        return np.ones_like(np.asarray(y, dtype=np.float64), dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)
