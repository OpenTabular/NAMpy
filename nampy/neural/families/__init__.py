"""Torch training families exported from their owning registration declarations.

Probability queries belong to ``nampy.distributions``. Quantile loss belongs to
``nampy.neural.objectives`` and is not exported as a probability family here.
"""

from .metrics import (
    beta_mean_mse,
    dirichlet_error,
    gamma_deviance,
    inverse_gamma_loss,
    negative_binomial_deviance,
    poisson_deviance,
    student_t_loss,
)
from .registry import FAMILY_REGISTRY

_FAMILY_TYPES = {
    spec.distribution.__name__: spec.distribution
    for spec in FAMILY_REGISTRY.values()
    if spec.distribution.__module__.startswith(__name__ + ".")
}
globals().update(_FAMILY_TYPES)

__all__ = [
    *_FAMILY_TYPES,
    "beta_mean_mse",
    "dirichlet_error",
    "gamma_deviance",
    "inverse_gamma_loss",
    "negative_binomial_deviance",
    "poisson_deviance",
    "student_t_loss",
]
