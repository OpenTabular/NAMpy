"""Structured lag data and design construction."""

from .lag_basis import DistributedLagBasis
from .lags import (
    cross_basis,
    cross_basis_penalty,
    cumulative_lag_matrix,
    exposure_history,
)

__all__ = [
    "DistributedLagBasis",
    "cross_basis",
    "cross_basis_penalty",
    "cumulative_lag_matrix",
    "exposure_history",
]
