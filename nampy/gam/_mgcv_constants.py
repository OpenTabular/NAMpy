"""Frozen mgcv numerical constants.

Reference target: mgcv 1.9-1.
"""

EIG_TOL_POWER = 0.8
PENALTY_RIDGE_REL = 1e-6
GAMMA_ABSTOL = 1e-12
QR_TOL_SCALE = 1.0

__all__ = [
    "EIG_TOL_POWER",
    "PENALTY_RIDGE_REL",
    "GAMMA_ABSTOL",
    "QR_TOL_SCALE",
]
