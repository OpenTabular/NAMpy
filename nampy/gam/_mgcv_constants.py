"""Frozen mgcv numerical constants.

Reference target: mgcv 1.9-1.
"""

EIG_TOL_POWER = 0.8
PENALTY_RIDGE_REL = 1e-6
GAMMA_ABSTOL = 1e-12
QR_TOL_SCALE = 1.0

# Default GLM / IRLS floor (family.eps, binomial probability clips, etc.).
FAMILY_EPS = 1e-9

# Symmetric bound on |eta| before exp / inverse-logit (overflow guard).
LINK_ETA_EXP_CLIP = 700.0

# log(max(x, LOG_GUARD_MIN)) and similar underflow guards in the mgcv port.
LOG_GUARD_MIN = 1e-300

# Smoothing optimizer: absolute floor in score_tol = max(floor, scale * FAMILY_EPS).
SMOOTHING_SCORE_ABS_FLOOR = 1e-8

__all__ = [
    "EIG_TOL_POWER",
    "PENALTY_RIDGE_REL",
    "GAMMA_ABSTOL",
    "QR_TOL_SCALE",
    "FAMILY_EPS",
    "LINK_ETA_EXP_CLIP",
    "LOG_GUARD_MIN",
    "SMOOTHING_SCORE_ABS_FLOOR",
]
