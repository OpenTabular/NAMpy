"""R nmath dgamma log-density construction (GPL-2-or-later).

Upstream dgamma.c by Catherine Loader. Copyright (C) 2000-2019 R Core Team,
2004-2019 R Foundation. Python translation for NAMpy; redistribution and
modification permitted under GNU GPL version 2 or later, WITHOUT ANY WARRANTY.
See R_MATH_LICENSE.txt for the complete license.
"""

import math

import numpy as np

from ._r_poisson_math import log_dpois_raw_scalar


def _log_dgamma_scalar(x, shape, scale):
    if math.isnan(x) or math.isnan(shape) or math.isnan(scale):
        return np.nan
    if shape < 0 or scale <= 0:
        return np.nan
    if x < 0:
        return -np.inf
    if shape == 0:
        return np.inf if x == 0 else -np.inf
    if x == 0:
        if shape < 1:
            return np.inf
        return -np.inf if shape > 1 else -math.log(scale)
    if shape < 1:
        density = log_dpois_raw_scalar(shape, x / scale)
        ratio = shape / x
        correction = (
            -np.inf
            if ratio == 0
            else (
                math.log(ratio)
                if math.isfinite(ratio)
                else math.log(shape) - math.log(x)
            )
        )
        return density + correction
    return log_dpois_raw_scalar(shape - 1, x / scale) - math.log(scale)


def log_dgamma(x, shape, scale):
    """Broadcast R's stable Gamma log density over observations and means."""
    x, shape, scale = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(shape, float), np.asarray(scale, float)
    )
    return np.fromiter(
        (
            _log_dgamma_scalar(float(y), float(a), float(s))
            for y, a, s in zip(x.flat, shape.flat, scale.flat, strict=True)
        ),
        float,
        count=x.size,
    ).reshape(x.shape)
