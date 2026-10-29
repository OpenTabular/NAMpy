"""Covariate-shift weighted conformal quantiles, including the test atom at infinity.

References
----------
- Tibshirani, R. J., Barber, R. F., Candès, E. J., and Ramdas, A. (2019). Conformal
  Prediction Under Covariate Shift.
  https://arxiv.org/abs/1904.06019
"""

from math import ceil

import numpy as np

from .conformal import _vector


def density_ratios(value, *, positive=False):
    ratios = _vector(value, "density_ratio")
    if np.any(ratios <= 0 if positive else ratios < 0):
        raise ValueError("Density ratios must be nonnegative; test-point ratios must be positive.")
    ratios.setflags(write=False)
    return ratios


def weighted_corrections(scores, calibration_ratio, prediction_ratio, level):
    """Left inverse of the weighted empirical CDF plus test-point mass at +inf.

    Ratios are dP_target(X)/dP_calibration(X), not case/frequency weights.
    Conditional Y|X must be unchanged. Ratios use a common arbitrary scale.
    Long-double accumulation avoids overflow and premature weight underflow;
    CDF comparisons use floating-point probabilities, as in the unweighted API.
    No estimated-ratio consistency or conditional coverage is asserted here.
    """
    test = density_ratios(prediction_ratio, positive=True)
    weights = density_ratios(calibration_ratio)
    if len(weights) != len(scores):
        raise ValueError("Calibration density ratios must align with scores.")
    order = np.argsort(scores, kind="stable")
    active = weights[order] > 0
    ordered = np.asarray(scores)[order][active]
    result = np.full(len(test), np.inf)
    if not len(ordered):
        return result
    scale = np.longdouble(weights.max())
    mass = weights[order][active].astype(np.longdouble)/scale
    cumulative = np.cumsum(mass)
    # Search the actual CDF, retaining the query mass in its denominator.
    # Renormalizing the requested probability first can move exact rank
    # boundaries by an entire score because of floating-point rounding.
    # Scaling keeps accumulation finite even on platforms where longdouble
    # has float64 range. Overflow of a test/calibration mass ratio then means
    # all representable finite-score CDF probabilities are zero: keep +inf.
    with np.errstate(over="ignore", under="ignore"):
        denominator = cumulative[-1]+test.astype(np.longdouble)/scale
    left, right = np.zeros(len(test), dtype=int), np.full(len(test), len(ordered), dtype=int)
    while np.any(left < right):
        active_query = left < right
        mid = (left+right)//2
        probability = np.asarray(cumulative[np.minimum(mid, len(ordered)-1)]/denominator, dtype=float)
        advance = active_query & (probability < level)
        left[advance] = mid[advance]+1
        right[active_query & ~advance] = mid[active_query & ~advance]
    indices = left
    # Constant ratios reduce algebraically to the original discrete-rank rule.
    # Use that rule directly rather than a second, rounded CDF representation.
    if np.all(weights == weights[0]):
        same = test == weights[0]
        indices[same] = ceil((len(scores)+1)*level)-1
    finite = indices < len(ordered)
    result[finite] = ordered[indices[finite]]
    return result
