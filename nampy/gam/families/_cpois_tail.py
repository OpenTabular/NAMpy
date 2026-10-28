"""Stable mean derivatives for concentrated, rare Poisson censoring events."""

import math

import numpy as np


def tail_deviance_derivatives(value, bound, mean):
    """Return four derivatives, or None when a bounded moment sum is unsuitable.

    Conditional count cumulants differentiate the log event probability. Shifting
    counts by the modal allowed count keeps the moment calculations small even
    when the event is thousands of counts into the tail. No tiny probabilities
    or differences of huge factorial moments are formed.
    """
    if not (math.isfinite(value) and math.isfinite(mean) and mean > 0):
        return None
    if value == bound:
        if value < 0 or abs(value - round(value)) > 1e-9 * max(1, abs(value)):
            return None
        lower = upper = int(round(value))
    elif bound == -math.inf:
        lower, upper = 0, math.floor(value + 1e-7)
    elif bound == math.inf:
        lower, upper = max(0, math.floor(value + 1e-7) + 1), None
    elif math.isfinite(bound):
        lower = max(0, math.floor(min(value, bound) + 1e-7) + 1)
        upper = math.floor(max(value, bound) + 1e-7)
    else:
        return None
    if upper is not None and upper < lower:
        return None

    anchor = max(lower, math.floor(mean))
    if upper is not None:
        anchor = min(anchor, upper)
    small_interval = upper is not None and upper - lower <= 64
    upper_tail = anchor == lower and mean / (anchor + 1) <= 0.5
    lower_tail = upper is not None and anchor == upper and anchor / mean <= 0.5
    if not (small_interval or upper_tail or lower_tail):
        return None

    samples = [(0, 1.0)]
    cutoff = np.finfo(float).eps / 16
    for direction in (-1, 1):
        count, weight = anchor, 1.0
        for distance in range(1, 1025):
            if direction < 0:
                if count <= lower:
                    break
                weight *= count / mean
            else:
                if upper is not None and count >= upper:
                    break
                weight *= mean / (count + 1)
            count += direction
            samples.append((direction * distance, weight))
            # With an outward probability ratio <= 1/2 and distance >= 8,
            # the remaining fourth-moment terms are bounded by four times
            # this term. Finite small intervals are summed in full.
            if (
                not small_interval
                and distance >= 8
                and weight * (1 + distance) ** 4 < cutoff
            ):
                break
        else:
            return None

    total = math.fsum(weight for _, weight in samples)
    m1, m2, m3, m4 = (
        math.fsum(weight * offset**order for offset, weight in samples) / total
        for order in range(1, 5)
    )
    k1 = anchor + m1
    k2 = m2 - m1 * m1
    k3 = m3 - 3 * m2 * m1 + 2 * m1 * m1 * m1
    k4 = m4 - 4 * m3 * m1 - 3 * m2 * m2 + 12 * m2 * m1 * m1 - 6 * m1**4
    with np.errstate(all="ignore"):
        inverse = np.float64(1) / mean
        return (
            -2 * (k1 * inverse - 1),
            -2 * (k2 - k1) * inverse**2,
            -2 * (k3 - 3 * k2 + 2 * k1) * inverse**3,
            -2 * (k4 - 6 * k3 + 11 * k2 - 6 * k1) * inverse**4,
        )
