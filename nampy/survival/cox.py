"""Cox baseline estimation from fitted relative-risk predictors.

Breslow event/censor ties match mgcv/src/coxph.c::coxpp: everyone observed
at an event time belongs to its risk set, including censored subjects.

References
----------
- Cox, D. R. (1972). Regression Models and Life-Tables.
  https://doi.org/10.1111/j.2517-6161.1972.tb00899.x
- Breslow, N. (1974). Covariance Analysis of Censored Survival Data.
  https://www.jstor.org/stable/2529620
"""

import numpy as np


def baseline_from_predictors(predictors, time, event, strata):
    """Return right-continuous cumulative baseline hazards by integer stratum."""
    with np.errstate(over="raise", invalid="raise"):
        gamma = np.exp(predictors)
    if np.any(gamma <= 0) or not np.isfinite(gamma).all():
        raise FloatingPointError("Cox risk weights exceed finite numerical range.")
    result = {}
    for label in np.unique(strata):
        take = strata == label
        times = np.unique(time[take])
        increments = []
        for t in times:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                total = np.sum(gamma[take & (time >= t)])
                if not np.isfinite(total) or total <= 0:
                    raise FloatingPointError("Cox risk totals exceed finite positive numerical range.")
                increments.append(np.sum(event[take & (time == t)]) / total)
        with np.errstate(over="raise", invalid="raise"):
            result[int(label)] = (np.r_[0.0, times], np.r_[0.0, np.cumsum(increments)])
    return result
