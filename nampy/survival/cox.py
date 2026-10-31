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
from scipy.special import logsumexp


def baseline_from_predictors(predictors, time, event, strata):
    """Return right-continuous cumulative hazards in the supplied predictor frame.

    Callers must use the same predictor reference when combining these hazards
    with new risks. Fitted estimators center within each stratum before calling
    this primitive, so harmless common shifts do not overflow baseline storage.
    """
    predictors = np.asarray(predictors, dtype=float)
    if not np.isfinite(predictors).all():
        raise FloatingPointError("Cox predictors must be finite.")
    result = {}
    for label in np.unique(strata):
        take = strata == label
        times = np.unique(time[take])
        increments = []
        for t in times:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                deaths = np.sum(event[take & (time == t)])
                log_total = logsumexp(predictors[take & (time >= t)])
                increments.append(0.0 if deaths == 0 else np.exp(np.log(deaths) - log_total))
        with np.errstate(over="raise", invalid="raise"):
            result[int(label)] = (np.r_[0.0, times], np.r_[0.0, np.cumsum(increments)])
    return result
