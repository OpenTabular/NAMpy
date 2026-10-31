"""Fixed-penalty Cox partial likelihood, following mgcv coxph.c risk sets.

Upstream mgcv/src/coxph.c::coxlpl accumulates all event/censor ties in the
risk set, then subtracts d*log(sum(exp(eta))). coxpp baseline increments use
that same d/risk_sum. There is no Efron/exact-tie substitution or Poisson
approximation here. This owner provides only coefficient derivatives through
order two; smoothing-parameter derivative/inference routes are not implied.

References
----------
- Cox, D. R. (1972). Regression Models and Life-Tables.
  https://doi.org/10.1111/j.2517-6161.1972.tb00899.x
- Breslow, N. (1974). Covariance Analysis of Censored Survival Data.
  https://www.jstor.org/stable/2529620
"""

import numpy as np


def cox_partial_likelihood(beta, matrix, time, event, strata, offset):
    eta = matrix @ beta + offset
    if not np.isfinite(eta).all():
        raise FloatingPointError("Cox predictors must be finite.")
    p = matrix.shape[1]
    value, gradient, hessian = 0.0, np.zeros(p), np.zeros((p, p))
    for label in np.unique(strata):
        rows = np.flatnonzero(strata == label)
        rows = rows[np.argsort(-time[rows], kind="stable")]
        total, first, second = 0.0, np.zeros(p), np.zeros((p, p))
        # Retain ordinary-scale accumulation until rescaling is needed. This
        # avoids gratuitous rounding changes at tightly converged source fits;
        # the half-log range leaves headroom for cumulative risk moments.
        reference = 0.0
        log_limit = 0.5 * np.log(np.finfo(float).max)
        j = 0
        while j < len(rows):
            stop = j + 1
            while stop < len(rows) and time[rows[stop]] == time[rows[j]]:
                stop += 1
            current = rows[j:stop]
            for row in current:
                difference = eta[row] - reference
                if difference > log_limit or (total == 0 and difference < -log_limit):
                    rescale = 0.0 if total == 0 else np.exp(reference - eta[row])
                    total *= rescale
                    first *= rescale
                    second *= rescale
                    reference = eta[row]
                weight = np.exp(eta[row] - reference)
                total += weight
                first += weight * matrix[row]
                second += weight * np.outer(matrix[row], matrix[row])
            deaths = current[event[current]]
            d = len(deaths)
            if d:
                value += (eta[deaths] - reference).sum() - d * np.log(total)
                mean = first / total
                gradient += matrix[deaths].sum(axis=0) - d / total * first
                hessian += -d * second / total + d * np.outer(mean, mean)
            j = stop
    return value, gradient, hessian


def cox_baseline(matrix, beta, time, event, strata, offset):
    """Stratum-specific right-continuous cumulative baseline hazard, coxpp."""
    from ...survival.cox import baseline_from_predictors

    return baseline_from_predictors(matrix @ beta + offset, time, event, strata)


def cox_higher_derivatives(beta, matrix, time, event, strata, offset, *, order=4):
    """Exact third/fourth coefficient derivatives from risk-set cumulants.

    The log risk-sum derivatives are cumulants of design rows under normalized
    risk weights. This is the coefficient-tensor form of co xlpl's directional
    derivatives. Only new Cox routes use this dense calculation.
    """
    if order not in {3, 4}:
        raise ValueError("Cox higher derivative order must be 3 or 4.")
    p = matrix.shape[1]
    third = np.zeros((p, p, p))
    fourth = np.zeros((p, p, p, p)) if order == 4 else None
    eta = matrix @ beta + offset
    if not np.isfinite(eta).all():
        raise FloatingPointError("Cox predictors must be finite.")
    for label in np.unique(strata):
        for t in np.unique(time[(strata == label) & event]):
            risk = (strata == label) & (time >= t)
            deaths = np.sum((strata == label) & (time == t) & event)
            probability = np.exp(eta[risk] - np.max(eta[risk]))
            probability /= probability.sum()
            centered = matrix[risk] - probability @ matrix[risk]
            third -= deaths * np.einsum(
                "n,na,nb,nc->abc", probability, centered, centered, centered
            )
            if fourth is not None:
                covariance = centered.T @ (probability[:, None] * centered)
                moment = np.einsum(
                    "n,na,nb,nc,nd->abcd",
                    probability,
                    centered,
                    centered,
                    centered,
                    centered,
                )
                cumulant = (
                    moment
                    - np.einsum("ab,cd->abcd", covariance, covariance)
                    - np.einsum("ac,bd->abcd", covariance, covariance)
                    - np.einsum("ad,bc->abcd", covariance, covariance)
                )
                fourth -= deaths * cumulant
    return third, fourth


def cox_baseline_uncertainty(matrix, beta, time, event, strata, offset):
    """coxpp baseline martingale variance q and coefficient sensitivity a."""
    with np.errstate(over="raise", invalid="raise"):
        gamma = np.exp(matrix @ beta + offset)
    result = {}
    for label in np.unique(strata):
        times = np.unique(time[strata == label])
        q, a = [], []
        for t in times:
            risk = (strata == label) & (time >= t)
            deaths = np.sum(event[(strata == label) & (time == t)])
            total = np.sum(gamma[risk])
            increment = (deaths / total) / total
            q.append(increment)
            mean = matrix[risk].T @ (gamma[risk] / total)
            a.append(mean * (deaths / total))
        result[int(label)] = (
            np.r_[0.0, times],
            np.r_[0.0, np.cumsum(q)],
            np.vstack([np.zeros(matrix.shape[1]), np.cumsum(a, axis=0)]),
        )
    return result
