"""Cox survival prediction uncertainty in fitted coefficient coordinates."""

import numpy as np


def prediction_standard_error(
    matrix,
    eta,
    survival,
    times,
    labels,
    strata,
    baseline,
    baseline_uncertainty,
    covariance,
):
    """mgcv coxpred baseline/coefficient covariance, without estimator ownership."""
    lookup = {label: i for i, label in enumerate(strata)}
    se = np.empty_like(survival)
    for row, label in enumerate(labels):
        code = lookup[label]
        t, q, a = baseline_uncertainty[code]
        index = np.searchsorted(t, times, side="right") - 1
        h = baseline[code][1][index]
        v = a[index] - h[:, None] * matrix[row]
        variance = q[index] + np.einsum("tp,pq,tq->t", v, covariance, v)
        if np.any(variance < 0) or not np.isfinite(variance).all():
            raise FloatingPointError(
                "Cox baseline/coefficient prediction variance is invalid."
            )
        se[row] = np.exp(eta[row]) * survival[row] * np.sqrt(variance)
    return se
