"""Checked first derivatives of the positive Box-Cox support normalizer.

The shape derivative differentiates the symmetric latent density integral;
there are no finite differences or approximate CDF gradient substitutions.
"""

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import digamma, expit

from ._boxcox_latent import PowerExponentialLatent, StudentLatent


def normalizer_with_scores(log_sigma, nu, log_tau, *, kind):
    """Return log F(a) and its log-sigma, nu and log-tau derivatives.

    Here a=1/(sigma*abs(nu)). Arrays must be finite and aligned. Adaptive
    integration is vectorized over rows and checked against the independent
    special-function probability. A singular BCT first derivative at nu=0
    with tau<=1 is rejected explicitly.
    """
    log_sigma, nu, log_tau = np.broadcast_arrays(log_sigma, nu, log_tau)
    if not all(np.isfinite(v).all() for v in (log_sigma, nu, log_tau)):
        raise ValueError("Box-Cox normalizer requires finite predictors.")
    latent = StudentLatent() if kind == "bct" else PowerExponentialLatent()
    tau = np.exp(log_tau)
    zero = nu == 0
    if kind == "bct" and np.any(zero & (tau <= 1)):
        raise NotImplementedError("BCT gradient at nu=0 requires tau>1.")
    result = [np.zeros_like(nu) for _ in range(4)]
    if np.all(zero):
        return tuple(result)
    shape = tau[~zero]
    log_a = -log_sigma[~zero] - np.log(np.abs(nu[~zero]))
    value = latent.logcdf(np.ones_like(log_a), log_a, shape)

    def integrand(s):
        log_z = log_a - s
        log_f = latent.logpdf(log_z, shape)
        if kind == "bct":
            t = 2 * log_z - np.log(shape)
            score = (
                shape / 2 * (digamma((shape + 1) / 2) - digamma(shape / 2))
                - 0.5
                - shape / 2 * np.logaddexp(0, t)
                + (shape + 1) / 2 * expit(t)
            )
        else:
            inverse = 1 / shape
            alpha = PowerExponentialLatent.log_alpha(shape)
            alpha_score = (
                -inverse * digamma(inverse) + 3 * inverse * digamma(3 * inverse)
            ) / 2
            power = shape * (log_z - alpha)
            with np.errstate(over="ignore", invalid="ignore"):
                score = (
                    1
                    - alpha_score
                    + inverse * digamma(inverse)
                    - np.exp(power) * (power - shape * alpha_score)
                )
            score = np.where(np.isneginf(log_f), 0, score)
        density = np.exp(log_f + log_z)
        return np.stack([density, density * score])

    integral, error, info = quad_vec(
        integrand,
        0,
        np.inf,
        epsabs=2e-11,
        epsrel=2e-11,
        norm="max",
        limit=300,
        full_output=True,
    )
    if (
        not info.success
        or error > 2e-9
        or not np.isfinite(integral).all()
        or not np.allclose(np.log(0.5 + integral[0]), value, atol=2e-10, rtol=0)
    ):
        raise NotImplementedError(
            "Box-Cox normalizer gradient integral did not converge."
        )
    boundary = np.exp(log_a + latent.logpdf(log_a, shape) - value)
    result[0][~zero] = value
    result[1][~zero] = -boundary
    result[2][~zero] = -boundary / nu[~zero]
    result[3][~zero] = integral[1] * np.exp(-value)
    return tuple(result)
