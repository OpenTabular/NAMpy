"""Quantile diagnostic data following mgcv R/plots.r::qq.gam."""

from __future__ import annotations

import copy
import warnings

import numpy as np
from scipy import stats

from ..model_state import _fit_scale, _fitted_mu, _require_fitted
from .residuals import _prior_weights, residuals_gam


def _family_quantiles(model, p):
    family = model.family
    mu, wt, scale = _fitted_mu(model), _prior_weights(model), _fit_scale(model)
    custom = getattr(family, "qf", None)
    if callable(custom):
        return custom(p, mu, wt, scale)
    name = family.name.lower()
    if name == "gaussian":
        return stats.norm.ppf(p, loc=mu, scale=np.sqrt(scale / wt))
    if name == "poisson":
        return stats.poisson.ppf(p, mu)
    if name == "binomial":
        if not np.allclose(wt, np.ceil(wt), rtol=1.49e-8, atol=1.49e-8):
            warnings.warn(
                "non-integer binomial denominator: quantiles incorrect",
                RuntimeWarning,
                stacklevel=3,
            )
            wt = np.ceil(wt)
        return stats.binom.ppf(p, wt, mu) / (wt + (wt == 0))
    if name == "gamma":
        return stats.gamma.ppf(p, 1 / scale, scale=mu * scale)
    if name == "negbin":
        return stats.nbinom.ppf(p, family.theta, family.theta / (family.theta + mu))
    if name == "betar":
        values = stats.beta.ppf(
            p, family.getTheta(True) * mu, family.getTheta(True) * (1 - mu)
        )
        return np.clip(values, family.eps, 1 - family.eps)
    return None


def _family_draw(model, rng):
    family = model.family
    mu, wt, scale = _fitted_mu(model), _prior_weights(model), _fit_scale(model)
    custom = getattr(family, "rd", None)
    if callable(custom):
        return custom(mu, wt, scale, rng=rng)
    name = family.name.lower()
    if name == "gaussian":
        return rng.normal(mu, np.sqrt(scale / wt))
    if name == "poisson":
        return rng.poisson(mu)
    if name == "binomial":
        if not np.allclose(wt, np.rint(wt), rtol=0, atol=1e-7):
            warnings.warn(
                "non-integer binomial denominator: simulation produced NaNs",
                RuntimeWarning,
                stacklevel=3,
            )
        valid = np.abs(wt - np.rint(wt)) <= 1e-7
        result = rng.binomial(np.rint(wt).astype(int), mu) / (wt + (wt == 0))
        return np.where(valid, result, np.nan)
    if name == "gamma":
        return rng.gamma(1 / scale, mu * scale)
    if name == "inverse_gaussian":
        return rng.wald(mu, 1 / scale)
    if name == "gaulss":
        return rng.normal(mu[:, 0], np.sqrt(scale / wt) / mu[:, 1])
    if name == "gammals":
        phi = np.exp(mu[:, 1])
        return rng.gamma(1 / phi, mu[:, 0] * phi)
    if name == "tw":
        power = family.p
        if power == 2:
            return rng.gamma(1 / scale, mu * scale)
        count = rng.poisson(mu ** (2 - power) / ((2 - power) * scale))
        shape = (2 - power) / (power - 1)
        gamma_scale = scale * (power - 1) * mu ** (power - 1)
        return rng.gamma(count * shape, gamma_scale)
    if name == "ocat":
        latent = mu + rng.logistic(size=len(mu))
        return (
            np.searchsorted(family._alpha_for_deviance()[1:-1], latent, side="left") + 1
        )
    if name == "negbin":
        return rng.negative_binomial(family.theta, family.theta / (family.theta + mu))
    if name == "betar":
        return np.clip(
            rng.beta(family.getTheta(True) * mu, family.getTheta(True) * (1 - mu)),
            family.eps,
            1 - family.eps,
        )
    return None


def _normal_qq(residuals, type):
    observed = np.sort(np.asarray(residuals, dtype=float))
    n = len(observed)
    a = 0.375 if n <= 10 else 0.5
    return {
        "theoretical": stats.norm.ppf((np.arange(1, n + 1) - a) / (n + 1 - 2 * a)),
        "residuals": observed,
        "simulated_residuals": None,
        "lower": None,
        "upper": None,
        "reference": "normal",
        "rep": 0,
        "level": 0,
        "type": type,
        "xlab": "Theoretical Quantiles",
        "ylab": f"{type} residuals",
    }


def prepare_qq_gam_data(
    model,
    *,
    rep=0,
    level=0.9,
    s_rep=10,
    type="deviance",
    seed=None,
    quantile_permutations=None,
    response_draws=None,
):
    """Prepare qq.gam quantiles and envelopes without changing fitted state.

    ``seed`` controls NumPy's generator, not R's random stream. Optional
    ``quantile_permutations`` (zero-based row indices, shape s_rep by n) and
    ``response_draws`` (shape rep by n) allow reproducible external draws.
    """
    _require_fitted(model)
    if type not in {"deviance", "pearson", "response"}:
        raise ValueError("type must be deviance, pearson or response")
    for name, value, minimum in (("rep", rep, 0), ("s_rep", s_rep, 1)):
        if not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(level):
        raise ValueError("level must be finite")
    rep, s_rep = int(rep), int(s_rep)
    observed = residuals_gam(model, type=type)
    n = len(observed)
    probabilities = (np.arange(n) + 0.5) / n
    rng = np.random.default_rng(seed)
    original_rep = rep
    if rep == 0:
        # qf availability is distinct from rd availability in upstream.
        known = model.family.name.lower() in {
            "gaussian",
            "poisson",
            "binomial",
            "gamma",
            "negbin",
            "betar",
        }
        if not known and not callable(getattr(model.family, "qf", None)):
            rep = 50
        level = 0
    if quantile_permutations is not None:
        permutations = np.asarray(quantile_permutations)
        if (
            rep
            or permutations.shape != (s_rep, n)
            or not np.all(np.sort(permutations, axis=1) == np.arange(n))
        ):
            raise ValueError(
                "quantile_permutations must contain s_rep permutations of zero-based observation indices"
            )
        permutations = permutations.astype(int)
    else:
        permutations = None
    draws = None if response_draws is None else np.asarray(response_draws, dtype=float)
    if draws is not None and (not rep or draws.shape != (rep, n)):
        raise ValueError("response_draws must have shape (rep, n_observations)")
    simulated = np.empty((n, rep if rep else s_rep))
    proxy = copy.copy(model)
    for i in range(simulated.shape[1]):
        if rep:
            response = draws[i] if draws is not None else _family_draw(model, rng)
        else:
            probabilities = (
                probabilities[rng.permutation(n)]
                if permutations is None
                else (permutations[i] + 0.5) / n
            )
            response = _family_quantiles(model, probabilities)
        if response is None:
            return _normal_qq(observed, type)
        proxy.y_ = np.asarray(response, dtype=float)
        # qq.gam replaces object$y before evaluating each residual series.
        # The normalized response owner must follow that replacement too;
        # otherwise deviance residuals retain the original response magnitude.
        proxy.response_ = proxy.y_
        simulated[:, i] = np.sort(residuals_gam(proxy, type=type))
    lower = upper = None
    if rep:
        theoretical = np.quantile(simulated, (np.arange(n) + 0.5) / n, method="linear")
        if 0 < level < 1:
            lower, upper = np.quantile(
                simulated, [(1 - level) / 2, (1 + level) / 2], axis=1, method="linear"
            )
    else:
        theoretical = np.sort(np.mean(simulated, axis=1))
    limits = np.asarray(observed)
    if lower is not None:
        limits = np.concatenate([limits, lower, upper])
    elif rep and level >= 1:
        # qq.gam stores lim <- level in the all-replicate-lines branch.
        limits = np.append(limits, level)
    return {
        "ylim": np.array([np.min(limits), np.max(limits)]),
        "theoretical": theoretical,
        "residuals": np.sort(observed),
        "simulated_residuals": simulated,
        "lower": lower,
        "upper": upper,
        "reference": "identity",
        "rep": rep,
        "requested_rep": original_rep,
        "level": level,
        "type": type,
        "xlab": "theoretical quantiles",
        "ylab": f"{type} residuals",
    }
