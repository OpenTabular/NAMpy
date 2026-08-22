from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm

from ..fit.capabilities import has_transformed_coefficients
from ..model_state import (
    _fit_scale,
    _fit_state,
    _fitted_eta,
    _fitted_mu,
    _require_fitted,
)


def _prior_weights(model) -> np.ndarray:
    w = getattr(model, "prior_weights_", None)
    if w is None:
        return np.ones(int(model.n_samples_), dtype=np.float64)
    return np.asarray(w, dtype=np.float64).ravel()


def _deviance_residuals(model) -> np.ndarray:
    y = np.asarray(model.y_, dtype=np.float64).ravel()
    mu = np.asarray(_fitted_mu(model), dtype=np.float64).ravel()
    w = _prior_weights(model)
    sign = np.sign(y - mu)

    # Per-observation deviance contributions (mgcv `family$dev.resids` analogue).
    dev_obs = model.family.deviance_obs(y, mu, w)
    result: np.ndarray = sign * np.sqrt(np.clip(dev_obs, 0.0, None))
    return result


def _quantile_residuals(model, y, mu, weights, *, seed=None) -> np.ndarray:
    scale = float(_fit_scale(model))
    lower, upper = model.family.quantile_residual_bounds(
        y, mu, weights=weights, scale=scale
    )
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    discrete = not np.array_equal(lower, upper)
    if discrete:
        rng = np.random.default_rng(seed)
        cdf = rng.uniform(lower, upper)
        cdf = np.where(cdf > 0.999999, cdf - 1e-16, cdf)
        cdf = np.where(cdf < 0.000001, cdf + 1e-16, cdf)
    else:
        cdf = upper.copy()
    return np.asarray(norm.ppf(cdf), dtype=np.float64)


def residuals_gam(
    model, type: str = "deviance", *, setseed=None
) -> np.ndarray:
    _require_fitted(model)

    type = str(type).lower()
    y = np.asarray(model.y_, dtype=np.float64).ravel()
    fitted = np.asarray(_fitted_mu(model), dtype=np.float64)
    eta_fitted = _fitted_eta(model)
    family_residuals = getattr(model.family, "residuals", None)
    # mgcv/R/mgcv.r::residuals.gam delegates every residual type to a
    # family-owned residual function when one exists. In particular, GAMLSS
    # families must reject residual types outside their own match.arg surface
    # instead of falling through to the default GAM implementation.
    if callable(family_residuals):
        try:
            try:
                return np.asarray(
                    family_residuals(
                        y,
                        fitted,
                        rtype=type,
                        eta=eta_fitted,
                        weights=_prior_weights(model),
                    ),
                    dtype=np.float64,
                ).ravel()
            except TypeError:
                try:
                    return np.asarray(
                        family_residuals(y, fitted, rtype=type),
                        dtype=np.float64,
                    ).ravel()
                except TypeError:
                    return np.asarray(
                        family_residuals(model, type),
                        dtype=np.float64,
                    ).ravel()
        except NotImplementedError:
            raise

    mu = np.asarray(fitted, dtype=np.float64).ravel()
    eta = np.asarray(eta_fitted, dtype=np.float64).ravel()
    w = _prior_weights(model)

    if type == "response":
        if fitted.ndim == 1:
            return y - fitted
        return np.asarray(y.reshape(-1, 1) - fitted, dtype=np.float64).ravel()
    if type == "working":
        if has_transformed_coefficients(model):
            return np.asarray(
                (y - mu) / model.family.mu_eta(eta), dtype=np.float64
            )
        fit_state = _fit_state(model)
        z_work = (
            None if fit_state is None else getattr(fit_state, "working_response", None)
        )
        if z_work is not None:
            z_work = np.asarray(z_work, dtype=np.float64).ravel()
            offset = None if fit_state is None else getattr(fit_state, "offset", None)
            if (
                str(getattr(getattr(model, "family", None), "name", "")).lower()
                == "gaussian"
                and offset is not None
            ):
                # mgcv::residuals.gam() returns the fitted object's stored working
                # residual series. Our Gaussian exact-fit state keeps
                # `working_response = y - offset` for inner-state parity, so restore
                # the fit-time offset before forming the user-facing residuals.
                offset_result: np.ndarray = (
                    z_work + np.asarray(offset, dtype=np.float64).ravel() - eta
                )
                return offset_result
            # mgcv::residuals.gam() returns the stored working residual series,
            # equivalent to z - eta in the fitted parameterization.
            return z_work - eta
        mu_eta = getattr(model.family, "mu_eta", None)
        if callable(mu_eta):
            mu_eta_val = np.asarray(mu_eta(eta), dtype=np.float64)
            result: np.ndarray = (y - mu) / mu_eta_val
            return result
        if fitted.ndim == 1:
            return y - fitted
        return np.asarray(y.reshape(-1, 1) - fitted, dtype=np.float64).ravel()
    if type == "deviance":
        if fitted.ndim != 1:
            raise NotImplementedError(
                f"Residual type {type!r} is not implemented for general family {model.family.name!r}."
            )
        return _deviance_residuals(model)
    if type in {"pearson", "scaled.pearson"}:
        variance = getattr(model.family, "variance", None)
        if not callable(variance):
            warnings.warn(
                "Pearson residuals not available for this family - returning deviance residuals",
                RuntimeWarning,
                stacklevel=2,
            )
            return residuals_gam(model)
        var = np.asarray(variance(mu), dtype=np.float64)
        res = (y - mu) * np.sqrt(w) / np.sqrt(var)
        if type == "scaled.pearson":
            res = res / np.sqrt(float(_fit_scale(model)))
        result = res
        return result
    if type in {"rquantile", "quantile"}:
        return _quantile_residuals(model, y, mu, w, seed=setseed)

    raise ValueError(
        "type must be one of {'deviance', 'pearson', 'scaled.pearson', "
        "'working', 'response', 'rquantile'}."
    )


__all__ = ["residuals_gam"]
