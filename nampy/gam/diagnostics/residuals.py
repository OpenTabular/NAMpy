from __future__ import annotations

import warnings

import numpy as np

from .._mgcv_constants import LOG_GUARD_MIN
from .._model_state import (
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
    sign[sign == 0.0] = 1.0

    # Per-observation deviance contributions (mgcv `family$dev.resids` analogue).
    dev_obs = model.family.deviance_obs(y, mu, w)
    return sign * np.sqrt(np.clip(dev_obs, 0.0, None))


def _general_family_residual_fallback(
    model,
    y: np.ndarray,
    fitted: np.ndarray,
    eta: np.ndarray | None,
) -> np.ndarray:
    """Conservative multi-predictor fallback mirroring the base GAMLSS contract."""
    mu = None

    eta_arr = None if eta is None else np.asarray(eta, dtype=np.float64)
    if eta_arr is not None and eta_arr.ndim == 1:
        eta_arr = eta_arr[:, None]
    linfo = getattr(model.family, "linfo", None)
    if (
        eta_arr is not None
        and eta_arr.ndim == 2
        and eta_arr.shape[0] == y.size
        and eta_arr.shape[1] >= 1
        and linfo
        and len(linfo) >= 1
        and hasattr(linfo[0], "linkinv")
    ):
        mu = np.asarray(linfo[0].linkinv(eta_arr[:, 0]), dtype=np.float64).ravel()

    if mu is None:
        fitted_arr = np.asarray(fitted, dtype=np.float64)
        if fitted_arr.ndim == 1:
            mu = fitted_arr.ravel()
        elif fitted_arr.ndim == 2 and fitted_arr.shape[0] == y.size and fitted_arr.shape[1] >= 1:
            mu = np.asarray(fitted_arr[:, 0], dtype=np.float64).ravel()
        else:
            raise NotImplementedError(
                "Residual fallback requires fitted values with a primary predictor column."
            )

    return y - mu


def residuals_gam(model, type: str = "deviance") -> np.ndarray:
    _require_fitted(model)

    type = str(type).lower()
    y = np.asarray(model.y_, dtype=np.float64).ravel()
    fitted = np.asarray(_fitted_mu(model), dtype=np.float64)
    eta_fitted = _fitted_eta(model)
    is_general_family = (
        str(getattr(getattr(model, "family", None), "family_class", "")).lower()
        == "general"
    )
    family_residuals = getattr(model.family, "residuals", None)
    if callable(family_residuals):
        try:
            try:
                return np.asarray(
                    family_residuals(y, fitted, rtype=type, eta=eta_fitted),
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
            if is_general_family:
                return _general_family_residual_fallback(model, y, fitted, eta_fitted)
            raise

    mu = np.asarray(fitted, dtype=np.float64).ravel()
    eta = np.asarray(eta_fitted, dtype=np.float64).ravel()
    w = _prior_weights(model)

    if type == "response":
        if fitted.ndim == 1:
            return y - fitted
        return np.asarray(y.reshape(-1, 1) - fitted, dtype=np.float64).ravel()
    if type == "working":
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
                return z_work + np.asarray(offset, dtype=np.float64).ravel() - eta
            # mgcv::residuals.gam() returns the stored working residual series,
            # equivalent to z - eta in the fitted parameterization.
            return z_work - eta
        mu_eta = getattr(model.family, "mu_eta", None)
        if callable(mu_eta):
            mu_eta_val = np.asarray(mu_eta(eta), dtype=np.float64)
            return (y - mu) / np.clip(mu_eta_val, LOG_GUARD_MIN, None)
        if fitted.ndim == 1:
            return y - fitted
        return np.asarray(y.reshape(-1, 1) - fitted, dtype=np.float64).ravel()
    if type == "deviance":
        if fitted.ndim != 1:
            if is_general_family:
                return _general_family_residual_fallback(model, y, fitted, eta_fitted)
            raise NotImplementedError(
                f"Residual type {type!r} is not implemented for general family {model.family.name!r}."
            )
        return _deviance_residuals(model)
    if type in {"pearson", "scaled.pearson"}:
        variance = getattr(model.family, "variance", None)
        if not callable(variance):
            if fitted.ndim != 1 and is_general_family:
                return _general_family_residual_fallback(model, y, fitted, eta_fitted)
            warnings.warn(
                "Pearson residuals not available for this family - returning deviance residuals",
                RuntimeWarning,
                stacklevel=2,
            )
            return residuals_gam(model)
        var = np.asarray(variance(mu), dtype=np.float64)
        res = (y - mu) * np.sqrt(w) / np.sqrt(np.clip(var, LOG_GUARD_MIN, None))
        if type == "scaled.pearson":
            res = res / np.sqrt(max(float(_fit_scale(model)), LOG_GUARD_MIN))
        return res

    raise ValueError(
        "type must be one of {'deviance', 'pearson', 'scaled.pearson', 'working', 'response'}."
    )


__all__ = ["residuals_gam"]
