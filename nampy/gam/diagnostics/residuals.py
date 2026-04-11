from __future__ import annotations

import numpy as np

from .._mgcv_constants import LOG_GUARD_MIN
from .._model_state import _require_fitted


def _prior_weights(model) -> np.ndarray:
    w = getattr(model, "prior_weights_", None)
    if w is None:
        return np.ones(int(model.n_samples_), dtype=np.float64)
    return np.asarray(w, dtype=np.float64).ravel()


def _deviance_residuals(model) -> np.ndarray:
    y = np.asarray(model.y_, dtype=np.float64).ravel()
    mu = np.asarray(model._fitted_mu, dtype=np.float64).ravel()
    w = _prior_weights(model)
    sign = np.sign(y - mu)
    sign[sign == 0.0] = 1.0

    # Per-observation deviance contributions (mgcv `family$dev.resids` analogue).
    dev_obs = model.family.deviance_obs(y, mu, w)
    return sign * np.sqrt(np.clip(dev_obs, 0.0, None))


def residuals_gam(model, type: str = "deviance") -> np.ndarray:
    _require_fitted(model)

    type = str(type).lower()
    y = np.asarray(model.y_, dtype=np.float64).ravel()
    if getattr(model.family, "family_class", "") == "general":
        fitted = np.asarray(model._fitted_mu, dtype=np.float64)
        if fitted.ndim == 1:
            fitted = fitted[:, None]
        family_residuals = getattr(model.family, "residuals", None)
        if callable(family_residuals):
            return np.asarray(
                family_residuals(y, fitted, rtype=type), dtype=np.float64
            ).ravel()
        if type == "response":
            return y - np.asarray(fitted[:, 0], dtype=np.float64).ravel()
        raise NotImplementedError(
            f"Residual type {type!r} is not implemented for general family {model.family.name!r}."
        )

    mu = np.asarray(model._fitted_mu, dtype=np.float64).ravel()
    eta = np.asarray(model._fitted_eta, dtype=np.float64).ravel()
    w = _prior_weights(model)

    if type == "response":
        return y - mu
    if type == "working":
        fit_state = getattr(model, "fit_state_", None)
        z_work = (
            None if fit_state is None else getattr(fit_state, "working_response", None)
        )
        if z_work is not None:
            z_work = np.asarray(z_work, dtype=np.float64).ravel()
            offset = getattr(fit_state, "offset", None)
            eta_base = (
                eta
                if offset is None
                else eta - np.asarray(offset, dtype=np.float64).ravel()
            )
            return z_work - eta_base
        mu_eta = np.asarray(model.family.mu_eta(eta), dtype=np.float64)
        return (y - mu) / np.clip(mu_eta, LOG_GUARD_MIN, None)
    if type == "deviance":
        return _deviance_residuals(model)
    if type in {"pearson", "scaled.pearson"}:
        var = np.asarray(model.family.variance(mu), dtype=np.float64)
        res = (y - mu) * np.sqrt(w) / np.sqrt(np.clip(var, LOG_GUARD_MIN, None))
        if type == "scaled.pearson":
            res = res / np.sqrt(max(float(model.scale_), LOG_GUARD_MIN))
        return res

    raise ValueError(
        "type must be one of {'deviance', 'pearson', 'scaled.pearson', 'working', 'response'}."
    )


__all__ = ["residuals_gam"]
