from __future__ import annotations

import numpy as np


def _prior_weights(model) -> np.ndarray:
    w = getattr(model, "prior_weights_", None)
    if w is None:
        return np.ones(int(model.n_samples_), dtype=np.float64)
    return np.asarray(w, dtype=np.float64).ravel()


def _deviance_residuals(model) -> np.ndarray:
    family = str(getattr(model.family, "name", "")).lower()
    y = np.asarray(model.y_, dtype=np.float64).ravel()
    mu = np.asarray(model._fitted_mu, dtype=np.float64).ravel()
    w = _prior_weights(model)
    eps = float(getattr(model.family, "eps", 1e-12))
    sign = np.sign(y - mu)
    sign[sign == 0.0] = 1.0

    if family == "gaussian":
        dev = (y - mu) ** 2
    elif family == "binomial":
        mu = np.clip(mu, eps, 1.0 - eps)
        dev = np.zeros_like(y, dtype=np.float64)
        mask1 = y > 0.0
        dev[mask1] += y[mask1] * np.log(y[mask1] / mu[mask1])
        mask2 = y < 1.0
        dev[mask2] += (1.0 - y[mask2]) * np.log((1.0 - y[mask2]) / (1.0 - mu[mask2]))
        dev *= 2.0
    elif family == "poisson":
        mu = np.clip(mu, eps, None)
        dev = np.zeros_like(y, dtype=np.float64)
        mask = y > 0.0
        dev[mask] = y[mask] * np.log(y[mask] / mu[mask])
        dev = 2.0 * (dev - (y - mu))
    elif family == "gamma":
        y = np.clip(y, eps, None)
        mu = np.clip(mu, eps, None)
        dev = 2.0 * ((y - mu) / mu - np.log(y / mu))
    elif family == "negbin":
        mu = np.clip(mu, eps, None)
        theta = float(getattr(model.family, "theta", 1.0))
        dev = np.zeros_like(y, dtype=np.float64)
        mask = y > 0.0
        dev[mask] = y[mask] * np.log(y[mask] / mu[mask])
        dev = 2.0 * (dev - (y + theta) * np.log((y + theta) / (mu + theta)))
    else:
        raise NotImplementedError(
            f"Deviance residuals are not implemented for family={model.family.name!r}."
        )

    return sign * np.sqrt(np.clip(w * dev, 0.0, None))


def residuals_gam(model, type: str = "deviance") -> np.ndarray:
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    type = str(type).lower()
    y = np.asarray(model.y_, dtype=np.float64).ravel()
    mu = np.asarray(model._fitted_mu, dtype=np.float64).ravel()
    eta = np.asarray(model._fitted_eta, dtype=np.float64).ravel()
    w = _prior_weights(model)

    if type == "response":
        return y - mu
    if type == "working":
        mu_eta = np.asarray(model.family.mu_eta(eta), dtype=np.float64)
        return (y - mu) / np.clip(mu_eta, 1e-300, None)
    if type == "deviance":
        return _deviance_residuals(model)
    if type in {"pearson", "scaled.pearson"}:
        var = np.asarray(model.family.variance(mu), dtype=np.float64)
        res = (y - mu) * np.sqrt(w) / np.sqrt(np.clip(var, 1e-300, None))
        if type == "scaled.pearson":
            res = res / np.sqrt(max(float(model.scale_), 1e-300))
        return res

    raise ValueError(
        "type must be one of {'deviance', 'pearson', 'scaled.pearson', 'working', 'response'}."
    )


__all__ = ["residuals_gam"]
