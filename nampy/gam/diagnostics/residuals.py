from __future__ import annotations

import numpy as np


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

    # Keep deviance algebra family-owned (mgcv-style `family$dev.resids` analogue)
    # to avoid drift between residual diagnostics and fitting code.
    dev = model.family.deviance(y, mu, w)
    return sign * np.sqrt(np.clip(dev, 0.0, None))


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
