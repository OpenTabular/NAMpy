"""Null deviance mirroring mgcv's post-fit assembly.

Upstream sources:

- GLM path: ``mgcv/R/gam.fit3.r:838-842`` (``wtdmu`` from the weighted mean of
  ``y`` when there is an intercept, else ``linkinv(offset)``).
- Offset correction: ``mgcv/R/mgcv.r:2072-2075`` (intercept-only GLM refit with
  the model offset).
- Extended families: ``mgcv/R/efam.r:98-117`` (``find.null.dev`` expanding
  interval 1-d search) via the family ``postproc`` (``efam.r:283-289`` for nb).
- General families: family-owned ``postproc`` expressions
  (``mgcv/R/gamlss.r:910-918`` for gaulss, ``:2737-2742`` for gammals),
  exposed as a per-family ``null_deviance`` hook.
"""

from __future__ import annotations

import numpy as np

from ..model_state import _fitted_eta, _fitted_mu, _require_fitted


def _prior_weights_vector(model, y: np.ndarray) -> np.ndarray:
    if model.prior_weights_ is None:
        return np.ones_like(y, dtype=np.float64)
    return np.asarray(model.prior_weights_, dtype=np.float64)


def _offset_vector(model, y: np.ndarray) -> np.ndarray:
    if model.offset_train_ is None:
        return np.zeros_like(y, dtype=np.float64)
    return np.asarray(model.offset_train_, dtype=np.float64).ravel()


def _offset_null_glm_deviance(family, y, weights, offset) -> float:
    """
    Mirror ``mgcv/R/mgcv.r:2072-2075``:
    ``glm(object$y ~ offset(G$offset), family, weights)$deviance``.
    """
    from ..fit.solvers.irls_core import irls_core

    X = np.ones((len(y), 1), dtype=np.float64)
    out = irls_core(
        X,
        y,
        family,
        np.zeros((1, 1), dtype=np.float64),
        offset=offset,
        weights=weights,
        fit_intercept=False,
        max_iter=200,
        tol=1e-10,
    )
    return float(out["deviance"])


def _find_null_dev(family, y, eta, offset, weights) -> float:
    """Port ``mgcv/R/efam.r:98-117`` ``find.null.dev`` exactly."""
    from scipy.optimize import minimize_scalar

    y = np.asarray(y, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64).ravel()
    offset = np.asarray(offset, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()

    def fnull(gamma: float) -> float:
        mu = np.asarray(family.inverse_link(gamma + offset), dtype=np.float64)
        return float(family.deviance(y, mu, weights=weights))

    mu0 = np.asarray(family.inverse_link(eta - offset), dtype=np.float64)
    mum = float(np.mean(mu0 * weights) / np.mean(weights))
    eta0 = float(np.asarray(family.link(np.array([mum])), dtype=np.float64)[0])
    deta = abs(eta0) * 0.1 + 1.0
    # R `optimize()` default tolerance is .Machine$double.eps^0.25.
    xatol = float(np.finfo(np.float64).eps ** 0.25)
    while True:
        lo, hi = eta0 - deta, eta0 + deta
        res = minimize_scalar(
            fnull, bounds=(lo, hi), method="bounded", options={"xatol": xatol}
        )
        if lo < float(res.x) < hi:
            return float(res.fun)
        deta *= 2.0


def compute_null_deviance(model) -> float:
    """Compute mgcv's ``object$null.deviance`` for a fitted model."""
    _require_fitted(model)
    family = model.family
    y = np.asarray(model.y_, dtype=np.float64)
    w = _prior_weights_vector(model, y)
    offset = _offset_vector(model, y)
    family_class = str(getattr(family, "family_class", "")).lower()

    if family_class == "general":
        hook = getattr(family, "null_deviance", None)
        if hook is None:
            raise NotImplementedError(
                f"Family {getattr(family, 'name', family)!r} does not provide "
                "a null_deviance postproc hook."
            )
        fitted = np.asarray(_fitted_mu(model), dtype=np.float64)
        return float(hook(y, fitted, w))

    if family_class == "extended":
        # mgcv/R/efam.r:283-289 (nb postproc): find.null.dev on the fitted
        # linear predictor.
        eta = np.asarray(_fitted_eta(model), dtype=np.float64)
        return _find_null_dev(family, y, eta, offset, w)

    # GLM path: mgcv/R/gam.fit3.r:838-842.  gam.outer hardcodes
    # ``intercept = TRUE`` into gam.fit3 (mgcv/R/mgcv.r:1667) regardless of
    # the formula, so the stored null deviance is always the weighted-mean
    # one — even for no-intercept models (the offset refit below is mgcv's
    # only correction, see the "Why not correct calc in gam.fit/3???" comment
    # at mgcv.r:2072).
    wtdmu = float(np.sum(w * y) / np.sum(w))
    mu0 = np.full_like(y, wtdmu, dtype=np.float64)
    nulldev = float(family.deviance(y, mu0, weights=w))

    # mgcv/R/mgcv.r:2072-2075: with an intercept and a nonzero offset the
    # null deviance comes from the offset-only GLM refit.
    if bool(model.fit_intercept) and np.any(offset != 0.0):
        nulldev = _offset_null_glm_deviance(family, y, w, offset)
    return nulldev


def null_deviance(model) -> float:
    """Cached accessor for :func:`compute_null_deviance`."""
    cached = getattr(model, "_null_deviance_", None)
    if cached is not None:
        return float(cached)
    value = compute_null_deviance(model)
    model._null_deviance_ = float(value)
    return float(value)


__all__ = ["compute_null_deviance", "null_deviance"]
