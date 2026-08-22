"""mgcv ``logLik.gam`` / ``AIC.gam`` value algebra.

Mirrors mgcv/R/mgcv.r ``logLik.gam`` and the ``object$aic`` bookkeeping in
mgcv/R/gam.fit3.r. The :class:`~nampy.gam.model.api.GAM` facade delegates
``loglik``/``aic``/``bic`` here.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from ..fit.selection.criteria.gaussian_reml_algebra import (
    gaussian_reml_weighted_degrees_and_log_weight_term,
)
from ..fit.selection.criteria.pirls.family_gamma import _solve_gamma_profile_scale
from ..fit.selection.reparam import _static_penalty_null_dim
from ..model_state import (
    _coef_column_offset,
    _coef_full,
    _edf2,
    _edf_total,
    _fit_result,
    _fit_scale,
    _fit_state,
    _predictor_full_slices,
)


def loglik_effective_df(model) -> float:
    """
    mgcv-style effective df used by ``logLik.gam`` / AIC / BIC.

    Mirrors mgcv ``logLik.gam`` in mgcv/R/mgcv.r.
    """
    family_class = str(getattr(model.family, "family_class", "")).lower()
    sc_p = (
        0.0
        if family_class == "general"
        else (1.0 if getattr(model.family, "known_scale", None) is None else 0.0)
    )
    p = float(_edf_total(model)) + sc_p
    edf2 = _edf2(model)
    if edf2 is not None:
        p = float(np.sum(np.asarray(edf2, dtype=np.float64))) + sc_p
    np_max = float(len(np.asarray(_coef_full(model), dtype=np.float64))) + sc_p
    p = min(p, np_max)
    n_theta = getattr(model.family, "n_theta", None)
    if family_class == "extended" and n_theta is not None:
        p += float(n_theta)
    return p

def loglik_value_and_effective_df(model) -> tuple[float, float]:
    """
    mgcv ``logLik.gam`` uses two df notions:

    - value uses ``sum(edf) + scale.estimated``
    - attr(df) uses ``sum(edf2) + scale.estimated`` when available
    """
    family_class = str(getattr(model.family, "family_class", "")).lower()
    sc_p = (
        0.0
        if family_class == "general"
        else (1.0 if getattr(model.family, "known_scale", None) is None else 0.0)
    )
    p_val = float(_edf_total(model)) + sc_p
    p_df = p_val
    edf2 = _edf2(model)
    if edf2 is not None:
        p_df = float(np.sum(np.asarray(edf2, dtype=np.float64))) + sc_p
    np_max = float(len(np.asarray(_coef_full(model), dtype=np.float64))) + sc_p
    if p_df > np_max:
        p_df = np_max
    n_theta = getattr(model.family, "n_theta", None)
    if family_class == "extended" and n_theta is not None:
        p_df += float(n_theta)
    return p_val, p_df

def object_aic(model) -> float | None:
    """
    Final ``object$aic`` before ``logLik.gam`` post-processing.

    For Gaussian fits, mirror ``gam.fit3.r`` raw-family AIC plus the later
    ``mgcv.r`` `+ 2*sum(object$edf)` update exactly.
    """
    fit_result = _fit_result(model)
    if fit_result is None:
        return None

    family_name = str(getattr(model.family, "name", "")).lower()
    weights = (
        np.ones_like(np.asarray(model.y_, dtype=np.float64), dtype=np.float64)
        if model.prior_weights_ is None
        else np.asarray(model.prior_weights_, dtype=np.float64)
    )
    positive = weights > 0.0
    nobs = float(np.sum(weights[positive]))
    if nobs <= 0.0:
        return None

    if family_name == "gaussian":
        dev = float(getattr(fit_result, "deviance", np.nan))
        if not np.isfinite(dev) or dev <= 0.0:
            return None
        n_row = float(len(weights))
        n_true = getattr(model, "n_true_", None)
        n_eff = (
            None
            if n_true is None
            or not np.isfinite(float(n_true))
            or float(n_true) <= 0.0
            else float(n_true)
        )
        nobs, sum_log_scaled = gaussian_reml_weighted_degrees_and_log_weight_term(
            weights,
            n_row,
            mp=0.0,
            n_effective_total=n_eff,
        )
        if not np.isfinite(nobs) or nobs <= 0.0 or not np.isfinite(sum_log_scaled):
            return None
        raw_aic = (
            nobs * (np.log(dev / nobs * 2.0 * np.pi) + 1.0)
            + 2.0
            - float(sum_log_scaled)
        )
        return float(raw_aic + 2.0 * float(_edf_total(model)))

    y = np.asarray(model.y_, dtype=np.float64)
    mu = np.asarray(model.predict(X=None, type="response"), dtype=np.float64)
    raw_aic = None

    if family_name == "poisson":
        mu = np.clip(mu, np.finfo(np.float64).tiny, None)
        raw_aic = -2.0 * float(
            np.sum(weights * (y * np.log(mu) - mu - gammaln(y + 1.0)))
        )
    elif family_name == "binomial":
        # stats::binomial()$aic owns the m <- wt reinterpretation of
        # non-unit prior weights as binomial denominators.
        raw_aic = float(model.family.aic(y, mu, edf=0.0, weights=weights))
    elif family_name == "gamma":
        dispersion_scale: float | None = None
        optim_result = getattr(model, "_optim_result", None)
        joint_log_phi = (
            None
            if optim_result is None
            else getattr(optim_result, "joint_log_phi", None)
        )
        if (
            joint_log_phi is not None
            and np.isfinite(float(joint_log_phi))
            and str(getattr(model, "_optim_method", "")).lower() in {"reml", "ml"}
        ):
            dispersion_scale = float(np.exp(float(joint_log_phi)))
        if dispersion_scale is None:
            fit_method = str(getattr(model, "smoothing_method", "")).lower()
            if fit_method == "fixed" and getattr(model.family, "known_scale", None) is None:

                penalty = float(
                    getattr(fit_result, "penalty_quadratic", 0.0) or 0.0
                )
                mp = float(
                    _static_penalty_null_dim(model) + _coef_column_offset(model)
                )
                init_scale = _fit_scale(model)
                if init_scale is None or not np.isfinite(float(init_scale)):
                    init_scale = 1.0
                # mgcv/R/gam.fit3.r::gam.fit3 uses `reml.scale`
                # rather than the reported Pearson `sig2` in
                # stats::Gamma()$aic when fixed sp are fitted through
                # the REML path.
                dispersion_scale = float(
                    _solve_gamma_profile_scale(
                        model,
                        y,
                        float(getattr(fit_result, "deviance", np.nan)) + penalty,
                        mp=mp,
                        method="REML",
                        init_scale=float(init_scale),
                    )
                )
            if dispersion_scale is None:
                fit_scale_value = _fit_scale(model)
                dispersion_scale = (
                    None
                    if fit_scale_value is None
                    else float(fit_scale_value)
                )
        if (
            dispersion_scale is None
            or not np.isfinite(dispersion_scale)
            or dispersion_scale <= 0.0
        ):
            return None
        disp = dispersion_scale
        shape = 1.0 / disp
        mu = np.clip(mu, np.finfo(np.float64).tiny, None)
        y = np.clip(y, np.finfo(np.float64).tiny, None)
        raw_aic = (
            -2.0
            * float(
                np.sum(
                    weights
                    * (
                        (shape - 1.0) * np.log(y)
                        - y / (mu * disp)
                        - gammaln(shape)
                        - shape * np.log(mu * disp)
                    )
                )
            )
            + 2.0
        )

    if raw_aic is not None:
        return float(raw_aic + 2.0 * float(_edf_total(model)))

    return None

def loglik_gam(model) -> float:
    """
    Unpenalized fitted log-likelihood at penalized MLE.

    Mirrors mgcv ``logLik.gam`` value semantics.
    """
    if not model._fitted:
        raise RuntimeError("Model is not fitted.")

    aic_value = object_aic(model)
    if aic_value is not None:
        p_val, _p_df = loglik_value_and_effective_df(model)
        return float(p_val - aic_value / 2.0)

    if getattr(model.family, "family_class", "") == "general":
        fit_result = _fit_result(model)
        if fit_result is not None and fit_result.loglik is not None:
            # General-family fits already store the mgcv-shaped unpenalized
            # log-likelihood in fit space. Recomputing from exported
            # coefficients is wrong when public coefficients have been
            # mapped to a prediction parameterization.
            return float(fit_result.loglik)
        X = np.asarray(_fit_state(model).X, dtype=np.float64)
        jj = [
            np.arange(sl.start, sl.stop, dtype=int)
            for sl in _predictor_full_slices(model)
        ]
        weights = (
            np.ones_like(np.asarray(model.y_, dtype=np.float64), dtype=np.float64)
            if model.prior_weights_ is None
            else np.asarray(model.prior_weights_, dtype=np.float64)
        )
        ll = model.family.ll(
            np.asarray(model.y_, dtype=np.float64),
            X,
            jj,
            np.asarray(_coef_full(model), dtype=np.float64),
            weights,
            offset=model._general_family_offset_list(),
            deriv=0,
        )
        return float(ll["l"])

    y = np.asarray(model.y_, dtype=np.float64)
    mu = np.asarray(model.predict(X=None, type="response"), dtype=np.float64)
    glm_weights = (
        None
        if model.prior_weights_ is None
        else np.asarray(model.prior_weights_, dtype=np.float64)
    )
    dev = float(model.family.deviance(y, mu, weights=glm_weights))
    fit_scale = _fit_scale(model)
    if (
        fit_scale is not None
        and np.isfinite(float(fit_scale))
        and float(fit_scale) > 0.0
    ):
        scale = float(fit_scale)
    elif getattr(model.family, "known_scale", None) is None:
        scale_est = model.family.estimate_dispersion(
            y,
            mu,
            edf=float(_edf_total(model)),
            weights=glm_weights,
        )
        scale = max(float(scale_est), float(np.finfo(np.float64).tiny))
    else:
        scale = float(model.family.known_scale)
    sat = float(
        model.family.saturated_loglik(
            y,
            weights=glm_weights,
            n=len(y),
            scale=scale,
        )
    )
    return float(sat - dev / (2.0 * scale))


__all__ = [
    "loglik_effective_df",
    "loglik_value_and_effective_df",
    "loglik_gam",
    "object_aic",
]
