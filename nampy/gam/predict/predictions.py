"""
GAM prediction entry point.

:func:`predict_values` is the main prediction function.  It supports multiple
output types controlled by the ``type`` argument:

- ``"response"`` (default): predicted mean ``mu = g^{-1}(eta)``.
- ``"link"``: linear predictor ``eta = X_new beta + offset``.
- ``"terms"``: per-term linear predictor contributions.
- ``"lpmatrix"``: the raw linear predictor matrix.

Standard errors are optionally returned alongside predictions when
``return_se=True``, using either the Bayesian posterior covariance (default)
or the frequentist sandwich covariance.
"""

import numpy as np

from .._model_state import (
    _coef,
    _coef_column_offset,
    _coef_full,
    _design_matrix,
    _require_fitted,
    _term_blocks_seq,
)
from ..fit.offsets import resolve_prediction_offset
from .general import predict_general_values
from .linear_predictor_matrix import _build_prediction_matrices


def _term_contribution_shift(model, tb):
    if str(getattr(tb, "term_type", "")) != "tensor_anova":
        return 0.0

    # mgcv's t2 prediction decomposition uses a prediction-time centering
    # convention for predict(type="terms") that differs from the fitted
    # coefficient split. Preserve link/lpmatrix parity and adjust only the
    # reported term contribution by the fitted training-sample mean.
    beta = np.asarray(_coef(model), dtype=np.float64)
    train_term = np.asarray(_design_matrix(model), dtype=np.float64)[:, tb.coef_slice] @ beta[tb.coef_slice]
    return -float(np.mean(np.asarray(train_term, dtype=np.float64)))


def _term_contribution(model, Z_new, tb):
    beta = np.asarray(_coef(model), dtype=np.float64)
    contrib = Z_new[:, tb.coef_slice] @ beta[tb.coef_slice]
    shift = _term_contribution_shift(model, tb)
    if shift != 0.0:
        contrib = np.asarray(contrib, dtype=np.float64) + shift
    return contrib


def predict_values(
    model, X=None, return_se=False, cov=None, type="response", offset=None
):
    _require_fitted(model)
    if getattr(model.family, "family_class", "") == "general":
        return predict_general_values(
            model,
            X=X,
            return_se=return_se,
            cov=cov,
            type=type,
            offset=offset,
        )

    type = str(type).lower()
    Z_new, Xp = _build_prediction_matrices(model, X_new=X)

    offset_vec = resolve_prediction_offset(model, X, offset)
    coef_full = np.asarray(_coef_full(model), dtype=np.float64)
    eta = Xp @ coef_full
    if offset_vec is not None:
        eta = eta + offset_vec

    mu = model.family.inverse_link(eta)

    if type == "lpmatrix":
        return Xp

    if type == "terms":
        terms = np.column_stack(
            [_term_contribution(model, Z_new, tb) for tb in _term_blocks_seq(model)]
        )
        if not return_se:
            return terms

        V = model._select_cov(cov)
        offset0 = _coef_column_offset(model)
        ses = []
        for tb in _term_blocks_seq(model):
            sl_full = slice(
                offset0 + tb.coef_slice.start,
                offset0 + tb.coef_slice.stop,
            )
            Xi = Xp[:, sl_full]
            Vi = V[sl_full, sl_full]
            var = np.einsum("ij,jk,ik->i", Xi, Vi, Xi)
            ses.append(np.sqrt(np.maximum(var, 0.0)))
        return terms, np.column_stack(ses)

    if type == "link":
        if not return_se:
            return eta
        V = model._select_cov(cov)
        var_eta = np.einsum("ij,jk,ik->i", Xp, V, Xp)
        se_eta = np.sqrt(np.maximum(var_eta, 0.0))
        return eta, se_eta

    if type != "response":
        raise ValueError(
            "type must be one of {'response', 'link', 'terms', 'lpmatrix'}"
        )

    if not return_se:
        return mu

    V = model._select_cov(cov)
    var_eta = np.einsum("ij,jk,ik->i", Xp, V, Xp)
    se_eta = np.sqrt(np.maximum(var_eta, 0.0))
    se_mu = np.abs(model.family.mu_eta(eta)) * se_eta
    return mu, se_mu


__all__ = ["predict_values"]
