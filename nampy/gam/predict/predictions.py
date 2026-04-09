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

from .linear_predictor_matrix import _build_prediction_matrices


def _term_contribution_shift(model, tb):
    if str(getattr(tb, "term_type", "")) != "tensor_anova":
        return 0.0

    # mgcv's t2 prediction decomposition uses a prediction-time centering
    # convention for predict(type="terms") that differs from the fitted
    # coefficient split. Preserve link/lpmatrix parity and adjust only the
    # reported term contribution by the fitted training-sample mean.
    train_term = model.Z[:, tb.coef_slice] @ model.coef_[tb.coef_slice]
    return -float(np.mean(np.asarray(train_term, dtype=np.float64)))


def _term_contribution(model, Z_new, tb):
    contrib = Z_new[:, tb.coef_slice] @ model.coef_[tb.coef_slice]
    shift = _term_contribution_shift(model, tb)
    if shift != 0.0:
        contrib = np.asarray(contrib, dtype=np.float64) + shift
    return contrib


def predict_values(
    model, X=None, return_se=False, cov=None, type="response", offset=None
):
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")

    type = str(type).lower()
    Z_new, Xp = _build_prediction_matrices(model, X_new=X)

    offset_vec = model._prediction_offset(X, offset)
    eta = Xp @ model.coef_full_
    if offset_vec is not None:
        eta = eta + offset_vec

    mu = model.family.inverse_link(eta)

    if type == "lpmatrix":
        return Xp

    if type == "terms":
        terms = np.column_stack(
            [_term_contribution(model, Z_new, tb) for tb in model.term_blocks_]
        )
        if not return_se:
            return terms

        V = model._select_cov(cov)
        offset0 = 1 if model.fit_intercept else 0
        ses = []
        for tb in model.term_blocks_:
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
