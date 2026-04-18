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
    _penalty_blocks_seq,
    _require_fitted,
    _term_blocks_seq,
)
from ..fit.offsets import resolve_prediction_offset
from .general import predict_general_values
from .linear_predictor_matrix import _build_prediction_matrices


def _tensor_anova_full_mode(tb):
    metadata = getattr(tb, "metadata", None) or {}
    term_spec = metadata.get("term_spec", {}) if isinstance(metadata, dict) else {}
    basis_options = (
        term_spec.get("basis_options", {}) if isinstance(term_spec, dict) else {}
    )
    if isinstance(basis_options, dict) and "full" in basis_options:
        return bool(basis_options["full"])
    return ";full)" in str(getattr(tb, "basis_name", ""))


def _fs_term_penalty_adjustment(model, tb):
    Z = _design_matrix(model)
    sp = getattr(model, "smoothing_params", None)
    if Z is None or sp is None:
        return None

    Z_term = np.asarray(Z, dtype=np.float64)[:, tb.coef_slice]
    if Z_term.size == 0:
        return None

    one = np.ones(Z_term.shape[0], dtype=np.float64)
    v_const, *_ = np.linalg.lstsq(Z_term, one, rcond=None)
    if np.max(np.abs(Z_term @ v_const - one)) > 1e-10:
        return None

    sp = np.asarray(sp, dtype=np.float64).ravel()
    numer_vec = np.zeros_like(v_const, dtype=np.float64)
    denom = 0.0
    for pb in _penalty_blocks_seq(model):
        if pb.coef_slice != tb.coef_slice:
            continue
        idx = int(pb.smoothing_index)
        if idx < 0 or idx >= sp.size:
            continue
        lam = float(sp[idx])
        if not np.isfinite(lam) or lam == 0.0:
            continue
        S = np.asarray(pb.matrix, dtype=np.float64)
        numer_vec += lam * (S @ v_const)
        denom += lam * float(v_const @ (S @ v_const))

    if not np.isfinite(denom) or abs(denom) <= np.finfo(np.float64).eps:
        return None
    return np.asarray(numer_vec / denom, dtype=np.float64)


def _term_contribution_shift(model, tb):
    if str(getattr(tb, "term_type", "")) == "factor_smooth_fs":
        beta_term = np.asarray(_coef(model), dtype=np.float64)[tb.coef_slice]
        adjust = _fs_term_penalty_adjustment(model, tb)
        if adjust is None:
            return 0.0
        return -float(adjust @ beta_term)

    if str(getattr(tb, "term_type", "")) != "tensor_anova":
        return 0.0

    # mgcv::predict.gam() splits term contributions after the prediction
    # matrix has already absorbed any fitted centering (`Xcentre`) and the
    # t2() smooth's null-block handling from
    # smooth.construct.t2.smooth.spec(). Our tensor-ANOVA port needs an
    # extra prediction-time mean correction for the default `full=FALSE`
    # decomposition, but not for `full=TRUE`.
    if _tensor_anova_full_mode(tb):
        return 0.0

    beta = np.asarray(_coef(model), dtype=np.float64)
    train_term = (
        np.asarray(_design_matrix(model), dtype=np.float64)[:, tb.coef_slice]
        @ beta[tb.coef_slice]
    )
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
