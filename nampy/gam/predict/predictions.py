"""
GAM prediction entry point.

:func:`predict_values` is the main prediction function.  It supports multiple
output types controlled by the ``type`` argument:

- ``"response"`` (default): predicted mean ``mu = g^{-1}(eta)``.
- ``"link"``: linear predictor ``eta = X_new beta + offset``.
- ``"terms"``: per-term linear predictor contributions.
- ``"iterms"``: ``"terms"`` with mgcv-style mean-carrying smooth SEs.
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
    _fit_intercept,
    _penalty_blocks_seq,
    _require_fitted,
    _term_blocks_seq,
)
from ..fit.offsets import resolve_prediction_offset
from .general import predict_general_values
from .linear_predictor_matrix import _build_prediction_matrices


def _parametric_formula_term(tb) -> str | None:
    metadata = dict(getattr(tb, "metadata", {}) or {})
    formula_term = metadata.get("formula_term", None)
    return None if formula_term is None else str(formula_term)


def _prediction_term_groups(model):
    groups = []
    for tb in _term_blocks_seq(model):
        term_type = str(getattr(tb, "term_type", ""))
        if term_type == "parametric":
            formula_term = _parametric_formula_term(tb)
            group_key = ("parametric", formula_term or str(getattr(tb, "label", "")))
            if groups and groups[-1]["key"] == group_key:
                groups[-1]["blocks"].append(tb)
                continue
            groups.append(
                {
                    "key": group_key,
                    "label": formula_term or str(getattr(tb, "label", "")),
                    "blocks": [tb],
                    "term_type": term_type,
                }
            )
            continue

        groups.append(
            {
                "key": ("term", str(getattr(tb, "label", ""))),
                "label": str(getattr(tb, "label", "")),
                "blocks": [tb],
                "term_type": term_type,
            }
        )
    return groups


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


def _term_has_iterms_mean_uncertainty(model, tb) -> bool:
    if str(getattr(tb, "term_type", "")) == "parametric":
        return False

    metadata = dict(getattr(tb, "metadata", {}) or {})
    if bool(metadata.get("absorbed_sum_to_zero_constraint", False)):
        return True

    constructor_metadata = dict(getattr(tb, "constructor_metadata", {}) or {})
    n_cons = constructor_metadata.get("n_constraints_absorbed", None)
    if n_cons is not None and int(n_cons) > 0:
        return True

    runtime_constraint_kind = constructor_metadata.get("runtime_constraint_kind", None)
    if runtime_constraint_kind is not None:
        return True

    runtime_skip_centering = bool(
        constructor_metadata.get("runtime_skip_centering", False)
    )
    runtime_by_name = constructor_metadata.get("runtime_by_name", None)
    runtime_by_is_constant = constructor_metadata.get("runtime_by_is_constant", None)
    if (
        not _fit_intercept(model)
        or runtime_skip_centering
        or (
            runtime_by_name is not None
            and runtime_by_is_constant is not None
            and not bool(runtime_by_is_constant)
        )
        or str(getattr(tb, "term_type", "")) in {"tensor_interaction", "tensor_anova"}
    ):
        return False

    B = np.asarray(getattr(tb, "basis_train", None), dtype=np.float64)
    if B.size == 0:
        return False
    return bool(np.max(np.abs(np.mean(B, axis=0))) <= 1e-10)


def _term_iterms_mean_scale(tb) -> float | None:
    for container in (
        getattr(tb, "metadata", None),
        getattr(tb, "constructor_metadata", None),
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("meanL1", None)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value) and value != 0.0:
            return value
    return None


def _prediction_cmX(model) -> np.ndarray:
    _, Xp_train = _build_prediction_matrices(model, X_new=None)
    return np.asarray(np.mean(Xp_train, axis=0), dtype=np.float64)


def _parametric_full_coef_indices(model) -> np.ndarray:
    offset0 = _coef_column_offset(model)
    indices = []
    if _fit_intercept(model):
        indices.append(0)
    for tb in _term_blocks_seq(model):
        if str(getattr(tb, "term_type", "")) != "parametric":
            continue
        indices.extend(
            range(offset0 + tb.coef_slice.start, offset0 + tb.coef_slice.stop)
        )
    return np.asarray(indices, dtype=int)


def _iterms_mean_row(model, *, iterms_type=None) -> np.ndarray:
    cmX = _prediction_cmX(model)
    if iterms_type == 2:
        keep = _parametric_full_coef_indices(model)
        restricted = np.zeros_like(cmX, dtype=np.float64)
        if keep.size > 0:
            restricted[keep] = cmX[keep]
        return restricted
    return cmX


def _term_standard_error_rows(model, Xp, tb, *, type="terms", iterms_mean_row=None):
    offset0 = _coef_column_offset(model)
    sl_full = slice(offset0 + tb.coef_slice.start, offset0 + tb.coef_slice.stop)
    if type != "iterms" or not _term_has_iterms_mean_uncertainty(model, tb):
        return np.asarray(Xp[:, sl_full], dtype=np.float64), sl_full

    X1 = np.tile(np.asarray(iterms_mean_row, dtype=np.float64), (Xp.shape[0], 1))
    meanL1 = _term_iterms_mean_scale(tb)
    if meanL1 is not None:
        X1 = X1 / meanL1
    X1[:, sl_full] = Xp[:, sl_full]
    return np.asarray(X1, dtype=np.float64), None


def _group_term_contribution(model, Z_new, group):
    contrib = np.zeros(Z_new.shape[0], dtype=np.float64)
    for tb in group["blocks"]:
        contrib = contrib + np.asarray(
            _term_contribution(model, Z_new, tb), dtype=np.float64
        )
    return np.asarray(contrib, dtype=np.float64)


def _group_standard_error_rows(model, Xp, group, *, type="terms", iterms_mean_row=None):
    if len(group["blocks"]) == 1:
        return _term_standard_error_rows(
            model,
            Xp,
            group["blocks"][0],
            type=type,
            iterms_mean_row=iterms_mean_row,
        )

    if group["term_type"] != "parametric":
        raise NotImplementedError("Only parametric prediction groups may span blocks.")

    offset0 = _coef_column_offset(model)
    cols = []
    for tb in group["blocks"]:
        sl = slice(offset0 + tb.coef_slice.start, offset0 + tb.coef_slice.stop)
        cols.extend(range(sl.start, sl.stop))
    if len(cols) == 0:
        return np.empty((Xp.shape[0], 0), dtype=np.float64), slice(0, 0)
    Xi = np.asarray(Xp[:, np.asarray(cols, dtype=int)], dtype=np.float64)
    return Xi, np.asarray(cols, dtype=int)


def predict_values(
    model,
    X=None,
    return_se=False,
    cov=None,
    type="response",
    offset=None,
    iterms_type=None,
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
            iterms_type=iterms_type,
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

    if type in {"terms", "iterms"}:
        groups = _prediction_term_groups(model)
        terms = np.column_stack(
            [_group_term_contribution(model, Z_new, group) for group in groups]
        )
        if not return_se:
            return terms

        V = model._select_cov(cov)
        iterms_mean_row = (
            None
            if type != "iterms"
            else _iterms_mean_row(model, iterms_type=iterms_type)
        )
        ses = []
        for group in groups:
            Xi, sl_full = _group_standard_error_rows(
                model,
                Xp,
                group,
                type=type,
                iterms_mean_row=iterms_mean_row,
            )
            if sl_full is None:
                var = np.einsum("ij,jk,ik->i", Xi, V, Xi)
            elif isinstance(sl_full, np.ndarray):
                Vi = V[np.ix_(sl_full, sl_full)]
                var = np.einsum("ij,jk,ik->i", Xi, Vi, Xi)
            else:
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
            "type must be one of {'response', 'link', 'terms', 'iterms', 'lpmatrix'}"
        )

    if not return_se:
        return mu

    V = model._select_cov(cov)
    var_eta = np.einsum("ij,jk,ik->i", Xp, V, Xp)
    se_eta = np.sqrt(np.maximum(var_eta, 0.0))
    se_mu = np.abs(model.family.mu_eta(eta)) * se_eta
    return mu, se_mu


__all__ = ["predict_values"]
