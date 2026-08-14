"""
GAM prediction entry point.

:func:`predict_values` is the main prediction function.  It supports multiple
output types controlled by the ``type`` argument:

- ``"response"`` (default): predicted mean ``mu = g^{-1}(eta)``.
- ``"link"``: linear predictor ``eta = X_new beta + offset``.
- ``"terms"``: per-term linear predictor contributions.
- ``"iterms"``: term contributions with mean-uncertainty standard errors.
- ``"lpmatrix"``: the raw linear predictor matrix.

The ``terms`` and ``exclude`` filters mirror ``mgcv::predict.gam`` by zeroing
unselected coefficient blocks before predictions and standard errors are formed.

Standard errors are optionally returned alongside predictions when
``return_se=True``, using either the Bayesian posterior covariance (default)
or the frequentist sandwich covariance.
"""

import warnings

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
from ..term_labels import normalize_mgcv_term_label
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
                "key": (
                    "term",
                    str(normalize_mgcv_term_label(getattr(tb, "label", ""))),
                ),
                "label": str(
                    normalize_mgcv_term_label(getattr(tb, "label", ""))
                ),
                "blocks": [tb],
                "term_type": term_type,
            }
        )
    return groups


def _coerce_prediction_term_filter(values, *, name):
    if values is None:
        return None
    if isinstance(values, str):
        return (values,)
    try:
        out = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a string or an iterable of strings.") from exc
    if not all(isinstance(value, str) for value in out):
        raise TypeError(f"{name} must contain only strings.")
    return out


def _prediction_group_selection(groups, *, terms, exclude):
    labels = tuple(str(group["label"]) for group in groups)
    selected = np.ones(len(groups), dtype=bool)
    if terms is not None:
        selected &= np.asarray([label in terms for label in labels], dtype=bool)
    if exclude is not None:
        selected &= np.asarray([label not in exclude for label in labels], dtype=bool)
    return labels, selected


def _filtered_prediction_matrix(model, Xp, groups, selected, *, terms, exclude):
    if terms is None and exclude is None:
        return Xp
    Xp_filtered = np.asarray(Xp, dtype=np.float64).copy()
    offset0 = _coef_column_offset(model)
    if offset0:
        keep_intercept = (terms is None or "(Intercept)" in terms) and (
            exclude is None or "(Intercept)" not in exclude
        )
        if not keep_intercept:
            Xp_filtered[:, :offset0] = 0.0
    for group, keep in zip(groups, selected, strict=True):
        if keep:
            continue
        for tb in group["blocks"]:
            sl = slice(offset0 + tb.coef_slice.start, offset0 + tb.coef_slice.stop)
            Xp_filtered[:, sl] = 0.0
    return Xp_filtered


def _filtered_term_output_indices(labels, *, terms, exclude):
    indices = list(range(len(labels)))
    if terms is not None:
        missing = [label for label in terms if label not in labels]
        if missing:
            warnings.warn(
                "non-existent terms requested - ignoring",
                stacklevel=3,
            )
        else:
            indices = [labels.index(label) for label in terms]
    if exclude is not None:
        missing = [label for label in exclude if label not in labels]
        if missing:
            warnings.warn(
                "non-existent exclude terms requested - ignoring",
                stacklevel=3,
            )
        else:
            indices = [index for index in indices if labels[index] not in exclude]
    return np.asarray(indices, dtype=int)


def _prediction_parameterization_wider(tb) -> bool:
    metadata = dict(getattr(tb, "metadata", {}) or {})
    expose_raw = bool(metadata.get("expose_raw_prediction_basis", False))
    return expose_raw and bool(
        metadata.get("prediction_parameterization_wider", expose_raw)
    )


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

    return 0.0


def _term_contribution(model, Z_new, tb):
    beta = np.asarray(_coef(model), dtype=np.float64)
    contrib = Z_new[:, tb.coef_slice] @ beta[tb.coef_slice]
    shift = _term_contribution_shift(model, tb)
    if shift != 0.0:
        contrib = np.asarray(contrib, dtype=np.float64) + shift
    return contrib


def _term_standard_error_rows(model, Xp, tb, *, type="terms"):
    offset0 = _coef_column_offset(model)
    sl_full = slice(offset0 + tb.coef_slice.start, offset0 + tb.coef_slice.stop)
    return np.asarray(Xp[:, sl_full], dtype=np.float64), sl_full


def _group_term_contribution(model, Z_new, group):
    contrib = np.zeros(Z_new.shape[0], dtype=np.float64)
    for tb in group["blocks"]:
        contrib = contrib + np.asarray(
            _term_contribution(model, Z_new, tb), dtype=np.float64
        )
    return np.asarray(contrib, dtype=np.float64)


def _group_standard_error_rows(model, Xp, group, *, type="terms"):
    if len(group["blocks"]) == 1:
        return _term_standard_error_rows(
            model,
            Xp,
            group["blocks"][0],
            type=type,
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


def _term_has_absorbed_constraint(tb) -> bool:
    metadata = dict(getattr(tb, "constructor_metadata", {}) or {})
    n_constraints = metadata.get("n_constraints_absorbed", None)
    if n_constraints is not None and int(n_constraints) > 0:
        return True
    if metadata.get("runtime_constraint_kind", None) is not None:
        return True

    # The ordinary univariate constructors store their already-centered basis
    # directly (for example `CubicSplines.basis`) rather than exposing a second
    # runtime transform. This is still mgcv's one absorbed smooth constraint.
    if str(getattr(tb, "term_type", "")) == "smooth":
        by_info = getattr(tb, "by_variable_info", None)
        return not bool(getattr(by_info, "is_active", False)) or bool(
            getattr(by_info, "is_constant", False)
        )
    return False


def _group_iterm_standard_error_rows(model, Xp, group, cmX):
    if group["term_type"] == "parametric" or not any(
        _term_has_absorbed_constraint(tb) for tb in group["blocks"]
    ):
        return _group_standard_error_rows(model, Xp, group, type="iterms")

    X1 = np.broadcast_to(
        np.asarray(cmX, dtype=np.float64), Xp.shape
    ).copy()
    offset0 = _coef_column_offset(model)
    for tb in group["blocks"]:
        sl = slice(offset0 + tb.coef_slice.start, offset0 + tb.coef_slice.stop)
        X1[:, sl] = Xp[:, sl]
    return X1, None


def predict_values(
    model,
    X=None,
    return_se=False,
    cov=None,
    type="response",
    offset=None,
    terms=None,
    exclude=None,
):
    _require_fitted(model)
    terms_filter = _coerce_prediction_term_filter(terms, name="terms")
    exclude_filter = _coerce_prediction_term_filter(exclude, name="exclude")
    if getattr(model.family, "family_class", "") == "general":
        return predict_general_values(
            model,
            X=X,
            return_se=return_se,
            cov=cov,
            type=type,
            offset=offset,
            terms=terms_filter,
            exclude=exclude_filter,
        )

    type = str(type).lower()
    Z_new, Xp = _build_prediction_matrices(model, X_new=X)
    groups = _prediction_term_groups(model)
    labels, selected = _prediction_group_selection(
        groups,
        terms=terms_filter,
        exclude=exclude_filter,
    )
    Xp = _filtered_prediction_matrix(
        model,
        Xp,
        groups,
        selected,
        terms=terms_filter,
        exclude=exclude_filter,
    )

    offset_vec = resolve_prediction_offset(model, X, offset)
    coef_full = np.asarray(_coef_full(model), dtype=np.float64)
    eta = Xp @ coef_full
    if offset_vec is not None:
        eta = eta + offset_vec

    mu = model.family.inverse_link(eta)

    if type == "lpmatrix":
        return Xp

    if type in {"terms", "iterms"}:
        if any(
            _prediction_parameterization_wider(tb)
            for tb in _term_blocks_seq(model)
        ):
            raise NotImplementedError(
                "type='terms' is not supported for models whose prediction "
                "parameterization is wider than the fitted coefficient space."
            )
        term_values = (
            np.column_stack(
                [
                    (
                        _group_term_contribution(model, Z_new, group)
                        if keep
                        else np.zeros(Z_new.shape[0], dtype=np.float64)
                    )
                    for group, keep in zip(groups, selected, strict=True)
                ]
            )
            if groups
            else np.empty((Z_new.shape[0], 0), dtype=np.float64)
        )
        output_indices = _filtered_term_output_indices(
            labels,
            terms=terms_filter,
            exclude=exclude_filter,
        )
        term_values = term_values[:, output_indices]
        if not return_se:
            return term_values

        V = model._select_cov(cov)
        cmX = None
        if type == "iterms":
            _Z_train, Xp_train = _build_prediction_matrices(model, X_new=None)
            cmX = np.mean(np.asarray(Xp_train, dtype=np.float64), axis=0)
        ses = []
        for group in groups:
            if type == "iterms":
                Xi, sl_full = _group_iterm_standard_error_rows(
                    model, Xp, group, cmX
                )
            else:
                Xi, sl_full = _group_standard_error_rows(
                    model,
                    Xp,
                    group,
                    type=type,
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
        se_values = (
            np.column_stack(ses)
            if ses
            else np.empty((Xp.shape[0], 0), dtype=np.float64)
        )
        return term_values, se_values[:, output_indices]

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
