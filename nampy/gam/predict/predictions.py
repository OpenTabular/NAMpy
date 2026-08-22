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

import re
import warnings

import numpy as np

from ..fit.offsets import resolve_prediction_offset
from ..model_state import (
    _coef_column_offset,
    _coef_full,
    _require_fitted,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from ..term_labels import normalize_mgcv_term_label
from .general import predict_general_values
from .linear_predictor_matrix import _build_prediction_matrices
from .terms import (
    _group_standard_error_rows,
    _group_term_contribution,
    _prediction_term_groups,
)


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


def _term_filter_key(value):
    """Canonicalize inconsequential deparse spacing in mgcv term filters."""
    normalized = str(normalize_mgcv_term_label(value))
    return re.sub(r",\s*", ",", normalized)


def _prediction_group_selection(groups, *, terms, exclude):
    labels = tuple(str(group["label"]) for group in groups)
    label_keys = tuple(_term_filter_key(label) for label in labels)
    term_keys = None if terms is None else {_term_filter_key(term) for term in terms}
    exclude_keys = (
        None if exclude is None else {_term_filter_key(term) for term in exclude}
    )
    selected = np.ones(len(groups), dtype=bool)
    if term_keys is not None:
        selected &= np.asarray([key in term_keys for key in label_keys], dtype=bool)
    if exclude_keys is not None:
        selected &= np.asarray([key not in exclude_keys for key in label_keys], dtype=bool)
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
            full_indices = _term_full_coefficient_indices(model, tb)
            Xp_filtered[:, full_indices] = 0.0
    return Xp_filtered


def _filtered_term_output_indices(labels, *, terms, exclude):
    indices = list(range(len(labels)))
    label_keys = [_term_filter_key(label) for label in labels]
    if terms is not None:
        term_keys = [_term_filter_key(label) for label in terms]
        missing = [
            label
            for label, key in zip(terms, term_keys, strict=True)
            if key not in label_keys
        ]
        if missing:
            warnings.warn(
                "non-existent terms requested - ignoring",
                stacklevel=3,
            )
        else:
            indices = [label_keys.index(key) for key in term_keys]
    if exclude is not None:
        exclude_keys = [_term_filter_key(label) for label in exclude]
        missing = [
            label
            for label, key in zip(exclude, exclude_keys, strict=True)
            if key not in label_keys
        ]
        if missing:
            warnings.warn(
                "non-existent exclude terms requested - ignoring",
                stacklevel=3,
            )
        else:
            indices = [
                index for index in indices if label_keys[index] not in exclude_keys
            ]
    return np.asarray(indices, dtype=int)


def _prediction_parameterization_wider(tb) -> bool:
    metadata = dict(getattr(tb, "metadata", {}) or {})
    expose_raw = bool(metadata.get("expose_raw_prediction_basis", False))
    return expose_raw and bool(
        metadata.get("prediction_parameterization_wider", expose_raw)
    )


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
    for tb in group["blocks"]:
        full_indices = _term_full_coefficient_indices(model, tb)
        X1[:, full_indices] = Xp[:, full_indices]
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

    response_from_eta = getattr(model.family, "response_from_eta", None)
    if callable(response_from_eta):
        if return_se:
            response_se_from_eta = getattr(
                model.family, "response_se_from_eta", None
            )
            if not callable(response_se_from_eta):
                raise NotImplementedError(
                    f"Predictive standard errors are not implemented for "
                    f"family={model.family.name!r}."
                )
            V = model._select_cov(cov)
            var_eta = np.einsum("ij,jk,ik->i", Xp, V, Xp)
            se_eta = np.sqrt(np.maximum(var_eta, 0.0))
            return response_from_eta(eta), response_se_from_eta(eta, se_eta)
        return response_from_eta(eta)

    if not return_se:
        return mu

    V = model._select_cov(cov)
    var_eta = np.einsum("ij,jk,ik->i", Xp, V, Xp)
    se_eta = np.sqrt(np.maximum(var_eta, 0.0))
    se_mu = np.abs(model.family.mu_eta(eta)) * se_eta
    return mu, se_mu


__all__ = ["predict_values"]
