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

import numpy as np

from ..fit.covariance import select_prediction_covariance_matrix
from ..fit.offsets import resolve_prediction_offset
from ..model_state import (
    _coef_column_offset,
    _coef_full,
    _require_fitted,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from .general import predict_general_values
from .linear_predictor_matrix import _build_prediction_matrices
from .terms import (
    _filtered_term_output_indices,
    _group_standard_error_rows,
    _group_term_contribution,
    _prediction_group_selection,
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


def _prediction_parameterization_wider(tb) -> bool:
    metadata = dict(getattr(tb, "metadata", {}) or {})
    expose_raw = bool(metadata.get("expose_raw_prediction_basis", False))
    return expose_raw and bool(
        metadata.get("prediction_parameterization_wider", expose_raw)
    )


def prediction_guaranteed_skip_contract(model, *, terms=None, exclude=None):
    """Return constructor ids and feature columns safe to omit on newdata."""
    terms_filter = _coerce_prediction_term_filter(terms, name="terms")
    exclude_filter = _coerce_prediction_term_filter(exclude, name="exclude")
    groups = _prediction_term_groups(model)
    _labels, selected = _prediction_group_selection(
        groups, terms=terms_filter, exclude=exclude_filter
    )
    skipped_ids = set()
    skipped_features = set()
    active_features = set()
    for group, keep in zip(groups, selected, strict=True):
        for term in group["blocks"]:
            feature_names = {
                str(value)
                for value in getattr(term.feature_info, "feature_names", ())
            }
            by_name = getattr(term.by_variable_info, "name", None)
            if by_name is not None:
                feature_names.add(str(by_name))
            can_skip = not keep and group["term_type"] != "parametric"
            if can_skip:
                skipped_ids.add(str(term.term_id))
                skipped_features.update(feature_names)
            else:
                active_features.update(feature_names)
    return frozenset(skipped_ids), frozenset(skipped_features - active_features)


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

    X1 = np.broadcast_to(np.asarray(cmX, dtype=np.float64), Xp.shape).copy()
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
    unconditional=False,
    iterms_type=1,
    skip_term_ids=(),
    allow_missing_numeric=False,
):
    _require_fitted(model)
    terms_filter = _coerce_prediction_term_filter(terms, name="terms")
    exclude_filter = _coerce_prediction_term_filter(exclude, name="exclude")
    prediction_cov = None
    if unconditional or return_se:
        prediction_cov = select_prediction_covariance_matrix(
            model, cov=cov, unconditional=bool(unconditional)
        )
    if getattr(model.family, "family_class", "") == "general":
        return predict_general_values(
            model,
            X=X,
            return_se=return_se,
            cov=prediction_cov if prediction_cov is not None else cov,
            type=type,
            offset=offset,
            terms=terms_filter,
            exclude=exclude_filter,
            skip_term_ids=skip_term_ids,
        )

    type = str(type).lower()
    Z_new, Xp = _build_prediction_matrices(
        model,
        X_new=X,
        skip_term_ids=skip_term_ids,
        allow_missing_numeric=allow_missing_numeric,
    )
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
            _prediction_parameterization_wider(tb) for tb in _term_blocks_seq(model)
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

        V = prediction_cov
        cmX = None
        if type == "iterms":
            _Z_train, Xp_train = _build_prediction_matrices(model, X_new=None)
            cmX = np.mean(np.asarray(Xp_train, dtype=np.float64), axis=0)
            if int(iterms_type) == 2:
                cmX = np.asarray(cmX, dtype=np.float64).copy()
                for term in _term_blocks_seq(model):
                    if str(getattr(term, "term_type", "")) != "parametric":
                        cmX[_term_full_coefficient_indices(model, term)] = 0.0
        ses = []
        for group in groups:
            if type == "iterms":
                Xi, sl_full = _group_iterm_standard_error_rows(model, Xp, group, cmX)
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
        V = prediction_cov
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
            response_se_from_eta = getattr(model.family, "response_se_from_eta", None)
            if not callable(response_se_from_eta):
                raise NotImplementedError(
                    f"Predictive standard errors are not implemented for "
                    f"family={model.family.name!r}."
                )
            V = prediction_cov
            var_eta = np.einsum("ij,jk,ik->i", Xp, V, Xp)
            se_eta = np.sqrt(np.maximum(var_eta, 0.0))
            return response_from_eta(eta), response_se_from_eta(eta, se_eta)
        return response_from_eta(eta)

    if not return_se:
        return mu

    V = prediction_cov
    var_eta = np.einsum("ij,jk,ik->i", Xp, V, Xp)
    se_eta = np.sqrt(np.maximum(var_eta, 0.0))
    se_mu = np.abs(model.family.mu_eta(eta)) * se_eta
    return mu, se_mu


__all__ = ["predict_values", "prediction_guaranteed_skip_contract"]
