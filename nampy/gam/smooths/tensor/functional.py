"""Paired matrix arguments for tensor-product linear functionals."""

import numpy as np

from ..categorical.categorical_utils import (
    functional_object_columns,
    stable_unique_levels,
)
from ..smooth_base import (
    _resolve_feature,
    column_as_numeric_array,
    linear_functional_by_state,
    resolve_by_state,
)


def _locations(X, indices):
    return np.stack(functional_object_columns(X, indices), axis=-1)


def _setup_points(locations):
    from ..smooth_base import linear_functional_setup_points

    numeric = np.empty(locations.shape, dtype=float)
    factors = {}
    for index in range(locations.shape[-1]):
        values = locations[:, :, index]
        try:
            # Encoding is for packing only: numeric-looking character labels
            # must retain their identity when setup rows are decoded.
            if any(
                isinstance(value, (str, bytes, bool, np.bool_))
                for value in values.reshape(-1)
            ):
                raise ValueError("categorical string coordinate")
            numeric[:, :, index] = np.asarray(values, dtype=float)
        except (ValueError, TypeError):
            levels = stable_unique_levels(values.reshape(-1))
            lookup = {value: code for code, value in enumerate(levels)}
            numeric[:, :, index] = np.asarray(
                [lookup[value] for value in values.reshape(-1)]
            ).reshape(values.shape)
            factors[index] = levels
    setup, unpack = linear_functional_setup_points(numeric)
    if factors:
        decoded = setup.astype(object)
        for index, levels in factors.items():
            decoded[:, index] = [levels[int(code)] for code in setup[:, index]]
        setup = decoded
    return setup, unpack


def _pool(term, X, indices, by_index):
    raw = np.asarray(X, dtype=object)
    if not any(np.ndim(value) > 0 for index in indices for value in raw[:, index]):
        return X, None, None
    locations = _locations(X, indices)
    if locations.ndim == 2:
        return X, None, None
    n, q = locations.shape[:2]
    weights = (
        np.ones((n, q)) if by_index is None else column_as_numeric_array(X, by_index)
    )
    if weights.shape != (n, q):
        raise ValueError(
            "Linear-functional feature locations and by weights must have equal shape."
        )
    # Pool paired positions, never Cartesian products of quadrature columns.
    pooled = np.repeat(np.asarray(X, dtype=object), q, axis=0)
    for position, index in enumerate(indices):
        pooled[:, index] = locations[:, :, position].reshape(-1)
    if by_index is not None:
        pooled[:, by_index] = weights.reshape(-1)
    return pooled, (n, q), weights


def prepare_tensor_functional_fit(term, X, feature_names):
    features = [
        value
        for margin in term.feature
        for value in ([margin] if isinstance(margin, (str, int)) else margin)
    ]
    indices = [_resolve_feature(value, feature_names)[0] for value in features]
    state = resolve_by_state(term.by, X, feature_names)
    pooled, shape, weights = _pool(term, X, indices, state.feature_index)
    term._tensor_functional_unpack_index = None
    if shape is not None:
        locations = _locations(X, indices)
        setup_points, unpack = _setup_points(locations)
        representatives = np.full(len(setup_points), len(unpack), dtype=int)
        np.minimum.at(representatives, unpack, np.arange(len(unpack)))
        pooled = pooled[representatives].copy()
        pooled[:, indices] = setup_points
        term._tensor_functional_unpack_index = unpack
    term._tensor_functional_indices = indices
    term._tensor_functional_shape = shape
    term._tensor_functional_weights = weights
    term._tensor_functional_by_state = (
        None if shape is None else linear_functional_by_state(state, weights=weights)
    )
    if shape is not None and (
        term.smoothing_id is not None or term._linked_id_setup() is not None
    ):
        raise NotImplementedError(
            "Linked tensor matrix arguments require pooled linked setup support."
        )
    return pooled


def set_tensor_functional_by_state(term):
    state = getattr(term, "_tensor_functional_by_state", None)
    if state is not None:
        term._by_state = state


def finish_tensor_functional_fit(term):
    # Prediction's generic factor guard must defer only the coordinates whose
    # every margin implements random-effect zero rows for unseen levels.
    zero_features = set()
    strict_features = set()
    for margin in term._marginals:
        names = margin.resolved_feature_names_list()
        zero_owned = set(getattr(margin, "zero_unseen_factor_features", ()))
        zero_features.update(zero_owned)
        strict_features.update(set(names) - zero_owned)
    term.metadata = dict(term.metadata or {})
    term.metadata["zero_unseen_factor_features"] = sorted(
        zero_features - strict_features
    )
    shape = term._tensor_functional_shape
    if shape is not None:
        term._basis_train = np.einsum(
            "nq,nqk->nk",
            term._tensor_functional_weights,
            term._basis_train[term._tensor_functional_unpack_index].reshape(*shape, -1),
        )


def prepare_tensor_functional_prediction(term, X):
    pooled, shape, _ = _pool(
        term, X, term._tensor_functional_indices, term._by_state.feature_index
    )
    return pooled, shape


def finish_tensor_functional_prediction(basis, shape):
    return basis if shape is None else basis.reshape(*shape, -1).sum(axis=1)
