"""Markov-random-field smooths matching mgcv ``bs='mrf'``."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...constraints.absorption import (
    fit_single_penalty_with_constraint_policy,
    fit_single_penalty_with_setup_basis,
)
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ...splines.basis.natparam import nat_param_type0
from ..registry import register_smooth
from ..smooth_base import BaseSmoothTerm, _resolve_feature, column_as_object
from .categorical_utils import factor_levels_from_metadata


def _canonical_label(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return format(number, ".15g")
    return str(value)


def _canonical_labels(values, *, boolean_coded=False) -> np.ndarray:
    raw = np.asarray(values, dtype=object).ravel()
    labels = []
    for value in raw:
        if (
            boolean_coded
            and isinstance(value, (int, np.integer, float, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and float(value) in {0.0, 1.0}
        ):
            labels.append("TRUE" if float(value) == 1.0 else "FALSE")
        else:
            labels.append(_canonical_label(value))
    return np.asarray(labels, dtype=object)


def _sorted_factor_levels(values) -> list[str]:
    raw = np.asarray(values, dtype=object).ravel()
    raw = np.asarray([value for value in raw if not pd.isna(value)], dtype=object)
    try:
        ordered = np.unique(raw).tolist()
    except (TypeError, ValueError):
        ordered = sorted(set(_canonical_labels(raw)))
    return [str(_canonical_label(value)) for value in ordered]


def _knot_levels(knots) -> list[str]:
    if isinstance(knots, pd.Categorical):
        return [str(_canonical_label(value)) for value in knots.categories]
    if isinstance(knots, pd.Series) and isinstance(knots.dtype, pd.CategoricalDtype):
        return [str(_canonical_label(value)) for value in knots.cat.categories]
    return _sorted_factor_levels(np.asarray(knots, dtype=object))


def _indicator_matrix(values, levels) -> np.ndarray:
    labels = _canonical_labels(values)
    index = {str(level): j for j, level in enumerate(levels)}
    out = np.zeros((labels.size, len(levels)), dtype=np.float64)
    for row, label in enumerate(labels):
        if label is not None and label in index:
            out[row, index[label]] = 1.0
    return out


def _polygons_to_neighbors(polygons) -> dict[str, list[str]]:
    if not isinstance(polygons, dict):
        raise TypeError(
            "MRF xt['polys'] must be a mapping from area names to polygons."
        )
    names = [str(_canonical_label(name)) for name in polygons]
    vertices: dict[str, set[tuple[float, float]]] = {}
    for raw_name, polygon in polygons.items():
        array = np.asarray(polygon, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("Each MRF polygon must be a two-column matrix.")
        finite = array[np.isfinite(array).all(axis=1)]
        vertices[str(_canonical_label(raw_name))] = {
            tuple(map(float, row)) for row in finite
        }
    return {
        name: [
            other
            for other in names
            if other != name and vertices[name] & vertices[other]
        ]
        for name in names
    }


def _neighbor_penalty(neighbors, levels) -> np.ndarray:
    if not isinstance(neighbors, dict) or len(neighbors) == 0:
        raise TypeError("MRF xt['nb'] must be a non-empty named mapping.")
    area_names = [str(_canonical_label(name)) for name in neighbors]
    if sorted(area_names) != sorted(levels):
        raise ValueError(
            "mismatch between nb/polys supplied area names and data area names"
        )

    entry_modes = set()
    for raw_adjacent in neighbors.values():
        adjacent = np.asarray(raw_adjacent, dtype=object).ravel().tolist()
        if not adjacent:
            continue
        numeric_flags = [
            isinstance(value, (int, np.integer, float, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and float(value).is_integer()
            for value in adjacent
        ]
        if any(numeric_flags) and not all(numeric_flags):
            raise TypeError(
                "MRF neighbour entries must use either all area names or all "
                "one-based numeric indices."
            )
        entry_modes.add("numeric" if all(numeric_flags) else "named")
    if len(entry_modes) > 1:
        raise TypeError(
            "MRF neighbour lists must use one uniform representation: either "
            "area names or one-based numeric indices."
        )
    neighbor_mode = next(iter(entry_modes), "named")

    S = np.zeros((len(levels), len(levels)), dtype=np.float64)
    level_index = {name: i for i, name in enumerate(levels)}
    for raw_name, raw_adjacent in neighbors.items():
        name = str(_canonical_label(raw_name))
        adjacent = np.asarray(raw_adjacent, dtype=object).ravel().tolist()
        if neighbor_mode == "numeric":
            adjacent_names = []
            for value in adjacent:
                position = int(value) - 1
                if position < 0 or position >= len(area_names):
                    raise IndexError("MRF neighbour index is outside the named list.")
                adjacent_names.append(area_names[position])
        else:
            requested = {str(_canonical_label(value)) for value in adjacent}
            # mgcv maps named neighbours through `which(nb.names %in% entry)`,
            # so names are set-like: ordering, duplicates, and unknown names
            # disappear before the degree is computed.
            adjacent_names = [value for value in area_names if value in requested]

        row = level_index[name]
        S[row, row] = float(len(adjacent_names))
        for adjacent_name in adjacent_names:
            column = level_index[adjacent_name]
            if column != row:
                S[row, column] = -1.0
    if np.any(S != S.T):
        raise ValueError("Something wrong with auto- penalty construction")
    return S


def _supplied_penalty(penalty, levels) -> np.ndarray:
    if isinstance(penalty, pd.DataFrame):
        column_names = [str(_canonical_label(value)) for value in penalty.columns]
        if sorted(column_names) != sorted(levels):
            raise ValueError("penalty column names don't match supplied area names!")
        lookup_rows = {str(_canonical_label(value)): value for value in penalty.index}
        lookup_cols = {str(_canonical_label(value)): value for value in penalty.columns}
        penalty = penalty.loc[
            [lookup_rows[level] for level in levels],
            [lookup_cols[level] for level in levels],
        ]
    matrix = np.asarray(penalty, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("supplied penalty not square!")
    if matrix.shape != (len(levels), len(levels)):
        raise ValueError("supplied penalty wrong dimension!")
    return matrix


@dataclass
class MarkovRandomFieldSetup:
    levels: tuple[str, ...]
    basis_train: np.ndarray
    penalty: np.ndarray
    raw_penalty: np.ndarray
    P: np.ndarray | None
    rank: int
    null_space_dim: int
    bs_dim: int
    used_low_rank: bool
    plot_me: bool
    boolean_coded: bool


def build_markov_random_field_setup(
    values, *, k=-1, xt=None, knots=None, factor_levels=None
) -> MarkovRandomFieldSetup:
    """Port ``smooth.construct.mrf.smooth.spec`` before smoothCon scaling."""
    raw_values = np.asarray(values, dtype=object).ravel()
    declared = (
        []
        if factor_levels is None
        else np.asarray(factor_levels, dtype=object).ravel().tolist()
    )
    boolean_coded = bool(
        declared and all(isinstance(value, (bool, np.bool_)) for value in declared)
    )
    labels = _canonical_labels(raw_values, boolean_coded=boolean_coded)
    if any(value is None for value in labels):
        raise ValueError("MRF smooths do not allow missing regions in fitting.")

    if factor_levels is None:
        data_levels = _sorted_factor_levels(labels)
    else:
        data_levels = sorted(str(_canonical_label(value)) for value in factor_levels)
    levels = data_levels if knots is None else _knot_levels(knots)
    if any(value not in levels for value in data_levels):
        raise ValueError(
            "data contain regions that are not contained in the knot specification"
        )
    observed_levels = {value for value in labels if value is not None}
    if any(value not in levels for value in observed_levels):
        raise ValueError(
            "data contain regions that are not contained in the knot specification"
        )

    bs_dim = len(levels) if int(k) < 0 else int(k)
    if bs_dim > len(levels):
        raise ValueError("MRF basis dimension set too high")
    if bs_dim <= 2 and bs_dim < len(levels):
        raise ValueError(
            "A reduced-rank MRF basis with k<=2 is malformed in mgcv 1.9-4; use k>=3."
        )
    if xt is None:
        raise ValueError(
            "penalty matrix, boundary polygons and/or neighbours list must be supplied in xt"
        )
    if not isinstance(xt, dict):
        raise TypeError("MRF xt must be a dictionary.")

    plot_me = xt.get("polys") is not None
    if xt.get("penalty") is not None:
        raw_penalty = _supplied_penalty(xt["penalty"], levels)
    else:
        neighbors = xt.get("nb")
        if neighbors is None:
            polygons = xt.get("polys")
            if polygons is None:
                raise ValueError("no spatial information provided!")
            neighbors = _polygons_to_neighbors(polygons)
        raw_penalty = _neighbor_penalty(neighbors, levels)

    X = _indicator_matrix(labels, levels)
    P = None
    if bs_dim < len(levels):
        missing = np.flatnonzero(np.sum(X, axis=0) == 0.0)
        X_complete = X
        if missing.size:
            dummy = np.zeros((missing.size, X.shape[1]), dtype=np.float64)
            dummy[np.arange(missing.size), missing] = 1.0
            X_complete = np.vstack([dummy, X])
        natural = nat_param_type0(X_complete, raw_penalty)
        # Retain the final (least penalized) natural-parameter columns.
        keep = np.arange(len(levels) - bs_dim, len(levels), dtype=int)
        X_natural = np.asarray(natural["X"], dtype=np.float64)
        X = X_natural[missing.size :, keep] if missing.size else X_natural[:, keep]
        P = np.asarray(natural["P"], dtype=np.float64)[:, keep]
        diagonal = np.asarray(
            [natural["D"][i] if i < natural["rank"] else 0.0 for i in keep],
            dtype=np.float64,
        )
        penalty = np.diag(diagonal)
        rank = int(np.sum(keep < natural["rank"]))
    else:
        penalty = np.asarray(raw_penalty, dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(penalty)
        largest = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
        rank = int(np.sum(eigenvalues > np.finfo(float).eps ** 0.8 * largest))

    return MarkovRandomFieldSetup(
        levels=tuple(levels),
        basis_train=np.asarray(X, dtype=np.float64),
        penalty=np.asarray(penalty, dtype=np.float64),
        raw_penalty=np.asarray(raw_penalty, dtype=np.float64),
        P=P,
        rank=rank,
        null_space_dim=int(bs_dim - rank),
        bs_dim=int(bs_dim),
        used_low_rank=bool(P is not None),
        plot_me=bool(plot_me),
        boolean_coded=boolean_coded,
    )


def predict_markov_random_field(values, setup: MarkovRandomFieldSetup) -> np.ndarray:
    labels = _canonical_labels(values, boolean_coded=setup.boolean_coded)
    unknown = sorted(
        {
            str(value)
            for value in labels
            if value is not None and value not in setup.levels
        }
    )
    if unknown:
        raise ValueError(
            f"MRF prediction data contain unknown regions {unknown}; "
            f"known regions are {list(setup.levels)}."
        )
    basis = _indicator_matrix(labels, setup.levels)
    if setup.P is not None:
        basis = basis @ setup.P
    return np.asarray(basis, dtype=np.float64)


@register_smooth("mrf")
class MarkovRandomFieldTerm(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "mrf"

    def __init__(
        self,
        feature,
        k=-1,
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        constraint_mode="auto",
        knots=None,
        xt=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) != 1:
            raise ValueError("MRF smooths require exactly one region covariate.")
        super().__init__(
            feature=features,
            label=label or f"s({features[0]})",
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.k = int(k)
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.knots = knots
        self.xt = xt
        self._feature_index = None
        self._feature_name = None
        self._factor_feature_indices = None
        self._factor_feature_names = None
        self._factor_levels = None
        self._setup = None
        self._basis_train = None
        self._penalties = None

    @property
    def expected_linked_penalty_count(self):
        return None if self.select else 1

    def fit(self, X, feature_names):
        if self.fixed and self.sp is not None:
            sp_values = np.asarray(self.sp, dtype=np.float64).ravel()
            if np.any(sp_values >= 0.0):
                raise ValueError(
                    "incorrect number of smoothing parameters supplied for a smooth term"
                )
        index, name = _resolve_feature(self.feature[0], feature_names)
        values = column_as_object(X, index)
        declared_levels = factor_levels_from_metadata(self.metadata, name)
        if declared_levels is None:
            try:
                np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                pass
            else:
                warnings.warn(
                    "argument of mrf should be a factor variable", stacklevel=2
                )

        self._set_by_state(X, feature_names)
        self._feature_index = index
        self._feature_name = name
        self._factor_feature_indices = [index]
        self._factor_feature_names = [name]
        self._set_resolved_features([name])

        shared_X = self._linked_id_setup_matrix(feature_names)
        setup_values = values if shared_X is None else column_as_object(shared_X, index)
        self._setup = build_markov_random_field_setup(
            setup_values,
            k=self.k,
            xt=self.xt,
            knots=self.knots,
            factor_levels=(declared_levels if shared_X is None else None),
        )
        self._factor_levels = [list(self._setup.levels)]
        if declared_levels is not None and self.knots is not None:
            meta = dict(self.metadata or {})
            factor_meta = dict(meta.get("factor_levels_by_feature", {}))
            current = dict(factor_meta.get(str(name), {}))
            allowed = list(current.get("levels", declared_levels))
            for value in self._setup.levels:
                if value not in allowed:
                    allowed.append(value)
            current["levels"] = allowed
            factor_meta[str(name)] = current
            meta["factor_levels_by_feature"] = factor_meta
            self.metadata = meta
        setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
        base = (
            setup_base
            if shared_X is None
            else predict_markov_random_field(values, self._setup)
        )
        raw_penalty = np.asarray(self._setup.penalty, dtype=np.float64)
        penalty = scale_penalty(setup_base, raw_penalty)
        self._set_penalty_rescale_factors(
            [penalty_rescale_factor(setup_base, raw_penalty)]
        )

        auto_constrain = bool(self._by_state.is_constant)
        if shared_X is None:
            result = fit_single_penalty_with_constraint_policy(
                base,
                penalty,
                self._by_state,
                constraint_mode=self.constraint_mode,
                fixed=self.fixed,
                auto_constrain_when=auto_constrain,
            )
        else:
            result = fit_single_penalty_with_setup_basis(
                base,
                setup_base,
                penalty,
                self._by_state,
                constraint_mode=self.constraint_mode,
                fixed=self.fixed,
                auto_constrain_when=auto_constrain,
            )
        self._basis_train = result.basis_train
        self._penalties = result.penalties
        self._record_constraint_result(
            result.constraint_kind,
            result.constraint_transform,
            absorbed_by="runtime" if result.constraint_transform is not None else None,
        )
        return self

    def get_penalty_definitions(self):
        self._require_fitted()
        if not self.penalties:
            return []
        metadata = self._penalty_metadata_with_scale(
            {
                "term_type": self.term_type,
                "basis_name": self.basis_name,
                "feature": list(self.feature),
                "label": self.label,
                "by": self.by,
                "by_name": self._by_state.feature_name,
                "constraint_mode": self.constraint_mode,
                "constraint_kind": self.constraint_kind,
                "knots": self.knots,
                "xt": self.xt,
                "levels": list(self._setup.levels),
                "fixed": self.fixed,
            },
            penalty_index=0,
        )
        return self._build_penalty_block(
            self.penalties[0],
            rank=min(int(self._setup.rank), int(self.penalties[0].shape[0])),
            smooth_metadata=metadata,
            selection_metadata={**metadata, "is_selection_penalty": True},
        )

    def transform_new(self, X_new):
        self._require_fitted()
        basis = predict_markov_random_field(
            column_as_object(X_new, self._feature_index), self._setup
        )
        return self._apply_constraint_transform_and_by(basis, X_new)

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del apply_np, x_train
        self._require_fitted()
        setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
        setup_penalty = np.asarray(self._setup.penalty, dtype=np.float64)
        if centered:
            return super().tensor_marginal_fit_matrices(centered=True)
        return setup_base, setup_penalty, None

    def factor_smooth_penalty_rank(self) -> int:
        self._require_fitted()
        return int(self._setup.rank)

    def validate_factor_smooth_base(self, mode: str) -> None:
        self._require_fitted()
        if str(mode).lower() == "fs" and self._setup.used_low_rank:
            raise NotImplementedError(
                "mgcv 1.9-4 cannot predict an fs smooth with a reduced-rank "
                "MRF base because its factor-smooth P matrix is dimensionally "
                "incompatible with the full region indicator."
            )

    def factor_smooth_reparameterize_prediction(self, basis, transform):
        """Preserve mgcv's observable double MRF transform inside ``fs``."""
        transform = np.asarray(transform, dtype=np.float64)
        return np.asarray(basis, dtype=np.float64) @ transform @ transform

    def factor_smooth_prediction_basis_map(self, transform, n_levels: int):
        transform = np.asarray(transform, dtype=np.float64)
        return np.kron(
            np.eye(int(n_levels), dtype=np.float64), np.linalg.inv(transform)
        )

    def tensor_marginal_predict_matrix(
        self, X_new, *, centered=False, np_transform=None
    ):
        basis = (
            self.transform_new(X_new)
            if centered
            else predict_markov_random_field(
                column_as_object(X_new, self._feature_index), self._setup
            )
        )
        if np_transform is not None:
            basis = basis @ np.asarray(np_transform, dtype=np.float64)
        return np.asarray(basis, dtype=np.float64)


__all__ = [
    "MarkovRandomFieldSetup",
    "MarkovRandomFieldTerm",
    "build_markov_random_field_setup",
    "predict_markov_random_field",
]
