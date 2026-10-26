"""Runtime ownership of mgcv's soap-film (so/sf/sw) smooths."""

from __future__ import annotations

import numpy as np

from ...constraints.absorption import apply_linear_constraint
from ...penalties import penalty_id_for_local_index
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ...splines.soap import build_soap_basis, predict_soap_basis
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_knots,
    _resolve_feature,
    build_penalty_definition,
    by_values_from_new_data,
    columns_as_float_matrix,
    linear_functional_basis,
    linear_functional_by_state,
    linear_functional_locations,
    linear_functional_setup_points,
)


@register_smooth("so")
@register_smooth("sf")
@register_smooth("sw")
class SoapFilmTerm(BaseSmoothTerm):
    """A soap film on a bounded plane, with optional known boundary values.

    ``xt['bnd']`` is a list of boundary loops, each mapping the two feature
    names to coordinate arrays. An optional ``f`` array fixes that loop's
    boundary values. ``knots`` supplies paired interior coordinates.
    """

    term_type = "smooth"

    def __init__(
        self,
        feature,
        k=-1,
        basis="so",
        xt=None,
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        constraint_mode="auto",
        pc=None,
        knots=None,
        null_penalty_tol=1e-10,
        metadata=None,
    ):
        features = [feature] if isinstance(feature, (str, int)) else list(feature)
        if len(features) != 2:
            raise ValueError("soap films are bivariate only")
        if basis not in {"so", "sf", "sw"}:
            raise ValueError("Soap basis must be 'so', 'sf', or 'sw'.")
        super().__init__(
            feature=features,
            label=label or f"s({', '.join(map(str, features))})",
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.basis_name = basis
        self.k, self.xt, self.knots, self.pc = k, xt, knots, pc
        self.select, self.fixed = bool(select), bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.null_penalty_tol = float(null_penalty_tol)
        if self.constraint_mode not in {"auto", "always", "never", "factor_by"}:
            raise ValueError("Invalid soap constraint_mode.")
        self._basis_train = self._penalties = self._setup = None
        self._feature_indices = self._feature_names = None
        self._linear_functional = False

    @property
    def expected_linked_penalty_count(self):
        return None

    def _build_setup(self, values):
        if not isinstance(self.xt, dict) or self.xt.get("bnd") is None:
            raise ValueError("can't soap smooth without a boundary")
        boundaries = self.xt["bnd"]
        if not isinstance(boundaries, (list, tuple)):
            raise ValueError("bnd must be a list of boundary loops")
        loops = []
        for boundary in boundaries:
            if any(name not in boundary for name in self._feature_names):
                raise ValueError("faulty bnd: each loop must name both smooth features")
            loop = dict(boundary)
            loop["x"], loop["y"] = (
                boundary[self._feature_names[0]],
                boundary[self._feature_names[1]],
            )
            loops.append(loop)
        if self.knots is None:
            raise ValueError("knots must be specified for soap")
        knots = _normalize_knots(self.knots, self.feature)
        if any(value is None for value in knots):
            raise ValueError("knots must specify both soap coordinates")
        return build_soap_basis(
            values[:, 0],
            values[:, 1],
            boundary=loops,
            knots={"x": knots[0], "y": knots[1]},
            k=self.k,
            nmax=self.xt.get("nmax", 200),
            basis=self.basis_name,
            boundary_spec=self.xt.get("bndSpec"),
        )

    def fit(self, X, feature_names):
        resolved = [_resolve_feature(f, feature_names) for f in self.feature]
        self._feature_indices = [item[0] for item in resolved]
        self._feature_names = [item[1] for item in resolved]
        self._set_resolved_features(self._feature_names)
        self._set_by_state(X, feature_names)
        locations = linear_functional_locations(X, self._feature_indices)
        self._linear_functional = locations.ndim == 3
        values = locations.reshape(-1, 2)
        weights = None
        if self._linear_functional:
            weights = self._by_state.values
            if weights is None:
                weights = np.ones(locations.shape[:2])
            if np.asarray(weights).shape != locations.shape[:2]:
                raise ValueError(
                    "Linear-functional feature locations and by weights must have equal shape."
                )
            self._by_state = linear_functional_by_state(self._by_state, weights=weights)
        shared_X = self._linked_id_setup_matrix(feature_names)
        setup_locations = (
            locations if shared_X is None
            else linear_functional_locations(shared_X, self._feature_indices)
        )
        setup_values = (
            linear_functional_setup_points(setup_locations)[0]
            if setup_locations.ndim == 3 else setup_locations
        )
        self._setup = self._build_setup(setup_values)
        setup_basis = np.asarray(self._setup.basis_train, dtype=float)
        if shared_X is None:
            basis, offset = setup_basis, self._setup.offset
        else:
            basis, offset = predict_soap_basis(values[:, 0], values[:, 1], self._setup)
        if self._linear_functional:
            if offset is not None:
                raise NotImplementedError(
                    "The mgcv 1.9-4 reference fails when aggregating known-boundary offsets "
                    "with matrix-valued soap arguments; NAMpy retains this guard."
                )
            basis = linear_functional_basis(
                locations,
                weights,
                lambda points: predict_soap_basis(
                    points[:, 0], points[:, 1], self._setup
                )[0],
            )
        penalties = [scale_penalty(setup_basis, S) for S in self._setup.penalties]
        self._set_penalty_rescale_factors(
            [penalty_rescale_factor(setup_basis, S) for S in self._setup.penalties]
            if not self.fixed
            else []
        )
        self._raw_basis_train = np.asarray(basis).copy()
        self._raw_penalties = [S.copy() for S in self._setup.penalties]
        self.rank = list(self._setup.ranks)
        self.null_space_dim = self._setup.null_space_dim
        transform, kind = None, None
        if self.pc is not None:
            basis, penalties, transform, _ = self._apply_point_constraint(
                basis,
                penalties,
                self.pc,
                feature_names=self._feature_names,
                point_basis_fn=lambda points: predict_soap_basis(
                    points[:, 0], points[:, 1], self._setup
                )[0],
                fixed=self.fixed,
            )
            kind = "pc"
        else:
            # soap.r explicitly supplies a zero-row C for sw and known bounds.
            constrain = self._setup.need_constraint and (
                self.constraint_mode in {"always", "factor_by"}
                or (self.constraint_mode == "auto" and self._by_state.is_constant)
                or bool(self.metadata.get("factor_by"))
            )
            if constrain:
                basis, penalties, transform = apply_linear_constraint(
                    basis, [] if self.fixed else penalties, setup_basis.mean(axis=0)
                )
                kind = "sum_to_zero"
            basis = self._apply_cached_by(basis)
        self._basis_train = np.asarray(basis, dtype=float)
        self._penalties = [] if self.fixed else penalties
        self.prediction_offset = self._scale_offset(offset, self._by_state.values)
        self._record_constraint_result(
            kind,
            transform,
            absorbed_by="runtime"
            if transform is not None or not self._setup.need_constraint
            else None,
        )
        return self

    @staticmethod
    def _scale_offset(offset, by):
        if offset is None:
            return None
        return np.asarray(offset, dtype=float) * (1.0 if by is None else by)

    def offset_new(self, X_new):
        if self._linear_functional:
            return None
        values = columns_as_float_matrix(X_new, self._feature_indices)
        _, offset = predict_soap_basis(values[:, 0], values[:, 1], self._setup)
        return self._scale_offset(
            offset, by_values_from_new_data(X_new, self._by_state)
        )

    def transform_new(self, X_new):
        self._require_fitted()
        locations = linear_functional_locations(X_new, self._feature_indices)
        if locations.ndim == 3:
            basis = linear_functional_basis(
                locations,
                by_values_from_new_data(X_new, self._by_state),
                lambda points: predict_soap_basis(
                    points[:, 0], points[:, 1], self._setup
                )[0],
            )
            if self.constraint_transform is not None:
                basis = basis @ self.constraint_transform
            return np.asarray(basis, dtype=float)
        basis, _ = predict_soap_basis(locations[:, 0], locations[:, 1], self._setup)
        return self._apply_constraint_transform_and_by(basis, X_new)

    @property
    def tensor_requires_centering(self):
        """Zero-boundary films supply an explicit zero-row constraint in mgcv."""
        return bool(self._setup.need_constraint)

    def factor_smooth_penalty_rank(self):
        """FS passes the soap constructor's declared rank to nat.param."""
        self._require_fitted()
        if len(self._setup.ranks) != 1:
            raise NotImplementedError(
                "Factor smooths require a single soap penalty."
            )
        return int(self._setup.ranks[0])

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del apply_np, x_train
        self._require_fitted()
        if len(self._setup.penalties) != 1:
            raise NotImplementedError(
                "Tensor products of soap smooths with multiple penalties are not supported."
            )
        if centered:
            return self.basis_train, self.penalties[0], None
        return self._setup.basis_train, self._setup.penalties[0], None

    def tensor_marginal_predict_matrix(
        self, X_new, *, centered=False, np_transform=None
    ):
        if centered:
            basis = self.transform_new(X_new)
        else:
            values = columns_as_float_matrix(X_new, self._feature_indices)
            basis, _ = predict_soap_basis(values[:, 0], values[:, 1], self._setup)
        if np_transform is not None:
            basis = basis @ np_transform
        return np.asarray(basis, dtype=float)

    def get_penalty_definitions(self):
        self._require_fitted()
        if not self.penalties:
            return []
        sp_values = self._normalized_term_sp(len(self.penalties))
        definitions = []
        for index, penalty in enumerate(self.penalties):
            sid = (
                None
                if self.smoothing_id is None
                else penalty_id_for_local_index(
                    self.smoothing_id, index, n_penalties=len(self.penalties)
                )
            )
            definitions.append(
                build_penalty_definition(
                    self,
                    penalty,
                    smoothing_id=sid,
                    sp_value_in=sp_values[index],
                    rank=self.rank[index],
                    local_penalty_index=index,
                    metadata_extra={
                        "original_null_space_dim": self._setup.null_space_dim
                    },
                )
            )
        definitions.extend(
            self._build_selection_penalty_definitions(
                self.penalties,
                null_penalty_tol=self.null_penalty_tol,
            )
        )
        return definitions
