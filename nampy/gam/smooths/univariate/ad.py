"""Adaptive one- and two-dimensional smooths (``bs='ad'``).

This is an operation-oriented port of ``mgcv::smooth.construct.ad.smooth.spec``.
The regression basis is an ordinary P-spline, cyclic P-spline, cubic regression
spline, or cyclic cubic spline.  Adaptivity is represented by a basis over the
second-difference penalty, producing one PSD penalty matrix per adaptive-basis
column.
"""

from __future__ import annotations

import numpy as np

from ...constraints.absorption import apply_linear_constraint
from ...penalties import penalty_id_for_local_index
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ...splines.univariate.ps import (
    build_pspline_term_setup,
    predict_pspline_term,
)
from ..algebra import rowwise_kronecker
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_knots,
    build_penalty_definition,
)
from .cr import CubicSplineTerm
from .ps import PSplineTerm1D


def _as_integer_vector(value, *, size: int, default: int, name: str) -> list[int]:
    if value is None:
        return [int(default)] * int(size)
    values = [value] if np.isscalar(value) else list(np.asarray(value).ravel())
    if len(values) == 1 and size > 1:
        values *= size
    if len(values) != size:
        raise ValueError(f"{name} must have length 1 or {size}, got {len(values)}.")
    out = []
    for item in values:
        if item is None or (isinstance(item, float) and np.isnan(item)):
            out.append(int(default))
            continue
        numeric = float(item)
        if not np.isfinite(numeric) or numeric != np.rint(numeric):
            raise ValueError(f"{name} entries must be integers or NA.")
        out.append(int(numeric))
    return out


def _adaptive_base_name(xt) -> str:
    if isinstance(xt, dict):
        value = xt.get("bs", None)
    elif isinstance(xt, str):
        value = xt
    elif isinstance(xt, (list, tuple, np.ndarray)) and len(xt):
        value = list(xt)[0]
    else:
        value = None
    if isinstance(value, (list, tuple, np.ndarray)):
        value = list(value)[0] if len(value) else None
    name = "ps" if value is None else str(value).lower()
    return name if name in {"cc", "cr", "ps", "cp"} else "ps"


def _adaptive_rank(matrix: np.ndarray, *, power: float, multiplier: float = 1.0) -> int:
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix).T)
    )
    largest = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    if largest <= 0.0:
        return 0
    return int(
        np.sum(eigenvalues > largest * multiplier * np.finfo(np.float64).eps ** power)
    )


def _d2_grid(ni: int, nj: int) -> dict[str, np.ndarray]:
    """Port mgcv's internal ``D2`` coefficient-grid difference matrices."""
    ni, nj = int(ni), int(nj)
    if ni < 3 or nj < 3:
        raise ValueError("Two-dimensional adaptive bases require k >= 3 per margin.")
    indices = np.arange(ni * nj, dtype=int).reshape((ni, nj), order="F")
    row_coord = np.tile(np.arange(1, ni + 1, dtype=np.float64), nj)
    col_coord = np.repeat(np.arange(1, nj + 1, dtype=np.float64), ni)

    def flat(rows, cols):
        return indices[np.ix_(rows, cols)].reshape(-1, order="F")

    rr_center = flat(np.arange(1, ni - 1), np.arange(nj))
    drr = np.zeros((rr_center.size, ni * nj), dtype=np.float64)
    rr_rows = np.arange(rr_center.size)
    drr[rr_rows, rr_center] = -2.0
    drr[rr_rows, flat(np.arange(ni - 2), np.arange(nj))] = 1.0
    drr[rr_rows, flat(np.arange(2, ni), np.arange(nj))] = 1.0

    cc_center = flat(np.arange(ni), np.arange(1, nj - 1))
    dcc = np.zeros((cc_center.size, ni * nj), dtype=np.float64)
    cc_rows = np.arange(cc_center.size)
    dcc[cc_rows, cc_center] = -2.0
    dcc[cc_rows, flat(np.arange(ni), np.arange(nj - 2))] = 1.0
    dcc[cc_rows, flat(np.arange(ni), np.arange(2, nj))] = 1.0

    cr_center = flat(np.arange(1, ni - 1), np.arange(1, nj - 1))
    dcr = np.zeros((cr_center.size, ni * nj), dtype=np.float64)
    cr_rows = np.arange(cr_center.size)
    weight = np.sqrt(0.125)
    dcr[cr_rows, flat(np.arange(ni - 2), np.arange(nj - 2))] = weight
    dcr[cr_rows, flat(np.arange(2, ni), np.arange(2, nj))] = weight
    dcr[cr_rows, flat(np.arange(ni - 2), np.arange(2, nj))] = -weight
    dcr[cr_rows, flat(np.arange(2, ni), np.arange(nj - 2))] = -weight

    return {
        "Dcc": dcc,
        "Drr": drr,
        "Dcr": dcr,
        "rr_ri": row_coord[rr_center],
        "rr_ci": col_coord[rr_center],
        "cc_ri": row_coord[cc_center],
        "cc_ci": col_coord[cc_center],
        "cr_ri": row_coord[cr_center],
        "cr_ci": col_coord[cr_center],
        "rmt": row_coord,
        "cmt": col_coord,
    }


def _weighted_crossproduct(difference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    difference = np.asarray(difference, dtype=np.float64)
    return difference.T @ (np.asarray(weights, dtype=np.float64)[:, None] * difference)


@register_smooth("ad")
class AdaptiveSmoothTerm(BaseSmoothTerm):
    """Runtime owner for mgcv-compatible adaptive smooth bases and penalties."""

    term_type = "smooth"
    basis_name = "ad"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=-1,
        m=None,
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
        if len(features) not in {1, 2}:
            raise ValueError("Adaptive smooths require one or two features.")
        super().__init__(
            feature=features[0] if len(features) == 1 else features,
            label=label or f"s({', '.join(map(str, features))})",
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.features = features
        self.dim = len(features)
        default_k = 40 if self.dim == 1 else 15
        if np.isscalar(k) and int(k) < 0:
            k = default_k
        self.k = _as_integer_vector(k, size=self.dim, default=default_k, name="k")
        self.m = _as_integer_vector(
            m, size=self.dim, default=(5 if self.dim == 1 else 3), name="m"
        )
        self.xt = xt
        self.base_basis = _adaptive_base_name(xt)
        self.penalty_basis = "cp" if self.base_basis in {"cc", "cp"} else "ps"
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.pc = pc
        self.knots = _normalize_knots(knots, features)
        self.null_penalty_tol = float(null_penalty_tol)
        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of {'auto', 'factor_by', 'always', 'never'}."
            )
        if self.select and self.fixed:
            raise ValueError("select=True and fixed=True are incompatible.")

        self._marginals = None
        self._feature_indices = None
        self._feature_names = None
        self._basis_train = None
        self._basis_setup = None
        self._penalties = None
        self.rank = []
        self.penalty_labels = []

    @property
    def expected_linked_penalty_count(self):
        count = int(np.prod(self.m)) if all(v > 0 for v in self.m) else 0
        return count + int(self.select and count > 0)

    def _base_marginal(self, feature, k, knots):
        kwargs = {
            "feature": feature,
            "k": int(k),
            "label": str(feature),
            "by": None,
            "sp": None,
            "select": False,
            "fixed": False,
            "constraint_mode": "never",
            "pc": None,
            "knots": knots,
            "metadata": self.metadata,
        }
        if self.base_basis in {"cr", "cc"}:
            return CubicSplineTerm(basis=self.base_basis, **kwargs)
        return PSplineTerm1D(basis=self.base_basis, m=(2, 2), **kwargs)

    def _fit_base(self, X, feature_names):
        marginals = []
        indices = []
        names = []
        bases = []
        base_penalties = []
        for feature, k_i, knots_i in zip(
            self.features, self.k, self.knots, strict=True
        ):
            marginal = self._base_marginal(feature, k_i, knots_i)
            marginal.fit(X, feature_names)
            basis, penalty, _ = marginal.tensor_marginal_fit_matrices(centered=False)
            marginals.append(marginal)
            indices.extend(marginal.resolved_feature_indices())
            names.extend(marginal.resolved_feature_names_list())
            bases.append(np.asarray(basis, dtype=np.float64))
            base_penalties.append(np.asarray(penalty, dtype=np.float64))
        return marginals, indices, names, bases, base_penalties

    def _one_dimensional_penalties(self, n_coef: int, base_penalty: np.ndarray):
        n_penalty = int(self.m[0])
        if n_penalty >= n_coef - 2:
            raise ValueError("penalty basis too large for smoothing basis")
        if n_penalty <= 0:
            return [], []
        if n_penalty == 1:
            penalty = np.asarray(base_penalty, dtype=np.float64)
            return [penalty], [_adaptive_rank(penalty, power=0.9)]
        grid = np.arange(1, n_coef - 1, dtype=np.float64) / float(n_coef)
        penalty_order = 1 if n_penalty == 3 else 2
        if n_penalty == 2:
            weights = np.column_stack([np.ones(n_coef - 2), grid])
        else:
            setup = build_pspline_term_setup(
                grid,
                feature_index=0,
                feature_name="x",
                bs_dim=n_penalty,
                m=(penalty_order, penalty_order),
                basis=self.penalty_basis,
            )
            weights = np.asarray(setup.basis_train, dtype=np.float64)
        difference = np.diff(np.eye(n_coef, dtype=np.float64), n=2, axis=0)
        penalties = [
            _weighted_crossproduct(difference, weights[:, j]) for j in range(n_penalty)
        ]
        ranks = [_adaptive_rank(penalty, power=0.9) for penalty in penalties]
        return penalties, ranks

    def _penalty_tensor_basis(self, grid, kp):
        order = min(min(kp) - 2, 1)
        # The constructor forms the penalty basis through te(...). te() resets
        # any marginal k below 3 to its default 5, while the adaptive outer
        # loop still consumes only prod(kp) columns from that model matrix.
        basis_dimensions = [5, 5] if any(value < 3 for value in kp) else kp
        setups = []
        for values, dimension, name in zip(
            (grid["rmt"], grid["cmt"]), basis_dimensions, ("i", "j"), strict=True
        ):
            setups.append(
                build_pspline_term_setup(
                    values,
                    feature_index=0,
                    feature_name=name,
                    bs_dim=int(dimension),
                    m=(order, order),
                    basis=self.penalty_basis,
                )
            )

        def evaluate(rows, cols):
            return rowwise_kronecker(
                [
                    predict_pspline_term(rows, setups[0]),
                    predict_pspline_term(cols, setups[1]),
                ]
            )

        return (
            evaluate(grid["rr_ri"], grid["rr_ci"]),
            evaluate(grid["cc_ri"], grid["cc_ci"]),
            evaluate(grid["cr_ri"], grid["cr_ci"]),
        )

    def _two_dimensional_penalties(self):
        kp = [int(v) for v in self.m]
        n_penalty = int(np.prod(kp))
        n_difference = int((self.k[0] - 2) * (self.k[1] - 2))
        if n_penalty > n_difference:
            raise ValueError("penalty basis too large for smoothing basis")
        if n_penalty <= 0:
            return [], []
        grid = _d2_grid(*self.k)
        if n_penalty == 1:
            penalty = sum(
                np.asarray(grid[name]).T @ np.asarray(grid[name])
                for name in ("Drr", "Dcc", "Dcr")
            )
            return [penalty], [penalty.shape[0] - 3]
        if n_penalty == 3:
            raise ValueError(
                "The upstream planar adaptive-penalty branch is not numerically defined."
            )
        if any(value < 2 for value in kp):
            raise ValueError("penalty basis too small")
        vrr, vcc, vcr = self._penalty_tensor_basis(grid, kp)
        penalties = []
        for column in range(n_penalty):
            penalties.append(
                _weighted_crossproduct(grid["Drr"], vrr[:, column])
                + _weighted_crossproduct(grid["Dcc"], vcc[:, column])
                + _weighted_crossproduct(grid["Dcr"], vcr[:, column])
            )
        ranks = [
            _adaptive_rank(penalty, power=1.0, multiplier=10.0) for penalty in penalties
        ]
        return penalties, ranks

    def fit(self, X, feature_names):
        X = np.asarray(X, dtype=object)
        self._set_by_state(X, feature_names)
        marginals, indices, names, bases, base_penalties = self._fit_base(
            X, feature_names
        )
        basis = bases[0] if self.dim == 1 else rowwise_kronecker(bases)
        raw_penalties, ranks = (
            self._one_dimensional_penalties(basis.shape[1], base_penalties[0])
            if self.dim == 1
            else self._two_dimensional_penalties()
        )
        self._raw_basis_train = np.asarray(basis, dtype=np.float64).copy()
        self._raw_penalties = [
            np.asarray(penalty, dtype=np.float64).copy() for penalty in raw_penalties
        ]
        if not raw_penalties:
            self.fixed = True
        penalty_scales = [penalty_rescale_factor(basis, S) for S in raw_penalties]
        penalties = [scale_penalty(basis, S) for S in raw_penalties]

        factor_by = bool(self.metadata.get("factor_by", None))
        should_constrain = self.constraint_mode == "always" or (
            self.constraint_mode == "auto" and self._by_state.is_constant
        )
        if self.constraint_mode == "factor_by":
            if not self._by_state.is_present:
                raise ValueError(
                    "constraint_mode='factor_by' requires a numeric indicator `by` column."
                )
            should_constrain = True
            factor_by = True

        transform = None
        constraint_kind = None
        if self.pc is not None:
            max_index = max(indices)

            def point_basis_fn(point):
                point_data = np.zeros((point.shape[0], max_index + 1), dtype=np.float64)
                point_data[:, indices] = point
                blocks = [
                    marginal.tensor_marginal_predict_matrix(
                        point_data, centered=False, np_transform=None
                    )
                    for marginal in marginals
                ]
                return blocks[0] if self.dim == 1 else rowwise_kronecker(blocks)

            basis, penalties, transform, _ = self._apply_point_constraint(
                basis,
                penalties,
                self.pc,
                feature_names=names,
                point_basis_fn=point_basis_fn,
                fixed=self.fixed or not penalties,
            )
            constraint_kind = "pc"
        elif should_constrain or factor_by:
            basis, penalties, transform = apply_linear_constraint(
                basis,
                [] if self.fixed else penalties,
                basis.mean(axis=0),
            )
            basis = self._apply_cached_by(basis)
            constraint_kind = "factor_by" if factor_by else "sum_to_zero"
        else:
            basis = self._apply_cached_by(basis)
            if self.fixed:
                penalties = []

        self._marginals = marginals
        self._feature_indices = indices
        self._feature_names = names
        self._set_resolved_features(names)
        self._basis_setup = np.asarray(
            bases[0] if self.dim == 1 else rowwise_kronecker(bases)
        )
        self._basis_train = np.asarray(basis, dtype=np.float64)
        self._penalties = [np.asarray(S, dtype=np.float64) for S in penalties]
        self.rank = list(ranks)
        self.penalty_labels = [
            f"{self.label}.{j + 1}" for j in range(len(raw_penalties))
        ]
        self._set_penalty_rescale_factors([] if self.fixed else penalty_scales)
        self._record_constraint_result(
            constraint_kind,
            transform,
            absorbed_by="runtime" if transform is not None else None,
        )
        return self

    def get_penalty_definitions(self):
        self._require_fitted()
        raw = list(self.penalties)
        if not raw:
            return []
        sp_values = self._normalized_term_sp(len(raw))
        definitions = []
        for index, penalty in enumerate(raw):
            smoothing_id = (
                None
                if self.smoothing_id is None
                else penalty_id_for_local_index(
                    self.smoothing_id, index, n_penalties=len(raw)
                )
            )
            definitions.append(
                build_penalty_definition(
                    self,
                    penalty,
                    smoothing_id=smoothing_id,
                    sp_value_in=sp_values[index],
                    rank=(self.rank[index] if index < len(self.rank) else None),
                    metadata_extra={
                        "adaptive_penalty_index": index,
                        "adaptive_penalty_label": self.penalty_labels[index],
                        "term_sp": sp_values[index],
                    },
                    local_penalty_index=index,
                )
            )
        definitions.extend(
            self._build_selection_penalty_definitions(
                raw,
                null_penalty_tol=self.null_penalty_tol,
            )
        )
        return definitions

    def transform_new(self, X_new):
        self._require_fitted()
        blocks = [
            marginal.tensor_marginal_predict_matrix(
                X_new, centered=False, np_transform=None
            )
            for marginal in self._marginals
        ]
        basis = blocks[0] if self.dim == 1 else rowwise_kronecker(blocks)
        return self._apply_constraint_transform_and_by(basis, X_new)


__all__ = ["AdaptiveSmoothTerm"]
