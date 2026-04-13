"""
P-spline smooth term (``bs='ps'``).

Implements the :class:`BaseSmoothTerm` interface for a P-spline: a B-spline
basis with a discrete difference penalty on adjacent coefficients.  Unlike
regression splines, P-splines do not require a set of knots to be chosen
ahead of time; instead, many equally-spaced knots are used and the smoothness
is controlled entirely by the penalty order and the smoothing parameter.
"""

import numpy as np

from ....splines.univariate_bases import (
    pspline_difference_penalty,
    pspline_knots,
    pspline_predict_matrix,
)
from ...constraints.absorption import apply_linear_constraint
from ...penalties.algebra import scale_penalty
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_point_constraint,
    _resolve_feature,
    column_as_float,
    resolve_by_state,
    sync_by_state_attributes,
)


@register_smooth("ps")
class PSplineTerm1D(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "ps"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=10,
        basis="ps",
        m=None,
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
        super().__init__(
            feature=feature,
            label=label,
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.k = int(k)
        self.basis_name = str(basis).lower()
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.pc = pc
        self.knots = knots
        self.null_penalty_tol = float(null_penalty_tol)

        if m is None:
            self.m = (2, 2)
        elif np.isscalar(m):
            self.m = (int(m), int(m))
        else:
            vals = tuple(int(v) for v in m)
            if len(vals) == 1:
                self.m = (vals[0], vals[0])
            elif len(vals) == 2:
                self.m = vals
            else:
                raise ValueError("For bs='ps', m must have length 1 or 2.")

        if self.basis_name != "ps":
            raise NotImplementedError(
                f"PSplineTerm1D currently supports only basis='ps', got {basis!r}."
            )
        if self.select and self.fixed:
            raise ValueError("select=True and fixed=True are incompatible.")
        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of "
                "{'auto', 'factor_by', 'always', 'never'}."
            )

        self._feature_index = None
        self._feature_name = None
        self._by_state = None

        self._basis_train = None
        self._penalties = None
        self._ps_knots = None

    def fit(self, X, feature_names):
        idx, feature_name = _resolve_feature(self.feature, feature_names)
        self._feature_index = idx
        self._feature_name = feature_name
        self._set_resolved_features([feature_name])

        xj = column_as_float(X, idx)

        self._by_state = resolve_by_state(self.by, X, feature_names)
        sync_by_state_attributes(self, self._by_state)

        basis_order, penalty_order = self.m
        if basis_order < 0 or penalty_order < 0:
            raise ValueError("For bs='ps', m entries must be >= 0.")

        self._ps_knots = pspline_knots(
            xj,
            bs_dim=self.k,
            basis_order=basis_order,
            supplied_knots=self.knots,
        )

        degree = basis_order + 1
        from ....splines.univariate_bases import bspline_design_matrix

        base = bspline_design_matrix(
            xj,
            self._ps_knots,
            degree=degree,
            deriv=0,
            extrapolate=True,
        )

        if self.pc is not None:
            pc_value = _normalize_point_constraint(self.pc, self._feature_name)
            pc_basis = pspline_predict_matrix(
                np.asarray([pc_value], dtype=np.float64),
                self._ps_knots,
                basis_order=basis_order,
                deriv=0,
            )[0]
            S = pspline_difference_penalty(base.shape[1], penalty_order)
            main_penalty = 0.5 * (S + S.T)
            penalties_in = [] if self.fixed else [main_penalty]
            Bc, Sc, C = apply_linear_constraint(base, penalties_in, pc_basis)
            if self._by_state.is_present:
                Bc = Bc * self._by_state.values[:, None]
            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._record_constraint_result("pc", C, absorbed_by="runtime")
            return self

        S_raw = pspline_difference_penalty(base.shape[1], penalty_order)
        main_penalty = scale_penalty(base, 0.5 * (S_raw + S_raw.T))

        if self.constraint_mode == "factor_by":
            if not self._by_state.is_present:
                raise ValueError(
                    "constraint_mode='factor_by' requires a numeric indicator `by` column."
                )
            penalties_in = [] if self.fixed else [main_penalty]
            mean_row = base.mean(axis=0)
            Bc_raw, Sc, C = apply_linear_constraint(base, penalties_in, mean_row)
            Bc = Bc_raw * self._by_state.values[:, None]
            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._record_constraint_result("factor_by", C, absorbed_by="runtime")
            return self

        if self.constraint_mode == "never":
            if self._by_state.is_present:
                base = base * self._by_state.values[:, None]
            self._basis_train = np.asarray(base, dtype=np.float64)
            self._penalties = (
                [] if self.fixed else [np.asarray(main_penalty, dtype=np.float64)]
            )
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        penalties_in = [] if self.fixed else [main_penalty]
        mean_row = base.mean(axis=0)
        Bc, Sc, C = apply_linear_constraint(base, penalties_in, mean_row)
        if self._by_state.is_present:
            Bc = Bc * self._by_state.values[:, None]
        self._basis_train = np.asarray(Bc, dtype=np.float64)
        self._penalties = Sc
        self._record_constraint_result("centering", C, absorbed_by="runtime")
        return self

    @property
    def basis_train(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._basis_train

    @property
    def penalties(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._penalties

    @property
    def n_coef(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return int(self._basis_train.shape[1])

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        if len(self.penalties) == 0:
            return []

        smooth_meta = {
            "term_type": self.term_type,
            "basis_name": self.basis_name,
            "feature": self.feature,
            "label": self.label,
            "by": self.by,
            "by_name": self._by_state.feature_name,
            "by_is_constant": bool(self._by_state.is_constant),
            "constraint_mode": self.constraint_mode,
            "pc": self.pc,
            "knots": self.knots,
            "m": self.m,
            "fixed": bool(self.fixed),
        }
        selection_meta = {
            **smooth_meta,
            "constraint_kind": self.constraint_kind,
            "is_selection_penalty": True,
        }
        return self._build_penalty_block(
            self.penalties[0],
            smooth_metadata=smooth_meta,
            selection_metadata=selection_meta,
        )

    def transform_new(self, X_new):
        self._require_fitted()

        xj = column_as_float(X_new, self._feature_index)

        B = pspline_predict_matrix(
            xj,
            self._ps_knots,
            basis_order=self.m[0],
            deriv=0,
        )

        return self._apply_constraint_transform_and_by(B, X_new)
