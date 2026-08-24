"""
P-spline smooth terms (``bs='ps'`` and cyclic ``bs='cp'``).

Implements the :class:`BaseSmoothTerm` interface for a P-spline: a B-spline
basis with a discrete coefficient-difference penalty. Cyclic P-splines use a
wrapped basis, a circular difference penalty, and periodic newdata mapping.
"""

import numpy as np

from ...constraints.absorption import fit_single_penalty_with_setup_basis
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ...splines.univariate.ps import (
    build_pspline_term_setup,
    predict_pspline_term,
    predict_pspline_term_derivative,
)
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _resolve_feature,
    by_values_from_new_data,
    column_as_numeric_array,
    linear_functional_basis,
    linear_functional_by_state,
)


@register_smooth("ps")
@register_smooth("cp")
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

        def normalize_order(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return 2
            numeric = float(value)
            if not np.isfinite(numeric) or numeric != np.rint(numeric):
                raise ValueError(
                    f"For bs={self.basis_name!r}, m entries must be integers or NA."
                )
            return int(numeric)

        if m is None:
            self.m = (2, 2)
        elif np.isscalar(m):
            value = normalize_order(m)
            self.m = (value, value)
        else:
            vals = tuple(normalize_order(v) for v in m)
            if len(vals) == 1:
                self.m = (vals[0], vals[0])
            elif len(vals) == 2 or (self.basis_name == "cp" and len(vals) > 2):
                self.m = vals
            else:
                raise ValueError(
                    f"For bs={self.basis_name!r}, m must have length 1 or 2."
                )

        if self.basis_name not in {"ps", "cp"}:
            raise NotImplementedError(
                "PSplineTerm1D supports only basis in {'ps', 'cp'}, "
                f"got {basis!r}."
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
        self._setup = None
        self._linear_functional = False

    def fit(self, X, feature_names):
        self._X_train = np.asarray(X, dtype=object).copy()
        idx, feature_name = _resolve_feature(self.feature, feature_names)
        self._feature_index = idx
        self._feature_name = feature_name
        self._set_resolved_features([feature_name])

        xj = column_as_numeric_array(X, idx)

        self._set_by_state(X, feature_names)
        self._linear_functional = np.asarray(xj).ndim == 2
        if self._linear_functional:
            if self._by_state is None or np.asarray(self._by_state.values).ndim != 2:
                raise ValueError(
                    "P-spline linear-functional terms require matrix-valued by weights."
                )
            by_weights = np.asarray(self._by_state.values, dtype=np.float64)
            if by_weights.shape != np.asarray(xj).shape:
                raise ValueError(
                    "Linear-functional feature locations and by weights must have equal shape."
                )
            x_setup_values = np.asarray(xj, dtype=np.float64).reshape(-1)
            self._by_state = linear_functional_by_state(self._by_state)
        else:
            x_setup_values = np.asarray(xj, dtype=np.float64).reshape(-1)

        basis_order, penalty_order = self.m[:2]
        if basis_order < 0 or penalty_order < 0:
            raise ValueError(
                f"For bs={self.basis_name!r}, m entries must be >= 0."
            )

        shared_X = self._linked_id_setup_matrix(feature_names)
        if shared_X is not None:
            x_setup = column_as_numeric_array(shared_X, idx)
            if np.asarray(x_setup).ndim != 1:
                raise NotImplementedError(
                    "Linked-id pooling is not available for linear-functional P-splines."
                )
            self._setup = build_pspline_term_setup(
                x_setup,
                feature_index=idx,
                feature_name=feature_name,
                bs_dim=self.k,
                m=self.m,
                knots=self.knots,
                basis=self.basis_name,
            )
            setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
            base = np.asarray(predict_pspline_term(xj, self._setup), dtype=np.float64)
            main_penalty = scale_penalty(
                setup_base, np.asarray(self._setup.penalty, dtype=np.float64)
            )
        else:
            self._setup = build_pspline_term_setup(
                x_setup_values,
                feature_index=idx,
                feature_name=feature_name,
                bs_dim=self.k,
                m=self.m,
                knots=self.knots,
                basis=self.basis_name,
            )
            point_base = np.asarray(self._setup.basis_train, dtype=np.float64)
            if self._linear_functional:
                base = linear_functional_basis(
                    xj,
                    by_weights,
                    lambda values: predict_pspline_term(values, self._setup),
                )
                setup_base = np.asarray(base, dtype=np.float64)
            else:
                setup_base = point_base
                base = setup_base
            main_penalty = scale_penalty(
                base, np.asarray(self._setup.penalty, dtype=np.float64)
            )
        self._set_penalty_rescale_factors(
            [
                penalty_rescale_factor(
                    setup_base, np.asarray(self._setup.penalty, dtype=np.float64)
                )
            ]
        )

        if self.pc is not None:
            Bc, Sc, C, _ = self._apply_point_constraint(
                base,
                [main_penalty],
                self.pc,
                feature_names=[self._feature_name],
                point_basis_fn=lambda pts: predict_pspline_term(pts, self._setup)[0],
                fixed=self.fixed,
            )
            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._record_constraint_result("pc", C, absorbed_by="runtime")
            return self

        if self.constraint_mode == "never":
            base = self._apply_cached_by(base)
            self._basis_train = np.asarray(base, dtype=np.float64)
            self._penalties = (
                [] if self.fixed else [np.asarray(main_penalty, dtype=np.float64)]
            )
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        result = fit_single_penalty_with_setup_basis(
            base,
            setup_base,
            main_penalty,
            self._by_state,
            constraint_mode=self.constraint_mode,
            fixed=self.fixed,
            auto_constrain_when=True,
        )
        self._basis_train = result.basis_train
        self._penalties = result.penalties
        self._record_constraint_result(
            result.constraint_kind,
            result.constraint_transform,
            absorbed_by=(
                "runtime" if result.constraint_transform is not None else None
            ),
        )
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
        smooth_meta = self._penalty_metadata_with_scale(smooth_meta, penalty_index=0)
        return self._build_penalty_block(
            self.penalties[0],
            smooth_metadata=smooth_meta,
            selection_metadata=selection_meta,
        )

    def transform_new(self, X_new):
        self._require_fitted()

        xj = column_as_numeric_array(X_new, self._feature_index)
        if self._linear_functional:
            B = linear_functional_basis(
                xj,
                by_values_from_new_data(X_new, self._by_state),
                lambda values: predict_pspline_term(values, self._setup),
            )
            if self.constraint_transform is not None:
                B = B @ self.constraint_transform
            return np.asarray(B, dtype=np.float64)
        B = predict_pspline_term(xj, self._setup)
        return self._apply_constraint_transform_and_by(B, X_new)

    def derivative_matrix(self, X_new=None, order=1):
        """Exact derivative design for scalar P-spline terms."""
        self._require_fitted()
        order = int(order)
        degree = int(self._setup.basis_order) + 1
        if order < 1 or order > degree:
            raise ValueError(f"order must be between 1 and {degree}.")
        if self._linear_functional:
            raise NotImplementedError(
                "Derivatives of linear-functional P-spline terms require an "
                "explicit functional derivative and are not inferred."
            )
        source = self._X_train if X_new is None else X_new
        xj = column_as_numeric_array(source, self._feature_index)
        B = predict_pspline_term_derivative(xj, self._setup, deriv=order)
        return self._apply_constraint_transform_and_by(B, source)

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del apply_np, x_train
        if self._setup is None:
            raise RuntimeError("Term is not fitted.")
        setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
        setup_penalty = scale_penalty(
            setup_base, np.asarray(self._setup.penalty, dtype=np.float64)
        )
        if centered:
            if (
                self._linked_id_setup() is not None
                and self.constraint_transform is not None
            ):
                T = np.asarray(self.constraint_transform, dtype=np.float64)
                return (
                    np.asarray(setup_base @ T, dtype=np.float64),
                    np.asarray(T.T @ setup_penalty @ T, dtype=np.float64),
                    None,
                )
            return super().tensor_marginal_fit_matrices(centered=centered)
        return (
            setup_base,
            np.asarray(self._setup.penalty, dtype=np.float64),
            None,
        )

    def tensor_marginal_predict_matrix(
        self, X_new, *, centered=False, np_transform=None
    ):
        if centered:
            B = np.asarray(self.transform_new(X_new), dtype=np.float64)
        else:
            xj = column_as_numeric_array(X_new, self._feature_index)
            if xj.ndim != 1:
                raise NotImplementedError(
                    "Matrix-valued P-splines cannot be tensor marginals."
                )
            B = np.asarray(predict_pspline_term(xj, self._setup), dtype=np.float64)
        if np_transform is not None:
            B = B @ np.asarray(np_transform, dtype=np.float64)
        return np.asarray(B, dtype=np.float64)
