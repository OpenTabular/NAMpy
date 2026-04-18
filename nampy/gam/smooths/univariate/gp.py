"""
Gaussian process smooth term (``bs='gp'``).

Implements the :class:`BaseSmoothTerm` interface for a Gaussian process spline
basis.  The GP smooth uses a kernel matrix as the penalty (or its pseudo-inverse
as the basis), parameterised by a length-scale and smoothness order.

GP smooths are a flexible alternative to polynomial splines when the data
suggest a specific correlation structure or when a probabilistic interpretation
of the smooth function is desired.
"""

import numpy as np

from ....splines.gaussian_process import build_gp_term_setup, predict_gp_term
from ...constraints.absorption import (
    fit_single_penalty_with_constraint_policy,
    fit_single_penalty_with_setup_basis,
)
from ...penalties.algebra import scale_penalty
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _resolve_feature,
    columns_as_float_matrix,
)


@register_smooth("gp")
class GPSmoothTerm(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "gp"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=-1,
        basis="gp",
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
        xt=None,
        null_penalty_tol=1e-10,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]

        super().__init__(
            feature=features,
            label=label or f"s({', '.join(map(str, features))})",
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        self.k = int(k)
        self.basis_name = str(basis).lower()
        self.m = m
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.constraint_mode = str(constraint_mode).lower()
        self.pc = pc
        self.knots = knots
        self.xt = xt
        self.null_penalty_tol = float(null_penalty_tol)

        if self.basis_name != "gp":
            raise NotImplementedError(
                f"GPSmoothTerm currently supports only basis='gp', got {basis!r}."
            )
        if self.select and self.fixed:
            raise ValueError("select=True and fixed=True are incompatible.")
        if self.constraint_mode not in {"auto", "factor_by", "always", "never"}:
            raise ValueError(
                "constraint_mode must be one of "
                "{'auto', 'factor_by', 'always', 'never'}."
            )

        self._feature_indices = None
        self._feature_names = None

        self._by_state = None

        self._basis_train = None
        self._penalties = None

        self._setup = None

    def fit(self, X, feature_names):
        feature_indices = []
        feature_names_resolved = []

        for feat in self.feature:
            idx, fname = _resolve_feature(feat, feature_names)
            feature_indices.append(idx)
            feature_names_resolved.append(fname)

        Xf = columns_as_float_matrix(X, feature_indices)

        self._set_by_state(X, feature_names)

        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)

        shared_X = self._linked_id_setup_matrix(feature_names)
        if shared_X is not None:
            Xf_setup = columns_as_float_matrix(shared_X, feature_indices)
            self._setup = build_gp_term_setup(
                Xf_setup,
                k=self.k,
                m=self.m,
                knots=self.knots,
                xt=self.xt,
            )
            setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
            base = np.asarray(predict_gp_term(Xf, self._setup), dtype=np.float64)
            pen = scale_penalty(
                setup_base,
                np.asarray(self._setup.penalty, dtype=np.float64),
            )
        else:
            self._setup = build_gp_term_setup(
                Xf,
                k=self.k,
                m=self.m,
                knots=self.knots,
                xt=self.xt,
            )
            setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
            base = setup_base
            pen = scale_penalty(
                base,
                np.asarray(self._setup.penalty, dtype=np.float64),
            )

        if self.pc is not None:
            constrained = self._apply_point_constraint(
                base,
                [pen],
                self.pc,
                feature_names=self._feature_names,
                point_basis_fn=lambda pts: predict_gp_term(pts, self._setup)[0],
                fixed=self.fixed,
            )
            Bc, Sc, C, _ = constrained
            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._record_constraint_result("pc", C, absorbed_by="runtime")
            return self

        auto_constrain = bool(self._by_state.is_constant) and (
            self._setup.null_space_dim > 0
        )
        if shared_X is None:
            result = fit_single_penalty_with_constraint_policy(
                base,
                pen,
                self._by_state,
                constraint_mode=self.constraint_mode,
                fixed=self.fixed,
                auto_constrain_when=auto_constrain,
            )
        else:
            result = fit_single_penalty_with_setup_basis(
                base,
                setup_base,
                pen,
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
            absorbed_by=(
                "runtime" if result.constraint_transform is not None else None
            ),
        )
        return self

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        if len(self.penalties) == 0:
            return []

        smooth_meta = {
            "term_type": self.term_type,
            "basis_name": self.basis_name,
            "feature": list(self.feature),
            "label": self.label,
            "by": self.by,
            "by_name": self._by_state.feature_name,
            "by_is_constant": bool(self._by_state.is_constant),
            "constraint_mode": self.constraint_mode,
            "constraint_kind": self.constraint_kind,
            "pc": self.pc,
            "knots": self.knots,
            "xt": self.xt,
            "m": self.m,
            "gp_defn": (None if self._setup is None else dict(self._setup.gp_defn)),
            "null_space_dim": (
                None if self._setup is None else self._setup.null_space_dim
            ),
            "rank": None if self._setup is None else self._setup.rank,
            "bs_dim": None if self._setup is None else self._setup.bs_dim,
            "fixed": bool(self.fixed),
        }
        selection_meta = {**smooth_meta, "is_selection_penalty": True}
        return self._build_penalty_block(
            self.penalties[0],
            smooth_metadata=smooth_meta,
            selection_metadata=selection_meta,
        )

    def transform_new(self, X_new):
        self._require_fitted()

        Xf_new = columns_as_float_matrix(X_new, self._feature_indices)

        B = predict_gp_term(Xf_new, self._setup)
        return self._apply_constraint_transform_and_by(B, X_new)

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del apply_np, x_train
        if self._setup is None:
            raise RuntimeError("Term is not fitted.")
        setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
        if centered:
            if (
                self._linked_id_setup() is not None
                and self.constraint_transform is not None
            ):
                T = np.asarray(self.constraint_transform, dtype=np.float64)
                pen = scale_penalty(
                    setup_base, np.asarray(self._setup.penalty, dtype=np.float64)
                )
                return (
                    np.asarray(setup_base @ T, dtype=np.float64),
                    np.asarray(T.T @ pen @ T, dtype=np.float64),
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
            Xf_new = columns_as_float_matrix(X_new, self._feature_indices)
            B = np.asarray(predict_gp_term(Xf_new, self._setup), dtype=np.float64)
        if np_transform is not None:
            B = B @ np.asarray(np_transform, dtype=np.float64)
        return np.asarray(B, dtype=np.float64)
