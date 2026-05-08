"""
Thin plate regression spline smooth term (``bs='tp'`` / ``bs='ts'``).

Implements the :class:`BaseSmoothTerm` interface for a rank-reduced thin plate
spline basis.  Thin plate splines are rotation-invariant and automatically
extend to multi-variate smooths, making them the default smooth type.

The ``'ts'`` variant uses a shrinkage version of the main thin-plate penalty,
making the penalty full-rank so the term can shrink entirely to zero.
Separately, ``select=True`` adds an explicit null-space selection penalty on
top of the main penalty.
"""

import numpy as np

from ....splines.univariate.tp import (
    build_tprs_term_setup,
    predict_tprs_term,
)
from ...constraints.absorption import (
    fit_single_penalty_with_constraint_policy,
    fit_single_penalty_with_setup_basis,
)
from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _resolve_feature,
    columns_as_float_matrix,
)


@register_smooth("tp")
@register_smooth("ts")
class ThinPlateSplineTerm(BaseSmoothTerm):
    term_type = "smooth"
    basis_name = "tp"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=-1,
        basis="tp",
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

        if self.basis_name not in {"tp", "ts"}:
            raise NotImplementedError(
                f"ThinPlateSplineTerm currently supports only basis in "
                f"{{'tp','ts'}}, got {basis!r}."
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
            self._setup = build_tprs_term_setup(
                Xf_setup,
                basis=self.basis_name,
                k=self.k,
                m=self.m,
                knots=self.knots,
                xt=self.xt,
            )
            setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
            base = np.asarray(predict_tprs_term(Xf, self._setup), dtype=np.float64)
            pen = scale_penalty(
                setup_base,
                np.asarray(self._setup.penalty, dtype=np.float64),
            )
        else:
            self._setup = build_tprs_term_setup(
                Xf,
                basis=self.basis_name,
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
                [pen],
                self.pc,
                feature_names=self._feature_names,
                point_basis_fn=lambda pts: predict_tprs_term(pts, self._setup)[0],
                fixed=self.fixed,
            )
            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._record_constraint_result("pc", C, absorbed_by="runtime")
            return self

        auto_constrain = (
            self.basis_name == "tp"
            and not self._setup.drop_null_effective
            and bool(self._by_state.is_constant)
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
            "penalty_order": (
                None if self._setup is None else self._setup.penalty_order
            ),
            "original_null_space_dim": (
                None if self._setup is None else self._setup.original_null_space_dim
            ),
            "drop_null_requested": (
                None if self._setup is None else bool(self._setup.drop_null_requested)
            ),
            "drop_null_effective": (
                None if self._setup is None else bool(self._setup.drop_null_effective)
            ),
            "fixed": bool(self.fixed),
        }
        selection_meta = {**smooth_meta, "is_selection_penalty": True}
        smooth_meta = self._penalty_metadata_with_scale(smooth_meta, penalty_index=0)
        rank = int(self._setup.rank) if self._setup is not None else None
        return self._build_penalty_block(
            self.penalties[0],
            rank=rank,
            smooth_metadata=smooth_meta,
            selection_metadata=selection_meta,
        )

    def transform_new(self, X_new):
        self._require_fitted()

        Xf_new = columns_as_float_matrix(X_new, self._feature_indices)

        B = predict_tprs_term(Xf_new, self._setup)
        return self._apply_constraint_transform_and_by(B, X_new)

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del apply_np, x_train
        if self._setup is None:
            raise RuntimeError("Term is not fitted.")
        setup_base = np.asarray(self._setup.basis_train, dtype=np.float64)
        setup_penalty = np.asarray(self._setup.penalty, dtype=np.float64)
        if centered:
            if (
                self._linked_id_setup() is not None
                and self.constraint_transform is not None
            ):
                T = np.asarray(self.constraint_transform, dtype=np.float64)
                pen = scale_penalty(setup_base, setup_penalty)
                return (
                    np.asarray(setup_base @ T, dtype=np.float64),
                    np.asarray(T.T @ pen @ T, dtype=np.float64),
                    None,
                )
            return super().tensor_marginal_fit_matrices(centered=centered)
        return (
            setup_base,
            setup_penalty,
            None,
        )

    def tensor_marginal_predict_matrix(
        self, X_new, *, centered=False, np_transform=None
    ):
        if centered:
            B = np.asarray(self.transform_new(X_new), dtype=np.float64)
        else:
            Xf_new = columns_as_float_matrix(X_new, self._feature_indices)
            B = np.asarray(predict_tprs_term(Xf_new, self._setup), dtype=np.float64)
        if np_transform is not None:
            B = B @ np.asarray(np_transform, dtype=np.float64)
        return np.asarray(B, dtype=np.float64)
