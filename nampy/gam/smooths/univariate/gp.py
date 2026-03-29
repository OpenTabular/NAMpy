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

from ..base import (
    BaseSmoothTerm,
    _normalize_point_constraint,
    _resolve_feature,
    columns_as_float_matrix,
    resolve_by_state,
    sync_by_state_attributes,
)
from ..registry import register_smooth
from ...constraints.absorption import (
    apply_linear_constraint,
    fit_single_penalty_with_constraint_policy,
)
from ...design.structures import PenaltySpec
from ...penalties.algebra import null_space_penalty_from_penalty
from ....splines.penalty_scaling import scale_penalty
from ....splines.gaussian_process import build_gp_term_setup, predict_gp_term


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

        self._by_state = resolve_by_state(self.by, X, feature_names)
        sync_by_state_attributes(self, self._by_state)

        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)

        self._setup = build_gp_term_setup(
            Xf,
            k=self.k,
            m=self.m,
            knots=self.knots,
            xt=self.xt,
        )

        base = np.asarray(self._setup.basis_train, dtype=np.float64)
        pen = scale_penalty(
            base,
            np.asarray(self._setup.penalty, dtype=np.float64),
        )

        if self.pc is not None:
            if len(self._feature_indices) > 1:
                raise NotImplementedError(
                    "pc= for multivariate gp smooths is not yet implemented."
                )
            pc_value = _normalize_point_constraint(self.pc, self._feature_names[0])
            pc_point = np.asarray([[pc_value]], dtype=np.float64)
            pc_basis = predict_gp_term(pc_point, self._setup)[0]
            penalties_in = [] if self.fixed else [pen]
            Bc, Sc, C = apply_linear_constraint(base, penalties_in, pc_basis)
            if self._by_state.is_present:
                Bc = Bc * self._by_state.values[:, None]
            self._basis_train = np.asarray(Bc, dtype=np.float64)
            self._penalties = Sc
            self._record_constraint_result("pc", C, absorbed_by="runtime")
            return self

        auto_constrain = bool(self._by_state.is_constant) and (self._setup.null_space_dim > 0)
        result = fit_single_penalty_with_constraint_policy(
            base, pen, self._by_state,
            constraint_mode=self.constraint_mode,
            fixed=self.fixed,
            auto_constrain_when=auto_constrain,
        )
        self._basis_train = result.basis_train
        self._penalties = result.penalties
        self._record_constraint_result(
            result.constraint_kind,
            result.constraint_transform,
            absorbed_by=("runtime" if result.constraint_transform is not None else None),
        )
        return self

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        if len(self.penalties) == 0:
            return []

        if self.select and self.sp is not None:
            raise NotImplementedError(
                "term-level sp is not yet implemented for select=True smooths in the "
                "current runtime, because select=True adds an extra explicit "
                "null-space penalty in addition to the main penalty."
            )

        main_penalty = np.asarray(self.penalties[0], dtype=np.float64)
        sp_vals = self._normalized_term_sp(1)
        sp_main = sp_vals[0] if sp_vals else None

        if sp_main is None:
            sp_mode = None
            sp_value = None
        elif sp_main >= 0:
            sp_mode = "fixed"
            sp_value = float(sp_main)
        else:
            sp_mode = "estimate"
            sp_value = None

        defs = [
            PenaltySpec(
                matrix=main_penalty,
                smoothing_id=(None if self.smoothing_id is None else str(self.smoothing_id)),
                kind="smooth",
                sp_mode=sp_mode,
                sp_value=sp_value,
                metadata={
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
                    "gp_defn": None if self._setup is None else dict(self._setup.gp_defn),
                    "null_space_dim": None if self._setup is None else self._setup.null_space_dim,
                    "rank": None if self._setup is None else self._setup.rank,
                    "bs_dim": None if self._setup is None else self._setup.bs_dim,
                    "fixed": bool(self.fixed),
                    "term_sp": sp_main,
                    "is_selection_penalty": False,
                },
            )
        ]

        if self.select:
            S0, meta = null_space_penalty_from_penalty(
                main_penalty,
                tol=self.null_penalty_tol,
            )
            if meta["rank"] > 0:
                defs.append(
                    PenaltySpec(
                        matrix=S0,
                        smoothing_id=(
                            None
                            if self.smoothing_id is None
                            else f"{self.smoothing_id}::select"
                        ),
                        kind="null_space",
                        rank=meta["rank"],
                        null_space_dim=meta["null_space_dim"],
                        is_null_space_penalty=True,
                        sp_mode=None,
                        sp_value=None,
                        metadata={
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
                            "gp_defn": None if self._setup is None else dict(self._setup.gp_defn),
                            "null_space_dim": None if self._setup is None else self._setup.null_space_dim,
                            "rank": None if self._setup is None else self._setup.rank,
                            "bs_dim": None if self._setup is None else self._setup.bs_dim,
                            "fixed": bool(self.fixed),
                            "is_selection_penalty": True,
                        },
                    )
                )

        return defs

    def transform_new(self, X_new):
        if self._feature_indices is None or self._setup is None:
            raise RuntimeError("Term is not fitted.")

        Xf_new = columns_as_float_matrix(X_new, self._feature_indices)

        B = predict_gp_term(Xf_new, self._setup)
        return self._apply_constraint_transform_and_by(B, X_new)
