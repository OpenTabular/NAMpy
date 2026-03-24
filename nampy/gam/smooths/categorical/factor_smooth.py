from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .categorical_utils import (
    as_object_1d,
    is_factor_like_vector,
    stable_unique_levels,
    factor_indicator_matrix,
)
from ...design.structures import PenaltySpec
from ...penalties.algebra import penalty_eigendecomposition
from ..base import BaseSmoothTerm, column_as_object
from ..registry import make_smooth_term
from ...basis.tensor import rowwise_kronecker
from ..univariate.cubic_regression import SplineTerm1D
from ..univariate.pspline import PSplineTerm1D
from .random_effect import RandomEffectTerm


def _as_object_2d(X):
    X = np.asarray(X, dtype=object)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X


def _sum_to_zero_contrast(n_levels: int):
    """
    Orthonormal basis for the null space of 1^T, i.e. sum-to-zero contrasts.
    Shape: (n_levels, n_levels - 1)
    """
    n_levels = int(n_levels)
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1.")
    if n_levels == 1:
        return np.empty((1, 0), dtype=np.float64)

    c = np.ones((n_levels, 1), dtype=np.float64)
    q, _ = np.linalg.qr(c, mode="complete")
    return q[:, 1:]


@dataclass
class _BaseSmoothSpec:
    bs: str
    xt_rest: dict[str, Any] | None = None


def _parse_factor_smooth_xt(xt, default_bs="tp") -> _BaseSmoothSpec:
    """
    mgcv-style xt parsing for fs/sz:
    - None -> default base basis
    - "cr" / "tp" / ...
    - {"bs": "...", ...}
    """
    if xt is None:
        return _BaseSmoothSpec(bs=str(default_bs).lower(), xt_rest=None)

    if isinstance(xt, str):
        return _BaseSmoothSpec(bs=str(xt).lower(), xt_rest=None)

    if isinstance(xt, dict):
        bs = str(xt.get("bs", default_bs)).lower()
        rest = {k: v for k, v in xt.items() if k != "bs"}
        return _BaseSmoothSpec(bs=bs, xt_rest=(rest or None))

    raise NotImplementedError(
        "For bs='fs'/'sz', xt must be None, a basis string, or a dict containing "
        "optional key 'bs'."
    )


def _build_base_smooth_term(
    *,
    metric_features,
    k,
    base_bs,
    label,
    fixed,
    knots,
    xt_rest,
    mode,  # "fs" or "sz"
    metadata,
):
    """
    Build the per-level base smooth used inside fs/sz.

    Supported base smooth classes in the current codebase:
    cr, cs, cc, ps, tp, ts, gp
    """
    base_bs = str(base_bs).lower()
    metric_features = list(metric_features)

    if len(metric_features) == 0:
        raise ValueError("At least one metric feature is required for the base smooth.")

    if len(metric_features) > 1 and base_bs not in {"tp", "ts", "gp"}:
        raise NotImplementedError(
            f"Current {mode} implementation supports multivariate base smooths only "
            f"for bs in {{'tp','ts','gp'}}, got base bs={base_bs!r}."
        )

    if xt_rest is not None and base_bs not in {"tp", "ts", "gp"}:
        raise NotImplementedError(
            f"Extra xt options are currently only supported for tp/ts/gp base smooths, "
            f"got xt={xt_rest!r} with base bs={base_bs!r}."
        )

    if base_bs in {"cr", "cs", "cc"}:
        return SplineTerm1D(
            feature=metric_features[0],
            k=k,
            basis=base_bs,
            label=label,
            smoothing_id=None,
            by=None,
            sp=None,
            select=False,
            fixed=bool(fixed),
            constraint_mode=("never" if mode == "fs" else "auto"),
            shared_basis_setup=None,
            pc=None,
            knots=knots,
            metadata=metadata,
        )

    if base_bs == "ps":
        return PSplineTerm1D(
            feature=metric_features[0],
            k=k,
            basis="ps",
            m=None,
            label=label,
            smoothing_id=None,
            by=None,
            sp=None,
            select=False,
            fixed=bool(fixed),
            constraint_mode=("never" if mode == "fs" else "auto"),
            pc=None,
            knots=knots,
            metadata=metadata,
        )

    if base_bs in {"tp", "ts"}:
        return make_smooth_term(
            base_bs,
            feature=metric_features,
            k=k,
            basis=base_bs,
            m=None,
            label=label,
            smoothing_id=None,
            by=None,
            sp=None,
            select=False,
            fixed=bool(fixed),
            constraint_mode=("never" if mode == "fs" else "auto"),
            pc=None,
            knots=knots,
            xt=xt_rest,
            metadata=metadata,
        )

    if base_bs == "gp":
        return make_smooth_term(
            "gp",
            feature=metric_features,
            k=k,
            basis="gp",
            m=None,
            label=label,
            smoothing_id=None,
            by=None,
            sp=None,
            select=False,
            fixed=bool(fixed),
            constraint_mode=("never" if mode == "fs" else "auto"),
            pc=None,
            knots=knots,
            xt=xt_rest,
            metadata=metadata,
        )

    raise NotImplementedError(
        f"Current {mode} implementation supports base bs in "
        f"{{'cr','cs','cc','ps','tp','ts','gp'}}, got {base_bs!r}."
    )


def _kron_many(mats):
    mats = [np.asarray(M, dtype=np.float64) for M in mats]
    if len(mats) == 0:
        return np.ones((1, 1), dtype=np.float64)
    out = mats[0]
    for M in mats[1:]:
        out = np.kron(out, M)
    return out


def _block_penalty_for_group(group_index, n_groups, S_base):
    """
    Penalty acting on one group-specific copy of the base smooth.
    """
    S_base = np.asarray(S_base, dtype=np.float64)
    p0 = S_base.shape[0]
    p = n_groups * p0
    P = np.zeros((p, p), dtype=np.float64)
    sl = slice(group_index * p0, (group_index + 1) * p0)
    P[sl, sl] = S_base
    return P


class _FactorSmoothBase(BaseSmoothTerm):
    """
    Common machinery for fs/sz classes.
    """

    def __init__(
        self,
        feature,
        *,
        basis_name,
        term_type,
        k=10,
        label=None,
        smoothing_id=None,
        by=None,
        sp=None,
        xt=None,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]

        super().__init__(
            feature=features,
            label=label or f"{basis_name}({', '.join(map(str, features))})",
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        self.basis_name = basis_name
        self.term_type = term_type
        self.k = int(k)
        self.xt = xt
        self.fixed = bool(fixed)
        self.knots = knots

        self._delegate_term = None

        self._feature_indices = None
        self._feature_names = None

        self._factor_feature_indices = None
        self._factor_feature_names = None
        self._metric_feature_indices = None
        self._metric_feature_names = None

        self._base_term = None
        self._factor_levels = None
        self._basis_train = None
        self._penalties = None
        self._smoothing_ids = None
        self._ranks = None

        self.constraints_absorbed = True
        self.fit_constraint_matrix = None
        self.predict_constraint_matrix = None

    @property
    def basis_train(self):
        if self._delegate_term is not None:
            return self._delegate_term.basis_train
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._basis_train

    @property
    def penalties(self):
        if self._delegate_term is not None:
            return self._delegate_term.penalties
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._penalties

    @property
    def n_coef(self):
        if self._delegate_term is not None:
            return self._delegate_term.n_coef
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return int(self._basis_train.shape[1])

    def transform_new(self, X_new):
        if self._delegate_term is not None:
            return self._delegate_term.transform_new(X_new)
        raise NotImplementedError

    def get_penalty_definitions(self):
        if self._delegate_term is not None:
            return self._delegate_term.get_penalty_definitions()
        raise NotImplementedError

    def _resolve_features(self, X, feature_names):
        X = _as_object_2d(X)
        feature_indices = []
        feature_names_resolved = []

        for feat in self.feature:
            idx = None
            fname = None
            if isinstance(feat, int):
                idx = int(feat)
                if idx < 0 or idx >= len(feature_names):
                    raise IndexError(
                        f"Feature index {idx} out of range for {len(feature_names)} features."
                    )
                fname = feature_names[idx]
            else:
                fname = str(feat)
                if fname not in feature_names:
                    raise KeyError(
                        f"Feature {fname!r} not found in feature_names={feature_names}."
                    )
                idx = feature_names.index(fname)

            feature_indices.append(idx)
            feature_names_resolved.append(fname)

        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)

        factor_feature_indices = []
        factor_feature_names = []
        metric_feature_indices = []
        metric_feature_names = []

        for idx, fname in zip(feature_indices, feature_names_resolved):
            col = X[:, idx]
            if is_factor_like_vector(col):
                factor_feature_indices.append(idx)
                factor_feature_names.append(fname)
            else:
                metric_feature_indices.append(idx)
                metric_feature_names.append(fname)

        self._factor_feature_indices = factor_feature_indices
        self._factor_feature_names = factor_feature_names
        self._metric_feature_indices = metric_feature_indices
        self._metric_feature_names = metric_feature_names

        return X

    def _build_delegate_base_or_re(self, X, feature_names, *, default_bs, mode):
        """
        mgcv source behavior:
        - no factor term -> just use the base smooth constructor
        - factor-only term -> transfer to re class
        """
        X = self._resolve_features(X, feature_names)

        if self.by is not None:
            raise NotImplementedError(
                f"{self.basis_name} terms do not currently support an additional by= variable."
            )

        if len(self._factor_feature_indices) == 0:
            base_spec = _parse_factor_smooth_xt(self.xt, default_bs=default_bs)
            base_term = _build_base_smooth_term(
                metric_features=self._metric_feature_names,
                k=self.k,
                base_bs=base_spec.bs,
                label=self.label,
                fixed=self.fixed,
                knots=self.knots,
                xt_rest=base_spec.xt_rest,
                mode=mode,
                metadata=dict(self.metadata),
            )
            base_term.fit(X, feature_names)
            self._delegate_term = base_term
            self.term_type = base_term.term_type
            self.basis_name = base_term.basis_name
            self._record_constraint_result(
                getattr(base_term, "constraint_kind", None),
                getattr(base_term, "constraint_transform", None),
                absorbed_by=getattr(base_term, "constraints_absorbed_by", None),
            )
            return True

        if len(self._metric_feature_indices) == 0:
            re_term = RandomEffectTerm(
                feature=self._factor_feature_names,
                label=self.label,
                smoothing_id=self.smoothing_id,
                by=None,
                sp=self.sp,
                xt=self.xt,
                metadata=dict(self.metadata),
            )
            re_term.fit(X, feature_names)
            self._delegate_term = re_term
            self.term_type = re_term.term_type
            self.basis_name = re_term.basis_name
            self._record_constraint_result(None, None, absorbed_by=None)
            return True

        return False


class FSmoothInteractionTerm(_FactorSmoothBase):
    """
    mgcv-like bs="fs":
    - exactly one factor variable
    - base smooth per factor level
    - fully penalized, no centering
    - same penalty structure shared across levels
    """
    def __init__(
        self,
        feature,
        k=10,
        label=None,
        smoothing_id=None,
        by=None,
        sp=None,
        xt=None,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        if smoothing_id is not None:
            raise NotImplementedError(
                "Current bs='fs' implementation does not support id/linkage across terms."
            )

        super().__init__(
            feature=feature,
            basis_name="fs",
            term_type="factor_smooth_fs",
            k=k,
            label=label,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            xt=xt,
            fixed=fixed,
            knots=knots,
            metadata=metadata,
        )

        self._levels = None
        self._base_transform = None
        self._range_rank = None
        self._null_dim = None

    def fit(self, X, feature_names):
        if self._build_delegate_base_or_re(X, feature_names, default_bs="tp", mode="fs"):
            return self

        if len(self._factor_feature_indices) != 1:
            raise NotImplementedError("bs='fs' requires exactly one factor variable.")

        X = _as_object_2d(X)

        base_spec = _parse_factor_smooth_xt(self.xt, default_bs="tp")
        base_term = _build_base_smooth_term(
            metric_features=self._metric_feature_names,
            k=self.k,
            base_bs=base_spec.bs,
            label=self.label,
            fixed=self.fixed,
            knots=self.knots,
            xt_rest=base_spec.xt_rest,
            mode="fs",
            metadata=dict(self.metadata),
        )
        base_term.fit(X, feature_names)

        if len(base_term.penalties) > 1:
            raise NotImplementedError(
                'bs="fs" currently requires a singly penalized base smooth.'
            )

        B0 = np.asarray(base_term.basis_train, dtype=np.float64)
        self._base_term = base_term

        fac_idx = self._factor_feature_indices[0]
        fac_name = self._factor_feature_names[0]
        fac = as_object_1d(X[:, fac_idx])
        levels = stable_unique_levels(fac)
        self._levels = levels

        Ifac = factor_indicator_matrix(fac, levels)
        n_levels = len(levels)

        if len(base_term.penalties) == 0:
            X_full = rowwise_kronecker([Ifac, B0])
            self._basis_train = np.asarray(X_full, dtype=np.float64)
            self._penalties = []
            self._smoothing_ids = []
            self._ranks = []
            self.constraints_absorbed = True
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        S0 = np.asarray(base_term.penalties[0], dtype=np.float64)
        dec = penalty_eigendecomposition(S0, tol=1e-10)

        U1 = dec["U1"]
        U0 = dec["U0"]
        d_pos = dec["d_pos"]

        r = U1.shape[1]
        q = U0.shape[1]
        self._range_rank = r
        self._null_dim = q

        T = np.column_stack([U1, U0]) if q > 0 else U1
        self._base_transform = T

        B_sep = B0 @ T
        X_full = rowwise_kronecker([Ifac, B_sep])

        p_sep = B_sep.shape[1]
        S_range = np.zeros((p_sep, p_sep), dtype=np.float64)
        if r > 0:
            S_range[:r, :r] = np.diag(d_pos)

        penalties = []
        smoothing_ids = []
        ranks = []

        P_range = np.kron(np.eye(n_levels, dtype=np.float64), S_range)
        penalties.append(0.5 * (P_range + P_range.T))
        smoothing_ids.append(f"__fs__:{self.label}:range")
        ranks.append(int(n_levels * r))

        for i in range(q):
            v = np.zeros((p_sep,), dtype=np.float64)
            v[r + i] = 1.0
            S_null_i = np.outer(v, v)
            P_null_i = np.kron(np.eye(n_levels, dtype=np.float64), S_null_i)
            penalties.append(0.5 * (P_null_i + P_null_i.T))
            smoothing_ids.append(f"__fs__:{self.label}:null:{i}")
            ranks.append(int(n_levels))

        self._basis_train = np.asarray(X_full, dtype=np.float64)
        self._penalties = penalties
        self._smoothing_ids = smoothing_ids
        self._ranks = ranks
        self.constraints_absorbed = True
        self._record_constraint_result(None, None, absorbed_by=None)
        return self

    def transform_new(self, X_new):
        if self._delegate_term is not None:
            return self._delegate_term.transform_new(X_new)

        if self._base_term is None or self._levels is None:
            raise RuntimeError("Term is not fitted.")

        fac_idx = self._factor_feature_indices[0]
        fac = as_object_1d(column_as_object(X_new, fac_idx))
        Ifac = factor_indicator_matrix(fac, self._levels)

        B0_new = np.asarray(self._base_term.transform_new(X_new), dtype=np.float64)
        if self._base_transform is not None:
            B0_new = B0_new @ self._base_transform

        return np.asarray(rowwise_kronecker([Ifac, B0_new]), dtype=np.float64)

    def get_penalty_definitions(self):
        if self._delegate_term is not None:
            return self._delegate_term.get_penalty_definitions()

        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")

        if len(self._penalties) == 0:
            return []

        sp_vals = self._normalized_term_sp(len(self._penalties))
        defs = []

        for j, P in enumerate(self._penalties):
            sp_j = sp_vals[j] if j < len(sp_vals) else None
            if sp_j is None:
                sp_mode = None
                sp_value = None
            elif sp_j >= 0:
                sp_mode = "fixed"
                sp_value = float(sp_j)
            else:
                sp_mode = "estimate"
                sp_value = None

            defs.append(
                PenaltySpec(
                    matrix=np.asarray(P, dtype=np.float64),
                    smoothing_id=self._smoothing_ids[j],
                    kind=("smooth" if j == 0 else "null_space"),
                    rank=self._ranks[j],
                    null_space_dim=(0 if j == 0 else 1),
                    is_null_space_penalty=(j > 0),
                    sp_mode=sp_mode,
                    sp_value=sp_value,
                    metadata={
                        "term_type": self.term_type,
                        "basis_name": self.basis_name,
                        "feature": list(self.feature),
                        "label": self.label,
                        "factor_name": self._factor_feature_names[0],
                        "levels": list(self._levels),
                        "base_basis_name": self._base_term.basis_name if self._base_term is not None else None,
                        "base_metric_features": list(self._metric_feature_names),
                    },
                )
            )

        return defs


class SZSmoothInteractionTerm(_FactorSmoothBase):
    """
    mgcv-like bs="sz":
    - one or more factor variables
    - one base smooth duplicated over factor combinations
    - coefficientwise sum-to-zero contrasts across factor margins
    - if smoothing_id is None: one penalty per factor combination
    - if smoothing_id is set: penalties summed to share one smoothing parameter
    """
    def __init__(
        self,
        feature,
        k=10,
        label=None,
        smoothing_id=None,
        by=None,
        sp=None,
        xt=None,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        super().__init__(
            feature=feature,
            basis_name="sz",
            term_type="factor_smooth_sz",
            k=k,
            label=label,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            xt=xt,
            fixed=fixed,
            knots=knots,
            metadata=metadata,
        )

        self._contrast_mats = None
        self._factor_transform = None
        self._n_groups = None

    def fit(self, X, feature_names):
        if self._build_delegate_base_or_re(X, feature_names, default_bs="tp", mode="sz"):
            return self

        X = _as_object_2d(X)

        base_spec = _parse_factor_smooth_xt(self.xt, default_bs="tp")
        base_term = _build_base_smooth_term(
            metric_features=self._metric_feature_names,
            k=self.k,
            base_bs=base_spec.bs,
            label=self.label,
            fixed=self.fixed,
            knots=self.knots,
            xt_rest=base_spec.xt_rest,
            mode="sz",
            metadata=dict(self.metadata),
        )
        base_term.fit(X, feature_names)

        if len(base_term.penalties) > 1:
            raise NotImplementedError(
                'bs="sz" currently requires a singly penalized base smooth.'
            )

        B0 = np.asarray(base_term.basis_train, dtype=np.float64)
        self._base_term = base_term

        level_lists = []
        indicator_mats = []
        contrast_mats = []

        for idx in self._factor_feature_indices:
            fac = as_object_1d(X[:, idx])
            lev = stable_unique_levels(fac)
            level_lists.append(lev)
            indicator_mats.append(factor_indicator_matrix(fac, lev))
            contrast_mats.append(_sum_to_zero_contrast(len(lev)))

        self._factor_levels = level_lists
        self._contrast_mats = contrast_mats

        X_raw = rowwise_kronecker(indicator_mats + [B0])

        p0 = B0.shape[1]
        C_fac = _kron_many(contrast_mats)
        self._factor_transform = C_fac
        self._n_groups = int(np.prod([len(lev) for lev in level_lists], dtype=np.int64))

        if C_fac.shape[1] == 0:
            self._basis_train = np.empty((X_raw.shape[0], 0), dtype=np.float64)
            self._penalties = []
            self._smoothing_ids = []
            self._ranks = []
            self.constraints_absorbed = True
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        T = np.kron(C_fac, np.eye(p0, dtype=np.float64))
        X_con = X_raw @ T

        if len(base_term.penalties) == 0:
            self._basis_train = np.asarray(X_con, dtype=np.float64)
            self._penalties = []
            self._smoothing_ids = []
            self._ranks = []
            self.constraints_absorbed = True
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        S0 = np.asarray(base_term.penalties[0], dtype=np.float64)
        penalties = []
        smoothing_ids = []
        ranks = []

        if self.smoothing_id is None:
            for g in range(self._n_groups):
                P_raw = _block_penalty_for_group(g, self._n_groups, S0)
                P_con = T.T @ P_raw @ T
                P_con = 0.5 * (P_con + P_con.T)

                penalties.append(P_con)
                smoothing_ids.append(f"__sz__:{self.label}:group:{g}")
                ranks.append(int(np.linalg.matrix_rank(P_con)))
        else:
            P_sum = np.zeros((self._n_groups * p0, self._n_groups * p0), dtype=np.float64)
            for g in range(self._n_groups):
                P_sum += _block_penalty_for_group(g, self._n_groups, S0)
            P_con = T.T @ P_sum @ T
            P_con = 0.5 * (P_con + P_con.T)

            penalties.append(P_con)
            smoothing_ids.append(str(self.smoothing_id))
            ranks.append(int(np.linalg.matrix_rank(P_con)))

        self._basis_train = np.asarray(X_con, dtype=np.float64)
        self._penalties = penalties
        self._smoothing_ids = smoothing_ids
        self._ranks = ranks
        self.constraints_absorbed = True
        self._record_constraint_result(None, None, absorbed_by=None)
        return self

    def transform_new(self, X_new):
        if self._delegate_term is not None:
            return self._delegate_term.transform_new(X_new)

        if self._base_term is None or self._factor_levels is None or self._factor_transform is None:
            raise RuntimeError("Term is not fitted.")

        indicator_mats = []
        for idx, lev in zip(self._factor_feature_indices, self._factor_levels):
            fac = as_object_1d(column_as_object(X_new, idx))
            indicator_mats.append(factor_indicator_matrix(fac, lev))

        B0_new = np.asarray(self._base_term.transform_new(X_new), dtype=np.float64)
        X_raw = rowwise_kronecker(indicator_mats + [B0_new])

        if self._factor_transform.shape[1] == 0:
            return np.empty((X_raw.shape[0], 0), dtype=np.float64)

        T = np.kron(self._factor_transform, np.eye(B0_new.shape[1], dtype=np.float64))
        return np.asarray(X_raw @ T, dtype=np.float64)

    def get_penalty_definitions(self):
        if self._delegate_term is not None:
            return self._delegate_term.get_penalty_definitions()

        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")

        if len(self._penalties) == 0:
            return []

        sp_vals = self._normalized_term_sp(len(self._penalties))
        defs = []

        for j, P in enumerate(self._penalties):
            sp_j = sp_vals[j] if j < len(sp_vals) else None
            if sp_j is None:
                sp_mode = None
                sp_value = None
            elif sp_j >= 0:
                sp_mode = "fixed"
                sp_value = float(sp_j)
            else:
                sp_mode = "estimate"
                sp_value = None

            defs.append(
                PenaltySpec(
                    matrix=np.asarray(P, dtype=np.float64),
                    smoothing_id=self._smoothing_ids[j],
                    kind="smooth",
                    rank=self._ranks[j],
                    null_space_dim=None,
                    is_null_space_penalty=False,
                    sp_mode=sp_mode,
                    sp_value=sp_value,
                    metadata={
                        "term_type": self.term_type,
                        "basis_name": self.basis_name,
                        "feature": list(self.feature),
                        "label": self.label,
                        "factor_names": list(self._factor_feature_names),
                        "factor_levels": [list(lev) for lev in self._factor_levels],
                        "base_basis_name": self._base_term.basis_name if self._base_term is not None else None,
                        "base_metric_features": list(self._metric_feature_names),
                        "shared_smoothing_id": self.smoothing_id,
                    },
                )
            )

        return defs
