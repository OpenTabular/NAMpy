from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import orthogonal_procrustes

from ....splines.basis.natparam import nat_param_type1
from ..._mgcv_constants import EIG_TOL_POWER
from ...compiler.structures import PenaltySpec
from ...penalties import penalty_id_for_local_index, rescale_tensor_penalties_for_fit
from ..algebra import rowwise_kronecker
from ..registry import make_smooth_term
from ..smooth_base import BaseSmoothTerm, by_values_from_new_data, column_as_object
from ..univariate.cr import CubicSplineTerm
from ..univariate.ps import PSplineTerm1D
from .categorical_utils import (
    as_object_1d,
    factor_indicator_matrix,
    factor_levels_from_metadata,
    is_factor_like_vector,
    stable_unique_levels,
)
from .re import RandomEffectTerm


def _as_object_2d(X):
    X = np.asarray(X, dtype=object)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X


def _sum_to_zero_contrast(n_levels: int):
    """
    Last-level-as-reference contrast matrix matching mgcv's XZKr().
    Shape: (n_levels, n_levels - 1)

    Each column subtracts the last level from one of the first (n_levels-1)
    levels, i.e. C = vstack([I_{m-1}, -ones^T]).
    """
    n_levels = int(n_levels)
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1.")
    if n_levels == 1:
        return np.empty((1, 0), dtype=np.float64)

    C = np.vstack(
        [
            np.eye(n_levels - 1, dtype=np.float64),
            -np.ones((1, n_levels - 1), dtype=np.float64),
        ]
    )
    return C


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
    by,
    knots,
    xt_rest,
    mode,  # "fs" or "sz"
    select,
    constraint_mode,
    metadata,
):
    """
    Build the per-level base smooth used inside fs/sz.

    Supported base smooth classes in the current codebase:
    cr, cs, cc, ps, tp, ts
    """
    base_bs = str(base_bs).lower()
    metric_features = list(metric_features)

    if len(metric_features) == 0:
        raise ValueError("At least one metric feature is required for the base smooth.")

    if mode == "fs" and base_bs in {"cs", "ts"}:
        raise NotImplementedError(_fs_full_rank_base_error(base_bs))

    if len(metric_features) > 1 and base_bs not in {"tp", "ts"}:
        raise NotImplementedError(
            f"Current {mode} implementation supports multivariate base smooths only "
            f"for bs in {{'tp','ts'}}, got base bs={base_bs!r}."
        )

    if xt_rest is not None and base_bs not in {"tp", "ts", "ps"}:
        raise NotImplementedError(
            f"Extra xt options are currently only supported for tp/ts/ps base smooths, "
            f"got xt={xt_rest!r} with base bs={base_bs!r}."
        )

    if base_bs in {"cr", "cs", "cc"}:
        return CubicSplineTerm(
            feature=metric_features[0],
            k=k,
            basis=base_bs,
            label=label,
            smoothing_id=None,
            by=by,
            sp=None,
            select=bool(select),
            fixed=bool(fixed),
            constraint_mode=str(constraint_mode),
            shared_basis_setup=None,
            pc=None,
            knots=knots,
            metadata=metadata,
        )

    if base_bs == "ps":
        ps_m = None if xt_rest is None else xt_rest.get("m", None)
        # For fs/sz, mgcv keeps the outer basis dimension and uses xt mainly to
        # choose the base smoother family / order parameters.
        ps_k = k
        return PSplineTerm1D(
            feature=metric_features[0],
            k=ps_k,
            basis="ps",
            m=ps_m,
            label=label,
            smoothing_id=None,
            by=by,
            sp=None,
            select=bool(select),
            fixed=bool(fixed),
            constraint_mode=str(constraint_mode),
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
            by=by,
            sp=None,
            select=bool(select),
            fixed=bool(fixed),
            constraint_mode=str(constraint_mode),
            pc=None,
            knots=knots,
            xt=xt_rest,
            metadata=metadata,
        )

    raise NotImplementedError(
        f"Current {mode} implementation supports base bs in "
        f"{{'cr','cs','cc','ps','tp','ts'}}, got {base_bs!r}."
    )


def _penalty_rank_from_base_term(base_term, basis_matrix, penalty_matrix) -> int:
    if isinstance(base_term, PSplineTerm1D) and len(base_term.penalties) > 0:
        # mgcv::smooth.construct.ps.smooth.spec uses rank <- bs.dim - m[2].
        penalty_order = int(base_term.m[1])
        return max(0, int(basis_matrix.shape[1]) - penalty_order)

    base_rank = int(getattr(base_term, "rank", 0) or 0)
    if base_rank > 0:
        return base_rank

    evals = np.linalg.eigvalsh(0.5 * (penalty_matrix + penalty_matrix.T))
    tol = (np.max(evals) if evals.size else 0.0) * (
        np.finfo(np.float64).eps ** EIG_TOL_POWER
    )
    return int(np.sum(evals > tol))


def _fs_full_rank_base_error(base_bs: str) -> str:
    return (
        f'`bs="fs"` with base bs={base_bs!r} is unsupported. '
        "Upstream mgcv::smooth.construct.fs.smooth.spec also rejects this "
        "full-rank shrinkage base because it leaves no null space for the "
        "factor-smooth null penalties."
    )


def _kron_many(mats):
    mats = [np.asarray(M, dtype=np.float64) for M in mats]
    if len(mats) == 0:
        return np.ones((1, 1), dtype=np.float64)
    out = mats[0]
    for M in mats[1:]:
        out = np.kron(out, M)
    return out


def _sum_to_zero_contrast_transform(level_sizes, block_dim):
    """
    Exact mgcv::XZKr() contrast transform.

    Returns ``T`` such that ``X_raw @ T`` matches
    ``t(mgcv:::XZKr(X_raw, m))``. Implemented by running the upstream reshape /
    subtraction logic on an identity basis, to preserve column order exactly.
    """
    level_sizes = [int(v) for v in level_sizes]
    block_dim = int(block_dim)
    if block_dim < 0:
        raise ValueError("block_dim must be non-negative.")
    if any(v < 1 for v in level_sizes):
        raise ValueError("level_sizes must all be >= 1.")

    q_in = int(np.prod(level_sizes, dtype=np.int64)) * block_dim
    q_out = int(np.prod([v - 1 for v in level_sizes], dtype=np.int64)) * block_dim
    if q_out == 0:
        return np.empty((q_in, 0), dtype=np.float64)

    work = np.eye(q_in, dtype=np.float64)
    n = int(work.shape[0])
    for m_i in level_sizes:
        work = np.reshape(work, (work.size // m_i, m_i), order="F")
        work = (work[:, : m_i - 1] - work[:, [m_i - 1]]).T
    work = np.reshape(work, (work.size // block_dim, block_dim), order="F")
    work = work.T
    work = np.reshape(work, (work.size // n, n), order="F")
    return np.asarray(work.T, dtype=np.float64)


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
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        xt=None,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]

        super().__init__(
            feature=features,
            label=label or f"{basis_name}({', '.join(map(str, features))})",
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        self.basis_name = basis_name
        self.term_type = term_type
        self.k = int(k)
        self.select = bool(select)
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

        self.skip_centering = True

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
        self._require_fitted()
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

        for idx, fname in zip(feature_indices, feature_names_resolved, strict=True):
            col = X[:, idx]
            if factor_levels_from_metadata(self.metadata, fname) is not None:
                factor_feature_indices.append(idx)
                factor_feature_names.append(fname)
            elif is_factor_like_vector(col):
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

        if len(self._factor_feature_indices) == 0:
            base_spec = _parse_factor_smooth_xt(self.xt, default_bs=default_bs)
            base_term = _build_base_smooth_term(
                metric_features=self._metric_feature_names,
                k=self.k,
                base_bs=base_spec.bs,
                label=self.label,
                fixed=self.fixed,
                by=self.by,
                knots=self.knots,
                xt_rest=base_spec.xt_rest,
                mode=mode,
                select=self.select,
                constraint_mode=("auto" if mode == "fs" else "never"),
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
                by=self.by,
                sp=self.sp,
                select=self.select,
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

    def _base_constructor_fit_matrices(self):
        if self._base_term is None:
            raise RuntimeError("Base smooth term is not fitted.")
        return self._base_term.tensor_marginal_fit_matrices(centered=False)

    def _base_constructor_predict_matrix(self, X_new):
        if self._base_term is None:
            raise RuntimeError("Base smooth term is not fitted.")
        return self._base_term.tensor_marginal_predict_matrix(X_new, centered=False)


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
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        xt=None,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        super().__init__(
            feature=feature,
            basis_name="fs",
            term_type="factor_smooth_fs",
            k=k,
            label=label,
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            select=select,
            xt=xt,
            fixed=fixed,
            knots=knots,
            metadata=metadata,
        )

        self._levels = None
        self._base_transform = None
        self._base_range_penalty_diag = None
        self._range_rank = None
        self._null_dim = None

    def _align_multivariate_base_reparameterization(
        self,
        X,
        X_reparam,
        P_coef,
        *,
        range_rank,
        null_dim,
    ):
        base_term = self._base_term
        basis_key = str(getattr(base_term, "basis_name", "")).lower()
        if null_dim <= 1:
            return X_reparam, P_coef

        X_reparam = np.asarray(X_reparam, dtype=np.float64).copy()
        P_coef = np.asarray(P_coef, dtype=np.float64).copy()

        if basis_key == "tp":
            n_metric = len(self._metric_feature_names)
            metric = np.column_stack(
                [
                    np.asarray(X[:, idx], dtype=np.float64)
                    for idx in base_term.resolved_feature_indices()
                ]
            )
            metric = metric - metric.mean(axis=0, keepdims=True)

            # 1D tp bases used by `bs="fs"` retain a 2D null block spanning the
            # centred linear function and the constant. For >=4 factor levels the
            # repeated-zero eigenspace in `nat.param(type=1)` can land in the
            # opposite linear/constant order from mgcv. Align that 2D block to the
            # corresponding model-space span before duplicating by level so the
            # later smoothCon scaling sees the same orientation.
            if n_metric == 1 and null_dim == 2 and len(self._levels or []) >= 4:
                X_null = X_reparam[:, range_rank:].copy()
                centered_norm = np.linalg.norm(
                    X_null - X_null.mean(axis=0, keepdims=True),
                    axis=0,
                )
                if float(centered_norm[0]) > float(centered_norm[1]):
                    target = np.column_stack(
                        [metric[:, 0], np.ones(metric.shape[0], dtype=np.float64)]
                    )
                else:
                    target = np.column_stack(
                        [np.ones(metric.shape[0], dtype=np.float64), metric[:, 0]]
                    )
                R, _ = orthogonal_procrustes(X_null, target)
                X_reparam[:, range_rank:] = X_null @ R
                P_coef[:, range_rank:] = P_coef[:, range_rank:] @ R
                return X_reparam, P_coef

            if n_metric > 1:
                target = np.column_stack(
                    [metric, np.ones(metric.shape[0], dtype=np.float64)]
                )
                if target.shape[1] != null_dim:
                    return X_reparam, P_coef
                R, _ = orthogonal_procrustes(X_reparam[:, range_rank:], target)
                X_reparam[:, range_rank:] = X_reparam[:, range_rank:] @ R
                P_coef[:, range_rank:] = P_coef[:, range_rank:] @ R
                return X_reparam, P_coef

        return X_reparam, P_coef

    def fit(self, X, feature_names):
        if self._build_delegate_base_or_re(
            X, feature_names, default_bs="tp", mode="fs"
        ):
            return self

        if len(self._factor_feature_indices) != 1:
            raise NotImplementedError("bs='fs' requires exactly one factor variable.")

        X = _as_object_2d(X)
        self._set_by_state(X, feature_names)

        base_spec = _parse_factor_smooth_xt(self.xt, default_bs="tp")
        # mgcv::smooth.construct.fs.smooth.spec calls smooth.construct on the
        # marginal spec without extra side constraints before duplicating by
        # factor level, so the base smooth must retain its full null space.
        base_constraint = "never"
        base_term = _build_base_smooth_term(
            metric_features=self._metric_feature_names,
            k=self.k,
            base_bs=base_spec.bs,
            label=self.label,
            fixed=self.fixed,
            by=None,
            knots=self.knots,
            xt_rest=base_spec.xt_rest,
            mode="fs",
            select=False,
            constraint_mode=base_constraint,
            metadata=dict(self.metadata),
        )
        base_term.fit(X, feature_names)

        if len(base_term.penalties) > 1:
            raise NotImplementedError(
                'bs="fs" currently requires a singly penalized base smooth.'
            )

        self._base_term = base_term
        B0, S0, _ = self._base_constructor_fit_matrices()
        B0 = np.asarray(B0, dtype=np.float64)
        S0 = np.asarray(S0, dtype=np.float64)

        base_rank = _penalty_rank_from_base_term(base_term, B0, S0)
        null_d = int(B0.shape[1] - base_rank)
        if null_d <= 0:
            raise NotImplementedError(_fs_full_rank_base_error(base_spec.bs))

        fac_idx = self._factor_feature_indices[0]
        fac_name = self._factor_feature_names[0]
        fac = as_object_1d(X[:, fac_idx])
        levels = stable_unique_levels(
            fac, levels=factor_levels_from_metadata(self.metadata, fac_name)
        )
        self._levels = levels

        Ifac = factor_indicator_matrix(fac, levels)
        n_levels = len(levels)

        if len(base_term.penalties) == 0:
            X_full = rowwise_kronecker([Ifac, B0])
            X_full = self._apply_cached_by(X_full)
            self._basis_train = np.asarray(X_full, dtype=np.float64)
            self._penalties = []
            self._smoothing_ids = []
            self._ranks = []
            self.skip_centering = True
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        # mgcv uses nat.param(X, S, rank, type=1): eigendecompose R^{-T} S R^{-1}
        # (R from QR of X) and normalise the range space to an identity penalty.
        rp = nat_param_type1(
            B0,
            S0,
            rank=base_rank,
            unit_fnorm=True,
        )
        X_reparam = rp["X"]  # (n, p0) reparameterised basis
        P_coef = rp["P"]  # (p0, p0) transform: B0 @ P_coef = X_reparam
        r = rp["rank"]  # penalty rank
        D = rp["D"]  # scale^2 * ones(r) after type=1 + unit_fnorm
        null_d = B0.shape[1] - r

        X_reparam, P_coef = self._align_multivariate_base_reparameterization(
            X,
            X_reparam,
            P_coef,
            range_rank=r,
            null_dim=null_d,
        )

        self._base_transform = P_coef
        self._base_range_penalty_diag = np.concatenate(
            [D, np.zeros(null_d, dtype=np.float64)]
        )
        self._range_rank = r
        self._null_dim = null_d

        # Build full design matrix: replicate reparameterised basis per factor level
        n = B0.shape[0]
        p0 = X_reparam.shape[1]
        X_full = np.zeros((n, p0 * n_levels), dtype=np.float64)
        for i, lev in enumerate(levels):
            mask = (fac == lev).astype(np.float64)
            X_full[:, i * p0 : (i + 1) * p0] = X_reparam * mask[:, None]

        # Build penalties matching mgcv's construction:
        #   S[[1]] = diag(rep(c(D, 0...0), nf))          (range space)
        #   S[[j+1]] = diag(rep(e_{r+j}, nf))  j=0..q-1  (null-space)
        d_vec = np.asarray(self._base_range_penalty_diag, dtype=np.float64)
        P_range = np.diag(np.tile(d_vec, n_levels))

        penalties = []
        smoothing_ids = []
        ranks = []
        n_penalties = 1 + int(null_d)

        def _fs_penalty_id(local_index: int) -> str | None:
            if self.smoothing_id is None:
                return None
            if n_penalties <= 1:
                return str(self.smoothing_id)
            return penalty_id_for_local_index(
                self.smoothing_id,
                local_index,
                n_penalties=n_penalties,
            )

        penalties.append(0.5 * (P_range + P_range.T))
        smoothing_ids.append(_fs_penalty_id(0))
        ranks.append(int(n_levels * r))

        for i in range(null_d):
            um = np.zeros(p0, dtype=np.float64)
            um[r + i] = 1.0
            P_null_i = np.diag(np.tile(um, n_levels))
            penalties.append(0.5 * (P_null_i + P_null_i.T))
            smoothing_ids.append(_fs_penalty_id(i + 1))
            ranks.append(int(n_levels))

        # Apply mgcv smoothCon scale_penalty step: normalise S relative to X.
        penalties, penalty_scales = rescale_tensor_penalties_for_fit(
            X_full, penalties, return_scales=True
        )

        X_full = self._apply_cached_by(X_full)
        self._basis_train = np.asarray(X_full, dtype=np.float64)
        self._penalties = penalties
        self._smoothing_ids = smoothing_ids
        self._ranks = ranks
        self._set_penalty_rescale_factors(penalty_scales)
        self.skip_centering = True
        self._record_constraint_result(None, None, absorbed_by=None)
        return self

    def transform_new(self, X_new):
        if self._delegate_term is not None:
            return self._delegate_term.transform_new(X_new)

        self._require_fitted()

        fac_idx = self._factor_feature_indices[0]
        fac = as_object_1d(column_as_object(X_new, fac_idx))
        Ifac = factor_indicator_matrix(fac, self._levels)

        B0_new = np.asarray(
            self._base_constructor_predict_matrix(X_new), dtype=np.float64
        )
        if self._base_transform is not None:
            B0_new = B0_new @ self._base_transform

        B_new = rowwise_kronecker([Ifac, B0_new])
        z = by_values_from_new_data(X_new, self._by_state)
        return np.asarray(
            self._apply_by_scale(B_new, z, allow_missing=True), dtype=np.float64
        )

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
                    metadata=self._penalty_metadata_with_scale(
                        {
                            "term_type": self.term_type,
                            "basis_name": self.basis_name,
                            "feature": list(self.feature),
                            "label": self.label,
                            "by": self.by,
                            "by_name": self._by_state.feature_name,
                            "factor_name": self._factor_feature_names[0],
                            "levels": list(self._levels),
                            "base_basis_name": (
                                self._base_term.basis_name
                                if self._base_term is not None
                                else None
                            ),
                            "base_metric_features": list(self._metric_feature_names),
                        },
                        penalty_index=j,
                    ),
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
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
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
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            select=select,
            xt=xt,
            fixed=fixed,
            knots=knots,
            metadata=metadata,
        )

        self._contrast_mats = None
        self._factor_transform = None
        self._n_groups = None

    def fit(self, X, feature_names):
        if self._build_delegate_base_or_re(
            X, feature_names, default_bs="tp", mode="sz"
        ):
            return self

        X = _as_object_2d(X)
        self._set_by_state(X, feature_names)

        base_spec = _parse_factor_smooth_xt(self.xt, default_bs="tp")
        base_term = _build_base_smooth_term(
            metric_features=self._metric_feature_names,
            k=self.k,
            base_bs=base_spec.bs,
            label=self.label,
            fixed=self.fixed,
            by=None,
            knots=self.knots,
            xt_rest=base_spec.xt_rest,
            mode="sz",
            select=False,
            constraint_mode="never",
            metadata=dict(self.metadata),
        )
        base_term.fit(X, feature_names)

        if len(base_term.penalties) > 1:
            raise NotImplementedError(
                'bs="sz" currently requires a singly penalized base smooth.'
            )

        self._base_term = base_term
        B0, S0, _ = self._base_constructor_fit_matrices()
        B0 = np.asarray(B0, dtype=np.float64)
        S0 = np.asarray(S0, dtype=np.float64)

        level_lists = []
        indicator_mats = []
        contrast_mats = []

        for idx in self._factor_feature_indices:
            fname = self._feature_names[self._feature_indices.index(idx)]
            fac = as_object_1d(X[:, idx])
            lev = stable_unique_levels(
                fac, levels=factor_levels_from_metadata(self.metadata, fname)
            )
            level_lists.append(lev)
            indicator_mats.append(factor_indicator_matrix(fac, lev))
            contrast_mats.append(_sum_to_zero_contrast(len(lev)))

        self._factor_levels = level_lists
        self._contrast_mats = contrast_mats

        X_raw = rowwise_kronecker(indicator_mats + [B0])

        p0 = B0.shape[1]
        level_sizes = [len(lev) for lev in level_lists]
        self._n_groups = int(np.prod(level_sizes, dtype=np.int64))
        T = _sum_to_zero_contrast_transform(level_sizes, p0)
        self._factor_transform = T

        if T.shape[1] == 0:
            self._basis_train = np.empty((X_raw.shape[0], 0), dtype=np.float64)
            self._penalties = []
            self._smoothing_ids = []
            self._ranks = []
            self.skip_centering = True
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

        X_con = X_raw @ T

        if self.fixed or len(base_term.penalties) == 0:
            X_con = self._apply_cached_by(X_con)
            self._basis_train = np.asarray(X_con, dtype=np.float64)
            self._penalties = []
            self._smoothing_ids = []
            self._ranks = []
            self.skip_centering = True
            self._record_constraint_result(None, None, absorbed_by=None)
            return self

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
            P_sum = np.zeros(
                (self._n_groups * p0, self._n_groups * p0), dtype=np.float64
            )
            for g in range(self._n_groups):
                P_sum += _block_penalty_for_group(g, self._n_groups, S0)
            P_con = T.T @ P_sum @ T
            P_con = 0.5 * (P_con + P_con.T)

            penalties.append(P_con)
            smoothing_ids.append(str(self.smoothing_id))
            ranks.append(int(np.linalg.matrix_rank(P_con)))

        # Apply mgcv smoothCon scale_penalty step: normalise S relative to X.
        # R applies this BEFORE the XZKr contrast, using the pre-contrast X_raw
        # and the pre-contrast block penalty S_raw (which has norm_1 = norm_1(S0)).
        # The scale factor is the same for all penalties (norm_1(S_raw[i]) = norm_1(S0)).
        if len(penalties) > 0:
            maXX = float(np.max(np.sum(np.abs(X_raw), axis=1)) ** 2)
            maS0 = float(np.max(np.sum(np.abs(S0), axis=0)))
            s_scale = 1.0
            if maS0 > 1e-12 and maXX > 1e-12:
                s_scale = maS0 / maXX
                scale = maXX / maS0
                penalties = [P * scale for P in penalties]
            self._set_penalty_rescale_factors([s_scale] * len(penalties))
        else:
            self._set_penalty_rescale_factors([])

        X_con = self._apply_cached_by(X_con)
        self._basis_train = np.asarray(X_con, dtype=np.float64)
        self._penalties = penalties
        self._smoothing_ids = smoothing_ids
        self._ranks = ranks
        self.skip_centering = True
        self._record_constraint_result(None, None, absorbed_by=None)
        return self

    def transform_new(self, X_new):
        if self._delegate_term is not None:
            return self._delegate_term.transform_new(X_new)

        self._require_fitted()

        indicator_mats = []
        for idx, lev in zip(
            self._factor_feature_indices,
            self._factor_levels,
            strict=True,
        ):
            fac = as_object_1d(column_as_object(X_new, idx))
            indicator_mats.append(factor_indicator_matrix(fac, lev))

        B0_new = np.asarray(self._base_term.transform_new(X_new), dtype=np.float64)
        X_raw = rowwise_kronecker(indicator_mats + [B0_new])

        if self._factor_transform.shape[1] == 0:
            return np.empty((X_raw.shape[0], 0), dtype=np.float64)

        B_new = X_raw @ self._factor_transform
        z = by_values_from_new_data(X_new, self._by_state)
        return np.asarray(
            self._apply_by_scale(B_new, z, allow_missing=True), dtype=np.float64
        )

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
                    metadata=self._penalty_metadata_with_scale(
                        {
                            "term_type": self.term_type,
                            "basis_name": self.basis_name,
                            "feature": list(self.feature),
                            "label": self.label,
                            "by": self.by,
                            "by_name": self._by_state.feature_name,
                            "factor_names": list(self._factor_feature_names),
                            "factor_levels": [list(lev) for lev in self._factor_levels],
                            "base_basis_name": (
                                self._base_term.basis_name
                                if self._base_term is not None
                                else None
                            ),
                            "base_metric_features": list(self._metric_feature_names),
                            "shared_smoothing_id": self.smoothing_id,
                        },
                        penalty_index=j,
                    ),
                )
            )

        if self.select and len(self._penalties) > 0:
            selection_meta = {
                "term_type": self.term_type,
                "basis_name": self.basis_name,
                "feature": list(self.feature),
                "label": self.label,
                "by": self.by,
                "by_name": self._by_state.feature_name,
                "factor_names": list(self._factor_feature_names),
                "factor_levels": [list(lev) for lev in self._factor_levels],
                "base_basis_name": (
                    self._base_term.basis_name if self._base_term is not None else None
                ),
                "base_metric_features": list(self._metric_feature_names),
                "shared_smoothing_id": self.smoothing_id,
            }
            defs.extend(
                self._build_selection_penalty_definitions(
                    [np.asarray(P, dtype=np.float64) for P in self._penalties],
                    selection_metadata=selection_meta,
                    fallback_selection_smoothing_id=f"__sz__:{self.label}:select",
                )
            )

        return defs
